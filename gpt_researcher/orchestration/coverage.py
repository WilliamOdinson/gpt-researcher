"""Coverage-based orchestration baselines: Table-2 rows "Greedy marginal
utility" (``GR_ORCHESTRATOR=greedy``) and "Heuristic stopping"
(``GR_ORCHESTRATOR=heuristic_stop``).

Both score evidence against the decomposed sub-questions
Q_sub = {q^(1), ..., q^(M)} through the coverage potential of PILOT Sec. 4.4,

    Phi(K) = (1/M) sum_k max_{c in K} cos(e(c), e(q^(k))),

and neither makes an LLM call. The only cost on top of the ``none`` row is
the one embedding call for the query and sub-questions at the start of the
run (``needs_embeddings``); everything else is numpy over cached vectors.

GreedyMarginalUtilityPolicy
    Budgeted greedy on a monotone submodular surrogate of report quality
    (Sec. 3.3). The max inside Phi saturates after at most M picks (M = 3
    with the default research plan), after which every remaining item has
    exactly zero marginal coverage. The greedy therefore ranks items with the
    saturating (noisy-OR) coverage

        Phi~(K) = (1/M) sum_k [ 1 - prod_{c in K} (1 - s_ck) ],
        s_ck    = max(0, cos(e(c), e(q^(k)))),

    whose per-sub-question gain decays geometrically instead of dropping to
    zero, and discounts it by the novelty feature of Sec. 4.2 (computed
    against the items already selected this round):

        v_i(K) = [ Phi~(K u {i}) - Phi~(K) ] * nu_i(K)^lambda,
        nu_i(K) = 1 - max_{j in K} cos(e(c_i), e(c_j)).

    Items are added in order of v_i while they fit the token budget; the
    loop stops when the budget is exhausted or max_i v_i(K) < eps. Every
    round re-selects from K_{t-1} u (new items), so earlier evidence can be
    superseded exactly as under ``topk``. Uniform branch allocation; never
    terminates early.

HeuristicStopPolicy
    Retention is the pipeline default: the per-sub-query EmbeddingsFilter
    verdict that ``GR_ORCHESTRATOR=legacy`` records, applied at chunk
    granularity (a kept page is trimmed to the chunks the filter kept).
    Previously retained items always survive. The policy only decides u:
    after ``min_rounds`` rounds it terminates once the coverage gain

        g_t = Phi(K_t) - Phi(K_{t-1})

    has stayed below ``threshold`` for ``patience`` consecutive rounds.
    ``retain="all"`` keeps every item instead (the ``none`` row plus the
    stopping rule), which isolates the effect of termination in Table 2.
    Uniform branch allocation.

Caveat for tuning ``heuristic_stop``: the fork expands the research tree
depth-first and a round is one node's children finishing, so consecutive
rounds are often narrow leaf sub-trees whose gain is exactly zero. A
``patience`` of 1 will then stop before sibling top-level branches have run
at all; sweep ``patience`` together with ``threshold``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .features import (
    FrontierRow,
    ItemRow,
    _dim_of,
    _unit_matrix,
    compute_features,
    coverage_potential_for,
)
from .policies import (
    OrchestrationDecision,
    OrchestrationInput,
    OrchestrationPolicy,
    PoolItem,
    estimate_tokens,
    uniform_allocation,
)

logger = logging.getLogger(__name__)

STOP_RETAIN_MODES = ("filter", "all")


def _rows(items: list[PoolItem]) -> list[ItemRow]:
    return [
        ItemRow(
            item_id=it.item_id,
            source_url=it.source_url,
            text=it.text,
            embedding=it.embedding,
            tree_depth=it.tree_depth,
            retrieval_round=it.retrieval_round,
            is_new=it.is_new,
            source_subquery=it.source_subquery or "",
        )
        for it in items
    ]


def _bundle(inp: OrchestrationInput, rows: list[ItemRow]):
    return compute_features(
        query_embedding=inp.query_embedding,
        subquestion_embeddings=inp.subquestion_embeddings,
        items=rows,
        frontier=[FrontierRow(f.node_id, f.subquery, f.embedding) for f in inp.frontier],
        round_id=inp.round_id,
    )


def soft_coverage(S: np.ndarray) -> float:
    """Phi~ of the item set whose rows make up ``S`` (n, M): mean over
    sub-questions of 1 - prod_i (1 - s_ik)."""
    if S.size == 0:
        return 0.0
    return float(np.mean(1.0 - np.prod(1.0 - S, axis=0)))


# --------------------------------------------------------------------------
# greedy
# --------------------------------------------------------------------------


class GreedyMarginalUtilityPolicy(OrchestrationPolicy):
    """Row "Greedy marginal utility": budgeted greedy on the saturating
    sub-question coverage, discounted by novelty against the items already
    selected. See the module docstring for the objective."""

    name = "greedy"
    needs_embeddings = True

    def __init__(
        self,
        budget: int | None = None,
        min_gain: float = 0.0,
        redundancy: float = 1.0,
    ) -> None:
        if min_gain < 0.0:
            raise ValueError(f"min_gain must be >= 0, got {min_gain}")
        if redundancy < 0.0:
            raise ValueError(f"redundancy exponent must be >= 0, got {redundancy}")
        self.budget = budget
        self.min_gain = float(min_gain)
        self.redundancy = float(redundancy)

    async def decide(self, inp: OrchestrationInput) -> OrchestrationDecision:
        pool = inp.pool
        n = len(pool)
        rows = _rows(pool)
        bundle = _bundle(inp, rows)
        budget = self.budget if self.budget is not None else inp.token_budget

        dim = _dim_of(rows, inp.subquestion_embeddings, [inp.query_embedding])
        mat, have = _unit_matrix([it.embedding for it in pool], dim)
        sq, sq_have = _unit_matrix(list(inp.subquestion_embeddings), dim)
        q, q_have = _unit_matrix([inp.query_embedding], dim)
        m = int(sq_have.sum())

        rho = np.clip((mat @ q.T)[:, 0], 0.0, 1.0) if q_have.any() else np.zeros(n, dtype=np.float32)
        # (n, n) cosine between items; rows without an embedding are all zero,
        # so such items are never counted as redundant with anything.
        sim = np.clip(mat @ mat.T, 0.0, 1.0) if n else np.zeros((0, 0), dtype=np.float32)
        np.fill_diagonal(sim, 0.0)

        # Coverage signal. Without sub-question vectors (plan or embedding
        # failure) fall back to root-query relevance, and without that to a
        # constant so the greedy degrades to "fill the budget" like ``topk``
        # with zero scores rather than to keeping a single item.
        if m:
            S = np.clip(mat @ sq[sq_have].T, 0.0, 1.0)
            S[~have] = 0.0
            coverage_source = "subquestions"
        elif q_have.any():
            S = rho[:, None].astype(np.float32)
            S[~have] = 0.0
            coverage_source = "root_query"
        else:
            S = np.ones((n, 1), dtype=np.float32)
            coverage_source = "none"
        residual = np.ones(S.shape[1], dtype=np.float64)
        # A constant "coverage" carries no information about saturation, so
        # in that case the residual is left at 1 and the budget alone binds.
        saturating = coverage_source != "none"

        tokens = np.asarray([it.tokens for it in pool], dtype=np.int64)
        selected: list[int] = []
        selected_mask = np.zeros(n, dtype=bool)
        feasible = np.ones(n, dtype=bool)
        red = np.zeros(n, dtype=np.float64)  # max cos to the selected set
        pick_values: dict[str, float] = {}
        used = 0
        stop_reason = "exhausted"

        while True:
            candidates = np.flatnonzero(~selected_mask & feasible)
            if candidates.size == 0:
                break
            gains = (S * residual[None, :]).mean(axis=1)
            novelty = np.clip(1.0 - red, 0.0, 1.0)
            values = gains * np.power(novelty, self.redundancy)
            # Ties: newer evidence, then closer to the root query, then pool order.
            best = max(
                candidates,
                key=lambda i: (float(values[i]), pool[i].retrieval_round, float(rho[i]), -int(i)),
            )
            value = float(values[best])
            if value <= 0.0:
                stop_reason = "no_positive_gain"
                break
            if value < self.min_gain:
                stop_reason = "min_gain"
                break
            if budget is not None and used + int(tokens[best]) > budget:
                feasible[best] = False
                continue
            selected.append(int(best))
            selected_mask[best] = True
            used += int(tokens[best])
            pick_values[pool[best].item_id] = value
            if saturating:
                residual *= 1.0 - S[best].astype(np.float64)
            red = np.maximum(red, sim[:, best].astype(np.float64))

        if (~feasible).any():
            # The budget excluded at least one positive-value item, so it is
            # the binding constraint even if the loop later ran out of gain.
            stop_reason = "budget"

        if not selected and n:
            # Mirror ``topk``: never hand the next round an empty context.
            gains0 = S.mean(axis=1)
            best = max(range(n), key=lambda i: (float(gains0[i]), float(rho[i]), pool[i].retrieval_round, -i))
            selected.append(best)
            selected_mask[best] = True
            used = int(tokens[best])
            pick_values[pool[best].item_id] = float(gains0[best])
            stop_reason = "forced_min_keep"

        kept = {pool[i].item_id for i in selected}
        phi_after = coverage_potential_for(kept, rows, inp.subquestion_embeddings)

        meta: dict[str, Any] = {
            "policy": self.name,
            "budget": budget,
            "min_gain": self.min_gain,
            "lambda": self.redundancy,
            "coverage_source": coverage_source,
            "n_pool": n,
            "n_new": len(inp.new_items),
            "n_selected": len(selected),
            "tokens_kept": used,
            "order": [pool[i].item_id for i in selected],
            "values": {k: round(v, 5) for k, v in pick_values.items()},
            "stop_reason": stop_reason,
            "phi_after": round(float(phi_after), 4),
            "soft_phi_after": round(soft_coverage(S[selected_mask]) if m else 0.0, 4),
        }
        meta.update(bundle.as_meta())

        return OrchestrationDecision(
            terminate=False,
            kept_ids=kept,
            branch_allocation=uniform_allocation(inp.frontier),
            meta=meta,
        )


# --------------------------------------------------------------------------
# heuristic stopping
# --------------------------------------------------------------------------


class HeuristicStopPolicy(OrchestrationPolicy):
    """Row "Heuristic stopping": default retention, terminate when the
    per-round coverage gain g_t = Phi(K_t) - Phi(K_{t-1}) stays below
    ``threshold`` for ``patience`` consecutive rounds (evaluated from round
    ``min_rounds`` on)."""

    name = "heuristic_stop"
    needs_embeddings = True

    def __init__(
        self,
        threshold: float = 0.01,
        min_rounds: int = 2,
        patience: int = 1,
        retain: str = "filter",
    ) -> None:
        if retain not in STOP_RETAIN_MODES:
            raise ValueError(f"retain must be one of {STOP_RETAIN_MODES}, got {retain!r}")
        self.threshold = float(threshold)
        self.min_rounds = max(1, int(min_rounds))
        self.patience = max(1, int(patience))
        self.retain = retain
        # Consecutive rounds (from min_rounds on) with g_t < threshold.
        self.low_streak = 0
        # (round_id, phi_prev, phi_after, gain) per decision, for analysis.
        self.history: list[tuple[int, float, float, float]] = []

    @staticmethod
    def _filter_text(item: PoolItem) -> str | None:
        """The page trimmed to the chunks the EmbeddingsFilter kept, or
        ``None`` when nothing changes (no chunk detail, or all chunks kept)."""
        if not item.chunks or len(item.chunk_kept) != len(item.chunks):
            return None
        if all(item.chunk_kept) or not any(item.chunk_kept):
            return None
        return "\n\n".join(c for c, k in zip(item.chunks, item.chunk_kept) if k)

    async def decide(self, inp: OrchestrationInput) -> OrchestrationDecision:
        pool = inp.pool
        rows = _rows(pool)
        bundle = _bundle(inp, rows)

        # m: previously retained items stay; new items follow the filter.
        kept: set[str] = {it.item_id for it in inp.retained_prev}
        rewritten: dict[str, str] = {}
        compressed_chars: dict[str, tuple[int, int]] = {}
        n_filter_pruned = 0
        for it in inp.new_items:
            if self.retain == "all" or it.filter_kept:
                kept.add(it.item_id)
                if self.retain == "filter":
                    text = self._filter_text(it)
                    if text is not None:
                        rewritten[it.item_id] = text
                        compressed_chars[it.item_id] = (len(it.text), len(text))
            else:
                n_filter_pruned += 1

        # u: coverage gain of this round's retained set over K_{t-1}.
        phi_prev = float(bundle.phi_prev)
        phi_after = float(coverage_potential_for(kept, rows, inp.subquestion_embeddings))
        gain = phi_after - phi_prev
        self.history.append((inp.round_id, phi_prev, phi_after, gain))

        stop_enabled = bool(inp.subquestion_embeddings) and any(it.embedding for it in pool)
        terminate = False
        reason = ""
        if not stop_enabled:
            reason = "stop_disabled_no_embeddings"
            logger.warning(
                "heuristic_stop: no sub-question or evidence embeddings at round %s; "
                "the stopping rule is disabled for this round",
                inp.round_id,
            )
        elif inp.round_id >= self.min_rounds:
            if gain < self.threshold:
                self.low_streak += 1
            else:
                self.low_streak = 0
            if self.low_streak >= self.patience:
                terminate = True
                reason = "coverage_gain_below_threshold"

        meta: dict[str, Any] = {
            "policy": self.name,
            "retain": self.retain,
            "threshold": self.threshold,
            "min_rounds": self.min_rounds,
            "patience": self.patience,
            "phi_after": round(phi_after, 4),
            "gain": round(gain, 5),
            "low_streak": self.low_streak,
            "stop_enabled": stop_enabled,
            "terminate_reason": reason,
            "n_pool": len(pool),
            "n_new": len(inp.new_items),
            "n_filter_pruned": n_filter_pruned,
            "tokens_kept": sum(
                estimate_tokens(rewritten.get(it.item_id, it.text)) for it in pool if it.item_id in kept
            ),
            "compressed_chars": compressed_chars,
        }
        meta.update(bundle.as_meta())

        return OrchestrationDecision(
            terminate=terminate,
            kept_ids=kept,
            branch_allocation=uniform_allocation(inp.frontier),
            rewritten=rewritten,
            meta=meta,
        )

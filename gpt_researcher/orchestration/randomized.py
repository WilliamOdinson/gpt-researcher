"""RandomizedPolicy: the data-collection policy for PILOT behavior cloning.

The five Table-4 baselines are deterministic given the state (``topk`` always
ranks by cosine, ``none`` always keeps everything), so their trajectories have
zero action diversity and nothing for BC to learn from. This policy instead
samples its *orchestration parameters* once per query and then acts on them
every round, so a corpus of rollouts spans aggressive pruning to full
retention, skewed to uniform branch allocation, and early to late stopping
(PILOT §4.5 "Randomized rollouts").

Decisions are perturbed heuristics rather than uniform noise so that the top
decile of rollouts is worth cloning:

  m  keep mask     score_i = rho + w_nov*nov - w_red*red + w_cov*cov - w_len*tok_norm
                   + Gumbel(temperature); keep the top ceil(keep_ratio * n_t)
                   (and, if a token budget is set, greedily under budget_frac * B)
  w  allocation    Dirichlet(concentration * (0.25 + gap_v)) over open branches
  u  terminate     after ``min_rounds``: Bernoulli(p_terminate) OR the coverage
                   potential gain Phi(C_t) - Phi(K_{t-1}) fell below ``phi_gain_stop``

Everything needed to rebuild the BC example offline is written to
``decision.meta``: the sampled ``params``, the per-item ``features`` (see
``features.FEATURE_NAMES``), frontier stats, Phi before/after, scores, and the
seed. Set ``GR_ORCH_SEED`` for reproducible rollouts; the per-query seed is
derived from it and the root query. ``GR_RANDOM_PARAMS='{"keep_ratio": 0.5}'``
pins any subset of parameters (useful for counterfactual forks and tests).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from .features import ItemRow, FrontierRow, compute_features, coverage_potential_for
from .policies import (
    OrchestrationDecision,
    OrchestrationInput,
    OrchestrationPolicy,
    PoolItem,
    uniform_allocation,
)

# (name, sampler) — samplers take a numpy Generator. Kept in one table so the
# sampled space is easy to audit and to widen/narrow.
PARAM_SPACE: dict[str, Any] = {
    "keep_ratio": lambda r: float(r.uniform(0.15, 1.0)),
    "budget_frac": lambda r: float(r.uniform(0.3, 1.0)),
    "temperature": lambda r: float(math.exp(r.uniform(math.log(0.02), math.log(0.5)))),
    "w_nov": lambda r: float(r.uniform(0.0, 1.0)),
    "w_red": lambda r: float(r.uniform(0.0, 1.0)),
    "w_cov": lambda r: float(r.uniform(0.0, 1.0)),
    "w_len": lambda r: float(r.uniform(0.0, 0.5)),
    "alloc_concentration": lambda r: float(math.exp(r.uniform(math.log(0.5), math.log(20.0)))),
    "min_rounds": lambda r: int(r.integers(1, 4)),  # 1..3
    "p_terminate": lambda r: float(r.uniform(0.0, 0.25)),
    "phi_gain_stop": lambda r: float(r.uniform(0.0, 0.05)),
}


def derive_seed(base_seed: int | None, root_query: str) -> int:
    """Stable per-query seed so a rollout is reproducible from (seed, query)."""
    base = 0 if base_seed is None else int(base_seed)
    digest = hashlib.sha256(f"{base}|{root_query.strip()}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def sample_params(rng: np.random.Generator, fixed: dict[str, Any] | None = None) -> dict[str, Any]:
    params = {name: fn(rng) for name, fn in PARAM_SPACE.items()}
    for key, value in (fixed or {}).items():
        if key not in PARAM_SPACE:
            raise ValueError(f"unknown RandomizedPolicy parameter {key!r}; known: {sorted(PARAM_SPACE)}")
        params[key] = type(params[key])(value)
    return params


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


class RandomizedPolicy(OrchestrationPolicy):
    name = "random"
    needs_embeddings = True
    needs_frontier_embeddings = True

    def __init__(
        self,
        seed: int | None = None,
        budget: int | None = None,
        fixed_params: dict[str, Any] | None = None,
    ) -> None:
        self.base_seed = seed
        self.budget = budget
        self.fixed_params = dict(fixed_params or {})
        # Lazily initialised on the first decision so the per-query seed can
        # include the root query.
        self._rng: np.random.Generator | None = None
        self._query_seed: int | None = None
        self.params: dict[str, Any] | None = None

    # -- setup -----------------------------------------------------------

    def _ensure_rng(self, root_query: str) -> np.random.Generator:
        if self._rng is None:
            self._query_seed = derive_seed(self.base_seed, root_query)
            self._rng = np.random.default_rng(self._query_seed)
            self.params = sample_params(self._rng, self.fixed_params)
        return self._rng

    # -- decision --------------------------------------------------------

    async def decide(self, inp: OrchestrationInput) -> OrchestrationDecision:
        rng = self._ensure_rng(inp.root_query)
        p = self.params or {}
        pool = inp.pool
        n = len(pool)

        rows = _rows(pool)
        frontier_rows = [FrontierRow(f.node_id, f.subquery, f.embedding) for f in inp.frontier]
        bundle = compute_features(
            query_embedding=inp.query_embedding,
            subquestion_embeddings=inp.subquestion_embeddings,
            items=rows,
            frontier=frontier_rows,
            round_id=inp.round_id,
        )

        # ---- m: perturbed heuristic ranking --------------------------------
        scores: dict[str, float] = {}
        max_tok = max([bundle.features[it.item_id][4] for it in pool] or [1.0])
        for it in pool:
            rho, nov, red, cov, tok, _depth, _age, _src = bundle.features[it.item_id]
            base = (
                rho
                + p["w_nov"] * nov
                - p["w_red"] * red
                + p["w_cov"] * cov
                - p["w_len"] * (tok / max_tok if max_tok > 0 else 0.0)
            )
            gumbel = -math.log(-math.log(max(1e-12, float(rng.uniform()))))
            scores[it.item_id] = float(base + p["temperature"] * gumbel)

        ranked = sorted(pool, key=lambda it: (-scores[it.item_id], -it.retrieval_round))
        keep_target = max(1, int(math.ceil(p["keep_ratio"] * n))) if n else 0

        budget = self.budget if self.budget is not None else inp.token_budget
        token_cap = int(budget * p["budget_frac"]) if budget else None

        kept: set[str] = set()
        used = 0
        for it in ranked:
            if len(kept) >= keep_target:
                break
            if token_cap is not None and kept and used + it.tokens > token_cap:
                continue
            kept.add(it.item_id)
            used += it.tokens

        # ---- w: Dirichlet around the coverage-gap prior ---------------------
        if inp.frontier:
            gaps = np.asarray(
                [bundle.frontier.get(f.node_id, {}).get("gap", 1.0) for f in inp.frontier], dtype=float
            )
            alpha = p["alloc_concentration"] * (0.25 + gaps)
            w = rng.dirichlet(alpha)
            allocation = {f.node_id: float(x) for f, x in zip(inp.frontier, w)}
        else:
            allocation = uniform_allocation(inp.frontier)

        # ---- u: stop rule ------------------------------------------------------
        phi_after = coverage_potential_for(kept, rows, inp.subquestion_embeddings)
        phi_gain = bundle.phi_pool - bundle.phi_prev
        terminate = False
        reason = ""
        if inp.round_id >= p["min_rounds"]:
            if rng.uniform() < p["p_terminate"]:
                terminate, reason = True, "bernoulli"
            elif inp.round_id > 1 and phi_gain < p["phi_gain_stop"]:
                terminate, reason = True, "phi_gain_below_threshold"

        meta: dict[str, Any] = {
            "policy": self.name,
            "seed": self.base_seed,
            "query_seed": self._query_seed,
            "params": dict(p),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "keep_target": keep_target,
            "token_cap": token_cap,
            "tokens_kept": used,
            "n_pool": n,
            "n_new": len(inp.new_items),
            "tokens_used": inp.tokens_used,
            "token_budget": budget,
            "phi_after": round(float(phi_after), 4),
            "phi_gain": round(float(phi_gain), 4),
            "terminate_reason": reason,
        }
        meta.update(bundle.as_meta())

        return OrchestrationDecision(
            terminate=terminate,
            kept_ids=kept,
            branch_allocation=allocation,
            meta=meta,
        )

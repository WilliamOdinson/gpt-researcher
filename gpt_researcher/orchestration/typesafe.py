"""TypeSafe orchestration (``GR_ORCHESTRATOR=typesafe``): the runtime
orchestrator of the "prompted" row rebuilt on TypeSafe's System One model,
Jev (https://docs.typesafe.ai). Jev does not generate text: it evaluates
typed questions against a state and returns calibrated probabilities, so the
orchestration action is assembled in code from per-item and per-sub-question
judgments instead of parsed out of a generated KEEP / ALLOC / DECISION string.

What the model judges (three Nouls per evidence item, once per item):

  useful_i       does the passage state something a report could cite?
  boilerplate_i  is the passage mostly navigation, notices, or link lists?
  sub_i_k        does the passage help answer decomposed sub-question k?

Items are judged in batches of ``batch_size`` (one request per batch, all
questions of the batch in that request), and the verdict is cached per item
id: Jev's judgments are absolute per passage and the question set is fixed
for the whole run, so a retained item is never re-sent. Everything that code
can compute exactly stays in code, as the TypeSafe guide recommends:

  m  keep mask     prune boilerplate and low-usefulness items; order the rest
                   coverage-first (top ``support_k`` supporters per
                   sub-question, round robin) then by usefulness; fill the
                   token budget greedily, skipping near-duplicates by
                   embedding cosine (``GR_FEAT_TAU``)
  u  terminate     one request per round asks, for each sub-question, whether
                   its best supporting passages already answer it; terminate
                   once every sub-question is answered with p >= ``stop_min``
                   after ``min_rounds`` rounds
  w  allocation    the same request asks, per open branch, whether its
                   sub-query is already answered by the closest retained
                   passages; effort goes to branches that are not

Cost is metered like the prompted row: every response's ``usage`` is
recorded under ``usage_tag="orchestrator"`` (model ``typesafe:<id>``, USD at
the published input-token price).

Environment (all optional except the key):

  TYPESAFE_API_KEY               required (read by the SDK)
  GR_TYPESAFE_MODEL              default ``jev-1.13.0`` (pinned: aliases move)
  GR_TYPESAFE_KEEP_MIN           p(useful) floor to keep an item (0.5)
  GR_TYPESAFE_BOILERPLATE_MAX    p(boilerplate) ceiling (0.7)
  GR_TYPESAFE_STOP_MIN           per-sub-question answered floor to stop (0.75)
  GR_TYPESAFE_MIN_ROUNDS         rounds before termination is considered (2)
  GR_TYPESAFE_SUPPORT_K          passages per sub-question / branch (3)
  GR_TYPESAFE_BATCH              evidence items per judging request (20)
  GR_TYPESAFE_SNIPPET_CHARS      characters of each passage sent (600)
  GR_CONTEXT_BUDGET_TOKENS       retained-evidence budget (shared)
  GR_FEAT_TAU                    near-duplicate cosine (shared, 0.85)

Requires ``pip install typesafe-sdk`` (the ``orchestration`` extra).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable
from urllib.parse import urlparse

import numpy as np

from .features import _dim_of, _unit_matrix
from .policies import (
    OrchestrationDecision,
    OrchestrationInput,
    OrchestrationPolicy,
    PoolItem,
    uniform_allocation,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "jev-1.13.0"
# Input tokens only; output tokens are free (docs.typesafe.ai/models).
JEV_USD_PER_MTOK = 0.042
_MAX_CONCURRENT_REQUESTS = 4

ENV_MODEL = "GR_TYPESAFE_MODEL"
ENV_KEEP_MIN = "GR_TYPESAFE_KEEP_MIN"
ENV_BOILERPLATE_MAX = "GR_TYPESAFE_BOILERPLATE_MAX"
ENV_STOP_MIN = "GR_TYPESAFE_STOP_MIN"
ENV_MIN_ROUNDS = "GR_TYPESAFE_MIN_ROUNDS"
ENV_SUPPORT_K = "GR_TYPESAFE_SUPPORT_K"
ENV_BATCH = "GR_TYPESAFE_BATCH"
ENV_SNIPPET_CHARS = "GR_TYPESAFE_SNIPPET_CHARS"
ENV_DEDUP_TAU = "GR_FEAT_TAU"

Meter = Callable[[str, int, int], None]


@dataclass(frozen=True)
class TypeSafeThresholds:
    """Every number the routing reads, in one place (see the "classifying
    RAG passages" cookbook): a change of policy is a constant edit."""

    keep_min: float = 0.5
    boilerplate_max: float = 0.7
    stop_min: float = 0.75
    min_rounds: int = 2
    support_k: int = 3
    batch_size: int = 20
    snippet_chars: int = 600
    dedup_tau: float = 0.85
    timeout: float = 60.0

    def __post_init__(self) -> None:
        for name in ("keep_min", "boilerplate_max", "stop_min", "dedup_tau"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        for name in ("min_rounds", "support_k", "batch_size", "snippet_chars"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


def thresholds_from_env() -> TypeSafeThresholds:
    def _f(name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        return float(raw) if raw else default

    def _i(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        return int(raw) if raw else default

    return TypeSafeThresholds(
        keep_min=_f(ENV_KEEP_MIN, 0.5),
        boilerplate_max=_f(ENV_BOILERPLATE_MAX, 0.7),
        stop_min=_f(ENV_STOP_MIN, 0.75),
        min_rounds=_i(ENV_MIN_ROUNDS, 2),
        support_k=_i(ENV_SUPPORT_K, 3),
        batch_size=_i(ENV_BATCH, 20),
        snippet_chars=_i(ENV_SNIPPET_CHARS, 600),
        dedup_tau=_f(ENV_DEDUP_TAU, 0.85),
    )


@dataclass
class ItemVerdict:
    useful: float
    boilerplate: float
    support: list[float]  # one probability per decomposed sub-question
    # The judging request failed; the item is kept unconditionally this run.
    fallback: bool = False


@dataclass
class TypeSafeUsage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0
    models: set[str] = field(default_factory=set)

    @property
    def usd(self) -> float:
        return self.input_tokens * JEV_USD_PER_MTOK / 1e6

    def as_meta(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "errors": self.errors,
            "usd": round(self.usd, 8),
            "models": sorted(self.models),
        }


def _snippet(text: str, n: int) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:n]


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc or url
    except Exception:  # pragma: no cover - defensive
        return url


def _noul(answer: Any) -> float:
    """Read a Noul probability from an SDK answer object or a raw dict."""
    value = getattr(answer, "noul", None)
    if value is None and isinstance(answer, dict):
        value = answer.get("noul")
    if value is None:
        raise ValueError(f"answer has no noul value: {answer!r}")
    return float(min(1.0, max(0.0, value)))


def _default_meter(model: str, input_tokens: int, output_tokens: int) -> None:
    from ..utils.token_tracker import TokenTracker

    TokenTracker.track_tokens(
        f"typesafe:{model}",
        int(input_tokens),
        int(output_tokens),
        cost=int(input_tokens) * JEV_USD_PER_MTOK / 1e6,
        usage_tag="orchestrator",
    )


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------
# Instructions name the state field they judge with a backticked path and put
# the boundary cases in the criteria, per docs.typesafe.ai/concepts/how-to-build-with-system-one.


def item_questions(index: int, n_subquestions: int) -> dict[str, dict[str, Any]]:
    path = f"`evidence[{index}].text`"
    q: dict[str, dict[str, Any]] = {
        f"useful_{index}": {
            "type": "noul",
            "instructions": (
                f"Does {path} state specific facts, figures, dates, names, or explanations "
                "that a research report answering `question` could cite?"
            ),
            "criteria": {
                "true": "Contains at least one concrete, citable statement about the subject of `question`",
                "false": "Off-topic, or only generic or promotional statements with nothing concrete to cite",
            },
        },
        f"boilerplate_{index}": {
            "type": "noul",
            "instructions": f"Is {path} mostly website boilerplate rather than article content?",
            "criteria": {
                "true": (
                    "Navigation menus, cookie or sign-in notices, link lists, headers and footers, "
                    "or error pages make up most of the text"
                ),
                "false": "Mostly prose, data, or discussion about a subject",
            },
        },
    }
    for k in range(n_subquestions):
        q[f"sub_{index}_{k}"] = {
            "type": "noul",
            "instructions": f"Does {path} contain information that helps answer `subquestions[{k}]`?",
            "criteria": {
                "true": f"States something that directly bears on `subquestions[{k}]`",
                "false": f"Does not address `subquestions[{k}]`",
            },
        }
    return q


def subquestion_answered_question(k: int) -> dict[str, Any]:
    return {
        "type": "noul",
        "instructions": (
            f"Do the passages in `subquestions[{k}].evidence` contain enough specific information "
            f"to write a report section that answers `subquestions[{k}].question`?"
        ),
        "criteria": {
            "true": "The passages together answer the question with specifics (facts, figures, or explanations)",
            "false": "The passages are missing, vague, or cover only part of the question",
        },
    }


def branch_answered_question(j: int) -> dict[str, Any]:
    return {
        "type": "noul",
        "instructions": f"Do the passages in `branches[{j}].evidence` already answer `branches[{j}].subquery`?",
        "criteria": {
            "true": "The answer to the sub-query is already stated in the passages",
            "false": "The passages do not answer the sub-query, or there are no passages",
        },
    }


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------


class TypeSafePolicy(OrchestrationPolicy):
    """Runtime orchestration with TypeSafe's Jev; see the module docstring."""

    name = "typesafe"
    needs_embeddings = True
    needs_frontier_embeddings = True

    def __init__(
        self,
        *,
        budget: int | None = None,
        model: str | None = None,
        thresholds: TypeSafeThresholds | None = None,
        client: Any | None = None,
        meter: Meter | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.budget = budget
        self.model = (model or "").strip() or DEFAULT_MODEL
        self.t = thresholds or TypeSafeThresholds()
        self._client = client
        self._meter = meter or _default_meter
        self._api_key = api_key
        self._base_url = base_url
        self._verdicts: dict[str, ItemVerdict] = {}
        self._n_subquestions: int | None = None
        self.usage = TypeSafeUsage()
        if client is None:
            self._require_sdk()

    def _require_sdk(self) -> None:
        try:
            import typesafe_sdk  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "GR_ORCHESTRATOR=typesafe needs the typesafe-sdk package: "
                "pip install typesafe-sdk (or pip install -e '.[orchestration]')"
            ) from exc
        if not (self._api_key or os.environ.get("TYPESAFE_API_KEY", "").strip()):
            raise RuntimeError("GR_ORCHESTRATOR=typesafe needs TYPESAFE_API_KEY in the environment")

    # ------------------------------------------------------------- transport

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[Any]:
        if self._client is not None:
            yield self._client
            return
        from typesafe_sdk import AsyncTypeSafeClient

        async with AsyncTypeSafeClient(
            api_key=self._api_key,
            base_url=self._base_url,
            model=self.model,
            timeout=self.t.timeout,
        ) as client:
            yield client

    async def _ask(self, client: Any, state: Any, questions: dict[str, Any]) -> Any | None:
        """One System One request; failures are logged, counted, and return
        ``None`` so the round degrades instead of crashing the run."""
        try:
            resp = await client.system_one(state=state, questions=questions, model=self.model)
        except Exception as exc:
            self.usage.errors += 1
            logger.warning("typesafe orchestrator request failed (%d questions): %s", len(questions), exc)
            return None
        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        model = str(getattr(resp, "model", None) or self.model)
        self.usage.requests += 1
        self.usage.input_tokens += in_tok
        self.usage.output_tokens += out_tok
        self.usage.models.add(model)
        try:
            self._meter(model, in_tok, out_tok)
        except Exception as exc:  # pragma: no cover - metering must never break a round
            logger.debug("typesafe usage metering failed: %s", exc)
        return resp

    # ------------------------------------------------------------- judging

    async def _judge(self, client: Any, inp: OrchestrationInput, items: list[PoolItem]) -> None:
        """Fill ``self._verdicts`` for ``items`` (batched, concurrent)."""
        subqs = list(inp.subquestions)
        batches = [items[i : i + self.t.batch_size] for i in range(0, len(items), self.t.batch_size)]
        sem = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)

        async def one(batch: list[PoolItem]) -> None:
            state = {
                "question": inp.root_query,
                "subquestions": subqs,
                "evidence": [
                    {"source": _domain(it.source_url), "text": _snippet(it.text, self.t.snippet_chars)}
                    for it in batch
                ],
            }
            questions: dict[str, Any] = {}
            for i in range(len(batch)):
                questions.update(item_questions(i, len(subqs)))
            async with sem:
                resp = await self._ask(client, state, questions)
            for i, it in enumerate(batch):
                if resp is None:
                    self._verdicts[it.item_id] = ItemVerdict(1.0, 0.0, [0.0] * len(subqs), fallback=True)
                    continue
                answers = resp.answers
                try:
                    self._verdicts[it.item_id] = ItemVerdict(
                        useful=_noul(answers[f"useful_{i}"]),
                        boilerplate=_noul(answers[f"boilerplate_{i}"]),
                        support=[_noul(answers[f"sub_{i}_{k}"]) for k in range(len(subqs))],
                    )
                except (KeyError, ValueError) as exc:
                    logger.warning("typesafe answer missing for item %s: %s", it.item_id, exc)
                    self._verdicts[it.item_id] = ItemVerdict(1.0, 0.0, [0.0] * len(subqs), fallback=True)

        await asyncio.gather(*(one(b) for b in batches))

    # ------------------------------------------------------------- selection

    def _select(
        self,
        inp: OrchestrationInput,
        unit: np.ndarray,
        have: np.ndarray,
    ) -> tuple[list[int], dict[str, str]]:
        """Keep mask in code from cached verdicts: filters, coverage-first
        ordering, budget fill with near-duplicate skipping."""
        pool = inp.pool
        t = self.t
        m = len(inp.subquestions)
        budget = self.budget if self.budget is not None else inp.token_budget
        reasons: dict[str, str] = {}

        candidates: list[int] = []
        for i, it in enumerate(pool):
            v = self._verdicts[it.item_id]
            if v.fallback:
                candidates.append(i)
            elif v.boilerplate >= t.boilerplate_max:
                reasons[it.item_id] = "boilerplate"
            elif v.useful < t.keep_min:
                reasons[it.item_id] = "low_usefulness"
            else:
                candidates.append(i)

        def useful(i: int) -> float:
            return self._verdicts[pool[i].item_id].useful

        order: list[int] = []
        seen: set[int] = set()
        if m:
            per_k = [
                sorted(
                    (i for i in candidates if self._verdicts[pool[i].item_id].support[k] >= t.keep_min),
                    key=lambda i, k=k: (-self._verdicts[pool[i].item_id].support[k], -useful(i), i),
                )
                for k in range(m)
            ]
            for _ in range(t.support_k):
                for k in range(m):
                    for i in per_k[k]:
                        if i not in seen:
                            order.append(i)
                            seen.add(i)
                            break
        for i in sorted(candidates, key=lambda i: (-useful(i), -pool[i].retrieval_round, i)):
            if i not in seen:
                order.append(i)
                seen.add(i)

        kept: list[int] = []
        used = 0
        for i in order:
            if kept and have[i]:
                sims = unit[kept] @ unit[i]
                if float(sims.max()) > t.dedup_tau:
                    reasons[pool[i].item_id] = "duplicate"
                    continue
            tok = pool[i].tokens
            if budget is not None and used + tok > budget:
                reasons[pool[i].item_id] = "budget"
                continue
            kept.append(i)
            used += tok

        if not kept and pool:
            # Mirror ``topk``: never hand the next round an empty context.
            best = max(range(len(pool)), key=lambda i: (useful(i), pool[i].retrieval_round, -i))
            kept.append(best)
            reasons.pop(pool[best].item_id, None)
        return kept, reasons

    # ------------------------------------------------------------- stop / alloc

    def _stop_state(
        self,
        inp: OrchestrationInput,
        kept: list[int],
        unit: np.ndarray,
        have: np.ndarray,
        dim: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        pool = inp.pool
        t = self.t
        subq_entries = []
        for k, q in enumerate(inp.subquestions):
            ranked = sorted(
                (i for i in kept if self._verdicts[pool[i].item_id].support[k] > 0.0),
                key=lambda i, k=k: (-self._verdicts[pool[i].item_id].support[k], i),
            )[: t.support_k]
            subq_entries.append(
                {"question": q, "evidence": [_snippet(pool[i].text, t.snippet_chars) for i in ranked]}
            )
        branch_entries = []
        kept_with_emb = [i for i in kept if have[i]]
        for f in inp.frontier:
            evidence: list[str] = []
            if f.embedding and kept_with_emb:
                fv, fhave = _unit_matrix([f.embedding], dim)
                if fhave.any():
                    sims = unit[kept_with_emb] @ fv[0]
                    top = np.argsort(-sims, kind="stable")[: t.support_k]
                    evidence = [_snippet(pool[kept_with_emb[j]].text, t.snippet_chars) for j in top]
            branch_entries.append({"subquery": f.subquery, "evidence": evidence})
        state = {"question": inp.root_query, "subquestions": subq_entries, "branches": branch_entries}
        questions: dict[str, Any] = {f"sub_{k}": subquestion_answered_question(k) for k in range(len(subq_entries))}
        questions.update({f"branch_{j}": branch_answered_question(j) for j in range(len(branch_entries))})
        return state, questions

    # ------------------------------------------------------------- decide

    async def decide(self, inp: OrchestrationInput) -> OrchestrationDecision:
        started = time.perf_counter()
        pool = inp.pool
        m = len(inp.subquestions)
        if self._n_subquestions is not None and self._n_subquestions != m:
            self._verdicts.clear()  # support vectors are tied to the question set
        self._n_subquestions = m
        usage_before = (self.usage.requests, self.usage.input_tokens, self.usage.output_tokens, self.usage.errors)

        dim = _dim_of(pool, [f.embedding for f in inp.frontier], [inp.query_embedding])
        unit, have = _unit_matrix([it.embedding for it in pool], dim)

        stop_probs: dict[str, float] = {}
        branch_probs: dict[str, float] = {}
        stop_error = False
        async with self._session() as client:
            to_judge = [it for it in pool if it.item_id not in self._verdicts]
            if to_judge:
                await self._judge(client, inp, to_judge)
            kept, reasons = self._select(inp, unit, have)

            if pool and (m or inp.frontier):
                state, questions = self._stop_state(inp, kept, unit, have, dim)
                resp = await self._ask(client, state, questions)
                if resp is None:
                    stop_error = True
                else:
                    try:
                        stop_probs = {
                            inp.subquestions[k]: _noul(resp.answers[f"sub_{k}"]) for k in range(m)
                        }
                        branch_probs = {
                            f.node_id: _noul(resp.answers[f"branch_{j}"]) for j, f in enumerate(inp.frontier)
                        }
                    except (KeyError, ValueError) as exc:
                        logger.warning("typesafe stop/allocation answers unusable: %s", exc)
                        stop_error = True
                        stop_probs, branch_probs = {}, {}

        terminate = bool(
            m
            and stop_probs
            and inp.round_id >= self.t.min_rounds
            and all(p >= self.t.stop_min for p in stop_probs.values())
        )
        if branch_probs and len(branch_probs) == len(inp.frontier):
            raw = {nid: (1.0 - p) + 0.05 for nid, p in branch_probs.items()}
            total = sum(raw.values())
            allocation = {nid: w / total for nid, w in raw.items()}
        else:
            allocation = uniform_allocation(inp.frontier)

        kept_ids = {pool[i].item_id for i in kept}
        reason_counts: dict[str, int] = {}
        for r in reasons.values():
            reason_counts[r] = reason_counts.get(r, 0) + 1
        req, in_tok, out_tok, errs = (
            self.usage.requests - usage_before[0],
            self.usage.input_tokens - usage_before[1],
            self.usage.output_tokens - usage_before[2],
            self.usage.errors - usage_before[3],
        )
        meta: dict[str, Any] = {
            "policy": self.name,
            "model": self.model,
            "budget": self.budget if self.budget is not None else inp.token_budget,
            "thresholds": {
                "keep_min": self.t.keep_min,
                "boilerplate_max": self.t.boilerplate_max,
                "stop_min": self.t.stop_min,
                "min_rounds": self.t.min_rounds,
                "support_k": self.t.support_k,
                "dedup_tau": self.t.dedup_tau,
            },
            "n_pool": len(pool),
            "n_new": len(inp.new_items),
            "n_judged": len([it for it in pool if it.item_id in self._verdicts]),
            "n_kept": len(kept),
            "tokens_kept": sum(pool[i].tokens for i in kept),
            "verdicts": {
                it.item_id: {
                    "useful": round(v.useful, 3),
                    "boilerplate": round(v.boilerplate, 3),
                    "support": [round(s, 3) for s in v.support],
                    "fallback": v.fallback,
                }
                for it in pool
                for v in [self._verdicts[it.item_id]]
            },
            "pruned_reasons": reasons,
            "pruned_reason_counts": reason_counts,
            "subquestion_answered": {q: round(p, 3) for q, p in stop_probs.items()},
            "branch_answered": {nid: round(p, 3) for nid, p in branch_probs.items()},
            "stop_error": stop_error,
            "terminate_reason": "all_subquestions_answered" if terminate else "",
            "round_usage": {"requests": req, "input_tokens": in_tok, "output_tokens": out_tok, "errors": errs},
            "usage": self.usage.as_meta(),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }
        return OrchestrationDecision(
            terminate=terminate,
            kept_ids=kept_ids,
            branch_allocation=allocation,
            meta=meta,
        )

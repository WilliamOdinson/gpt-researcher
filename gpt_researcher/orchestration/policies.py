"""Orchestration policies: the five Table-4 baselines behind one interface.

Every policy implements ``async decide(inp) -> OrchestrationDecision``. The
deep-research checkpoint builds an ``OrchestrationInput`` from this round's
new pages plus everything retained so far, calls the policy, and then *acts*
on the answer: rebuilds the forward context from ``kept_ids`` (and any
``rewritten`` texts), splits the next level's breadth by ``branch_allocation``,
and stops descending when ``terminate`` is set.

Policies are pure with respect to I/O except:
  * ``LLMLinguaPolicy`` runs a local compression model (CPU),
  * ``PromptedPolicy`` makes one LLM call through the callable it is given, so
    the call is metered by the fork's TokenTracker like any other.

Token counts are approximate (``len(text) // 4``) and used only for budgeting;
the trajectory's cost fields still come from API-returned usage.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)

POLICY_NAMES = ("legacy", "none", "topk", "extractive", "llmlingua", "prompted")
DEFAULT_POLICY = "legacy"

ENV_POLICY = "GR_ORCHESTRATOR"
ENV_BUDGET = "GR_CONTEXT_BUDGET_TOKENS"
ENV_TOPK_K = "GR_TOPK_K"
ENV_RATE = "GR_COMPRESSION_RATE"
ENV_LLMLINGUA_MODEL = "GR_LLMLINGUA_MODEL"

_DEFAULT_LLMLINGUA_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic, model-agnostic token estimate."""
    return max(1, len(text) // 4)


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


@dataclass
class PoolItem:
    """One evidence item (a page) as the orchestrator sees it."""

    item_id: str
    source_url: str
    text: str
    embedding: list[float] | None
    tree_depth: int
    retrieval_round: int
    is_new: bool
    source_subquery: str = ""
    # Chunk-level detail, present only for items first seen this round.
    chunks: list[str] = field(default_factory=list)
    chunk_embeddings: list[list[float]] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass
class FrontierInfo:
    node_id: str
    subquery: str


@dataclass
class OrchestrationInput:
    root_query: str
    query_embedding: list[float] | None
    subquestions: list[str]
    new_items: list[PoolItem]
    retained_prev: list[PoolItem]
    frontier: list[FrontierInfo]
    round_id: int
    tree_depth: int
    tokens_used: int
    token_budget: int | None

    @property
    def pool(self) -> list[PoolItem]:
        return [*self.retained_prev, *self.new_items]


@dataclass
class OrchestrationDecision:
    terminate: bool                                  # u
    kept_ids: set[str]                               # m  (over new ∪ prev)
    branch_allocation: dict[str, float]              # w  (node_id -> weight)
    rewritten: dict[str, str] = field(default_factory=dict)  # item_id -> new text
    meta: dict[str, Any] = field(default_factory=dict)

    def text_for(self, item: PoolItem) -> str:
        return self.rewritten.get(item.item_id, item.text)


# --------------------------------------------------------------------------
# Helpers shared by policies and the checkpoint
# --------------------------------------------------------------------------


def uniform_allocation(frontier: list[FrontierInfo]) -> dict[str, float]:
    if not frontier:
        return {}
    w = 1.0 / len(frontier)
    return {f.node_id: w for f in frontier}


def allocate_breadth(
    weights: dict[str, float], node_ids: list[str], total: int
) -> dict[str, int]:
    """Split ``total`` child searches across ``node_ids`` proportionally to
    ``weights`` with a floor of 1 per node (largest-remainder rounding).

    Uniform weights reproduce ``total // len(node_ids)`` per node exactly, i.e.
    the fork's historical ``max(2, breadth // 2)`` when ``total`` is that value
    times the number of nodes.
    """
    if not node_ids:
        return {}
    n = len(node_ids)
    raw = np.asarray([max(0.0, float(weights.get(nid, 0.0))) for nid in node_ids])
    if raw.sum() <= 0:
        raw = np.ones(n)
    raw = raw / raw.sum()

    total = max(total, n)  # floor of 1 each
    shares = raw * total
    floors = np.maximum(1, np.floor(shares).astype(int))
    remainder = total - int(floors.sum())
    if remainder > 0:
        order = np.argsort(-(shares - np.floor(shares)))
        for idx in order[:remainder]:
            floors[idx] += 1
    else:
        # Floors of 1 pushed us over; take repeatedly from the current largest.
        while remainder < 0:
            idx = int(np.argmax(floors))
            if floors[idx] <= 1:
                break
            floors[idx] -= 1
            remainder += 1
    return {nid: int(v) for nid, v in zip(node_ids, floors)}


def _snippet(text: str, n: int = 200) -> str:
    return re.sub(r"\s+", " ", text).strip()[:n]


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------


class OrchestrationPolicy:
    name: str = "base"
    needs_embeddings: bool = False

    async def decide(self, inp: OrchestrationInput) -> OrchestrationDecision:  # pragma: no cover
        raise NotImplementedError


class NoPruningPolicy(OrchestrationPolicy):
    """Row 1. Keep every item; uniform allocation; never stop early."""

    name = "none"

    async def decide(self, inp: OrchestrationInput) -> OrchestrationDecision:
        return OrchestrationDecision(
            terminate=False,
            kept_ids={it.item_id for it in inp.pool},
            branch_allocation=uniform_allocation(inp.frontier),
        )


class TopKPolicy(OrchestrationPolicy):
    """Row 2. Rank the whole pool by cosine to the root query; keep greedily
    while the running token total fits the budget (or the top ``k`` items when
    no budget is set). Uniform allocation; never stops early."""

    name = "topk"
    needs_embeddings = True

    def __init__(self, budget: int | None = None, k: int | None = None) -> None:
        self.budget = budget
        self.k = k

    async def decide(self, inp: OrchestrationInput) -> OrchestrationDecision:
        scored = [
            (cosine(inp.query_embedding, it.embedding), it.retrieval_round, it)
            for it in inp.pool
        ]
        # Highest similarity first; ties broken toward newer evidence.
        scored.sort(key=lambda s: (-s[0], -s[1]))

        kept: set[str] = set()
        used = 0
        budget = self.budget if self.budget is not None else inp.token_budget
        for rank, (score, _, it) in enumerate(scored):
            if budget is not None:
                if used + it.tokens > budget and kept:
                    continue
                if used + it.tokens > budget and not kept:
                    # Always keep at least the single best item.
                    kept.add(it.item_id)
                    used += it.tokens
                    continue
            elif self.k is not None and rank >= self.k:
                break
            kept.add(it.item_id)
            used += it.tokens

        return OrchestrationDecision(
            terminate=False,
            kept_ids=kept,
            branch_allocation=uniform_allocation(inp.frontier),
            meta={
                "scores": {it.item_id: round(s, 4) for s, _, it in scored},
                "budget": budget,
                "k": self.k,
                "tokens_kept": used,
            },
        )


class ExtractivePolicy(OrchestrationPolicy):
    """Row 3 (RECOMP-style, chunk granularity). Every item is kept, but each
    new item is rewritten to its most on-topic chunks so the retained total
    fits the budget. Items already retained in earlier rounds are left as-is
    (they were compressed when they arrived). Each item keeps at least
    ``min_chunks`` chunk(s), so nothing is dropped outright."""

    name = "extractive"
    needs_embeddings = True

    def __init__(self, budget: int | None = None, rate: float = 0.5, min_chunks: int = 1) -> None:
        self.budget = budget
        self.rate = rate
        self.min_chunks = max(1, min_chunks)

    async def decide(self, inp: OrchestrationInput) -> OrchestrationDecision:
        budget = self.budget if self.budget is not None else inp.token_budget
        total_now = sum(it.tokens for it in inp.pool)
        if budget is None:
            budget = int(total_now * self.rate)

        prev_tokens = sum(it.tokens for it in inp.retained_prev)
        compressible = [it for it in inp.new_items if it.chunks and it.chunk_embeddings]
        fixed_new = [it for it in inp.new_items if it not in compressible]
        fixed_tokens = prev_tokens + sum(it.tokens for it in fixed_new)

        # Score every chunk of every compressible item.
        per_item: dict[str, list[tuple[int, float, int]]] = {}
        for it in compressible:
            rows = []
            for idx, (chunk, emb) in enumerate(zip(it.chunks, it.chunk_embeddings)):
                rows.append((idx, cosine(inp.query_embedding, emb), estimate_tokens(chunk)))
            rows.sort(key=lambda r: -r[1])
            per_item[it.item_id] = rows

        # Mandatory: the top ``min_chunks`` of each item.
        chosen: dict[str, set[int]] = {}
        used = fixed_tokens
        for it in compressible:
            top = per_item[it.item_id][: self.min_chunks]
            chosen[it.item_id] = {idx for idx, _, _ in top}
            used += sum(t for _, _, t in top)

        # Then the best remaining chunks anywhere in the pool until budget.
        remaining = [
            (score, it.item_id, idx, toks)
            for it in compressible
            for idx, score, toks in per_item[it.item_id][self.min_chunks:]
        ]
        remaining.sort(key=lambda r: -r[0])
        for score, iid, idx, toks in remaining:
            if used + toks > budget:
                continue
            chosen[iid].add(idx)
            used += toks

        rewritten: dict[str, str] = {}
        compressed_chars: dict[str, tuple[int, int]] = {}
        for it in compressible:
            keep = sorted(chosen[it.item_id])
            new_text = "\n\n".join(it.chunks[i] for i in keep)
            if new_text != it.text:
                rewritten[it.item_id] = new_text
            compressed_chars[it.item_id] = (len(it.text), len(new_text))

        return OrchestrationDecision(
            terminate=False,
            kept_ids={it.item_id for it in inp.pool},
            branch_allocation=uniform_allocation(inp.frontier),
            rewritten=rewritten,
            meta={
                "budget": budget,
                "tokens_before": total_now,
                "tokens_after": used,
                "compressed_chars": compressed_chars,
            },
        )


class LLMLinguaPolicy(OrchestrationPolicy):
    """Row 4. Every item is kept; each new item's text is compressed with
    LLMLingua-2 at ``rate = budget / tokens_now`` (or a fixed rate when no
    budget is set). Requires ``pip install llmlingua``; fails loudly otherwise
    so a row can never silently degrade to no-pruning."""

    name = "llmlingua"

    def __init__(
        self,
        budget: int | None = None,
        rate: float = 0.5,
        model_name: str | None = None,
        compressor: Any | None = None,
    ) -> None:
        self.budget = budget
        self.rate = rate
        self.model_name = model_name or os.environ.get(ENV_LLMLINGUA_MODEL, _DEFAULT_LLMLINGUA_MODEL)
        self._compressor = compressor

    def _get_compressor(self) -> Any:
        if self._compressor is None:
            try:
                from llmlingua import PromptCompressor  # type: ignore
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "GR_ORCHESTRATOR=llmlingua needs the llmlingua package: "
                    "pip install llmlingua (downloads a ~500MB model on first use)"
                ) from exc
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._compressor = PromptCompressor(
                model_name=self.model_name,
                use_llmlingua2=True,
                device_map=device,
            )
        return self._compressor

    def _compress_one(self, text: str, rate: float) -> str:
        out = self._get_compressor().compress_prompt(
            text, rate=rate, force_tokens=["\n", "?", ".", "!", ","]
        )
        if isinstance(out, dict):
            return str(out.get("compressed_prompt", text))
        return str(out)

    async def decide(self, inp: OrchestrationInput) -> OrchestrationDecision:
        budget = self.budget if self.budget is not None else inp.token_budget
        total_now = sum(it.tokens for it in inp.pool)
        if budget is None:
            rate = self.rate
        else:
            prev_tokens = sum(it.tokens for it in inp.retained_prev)
            new_tokens = max(1, total_now - prev_tokens)
            rate = min(1.0, max(0.05, (budget - prev_tokens) / new_tokens))

        rewritten: dict[str, str] = {}
        compressed_chars: dict[str, tuple[int, int]] = {}
        if rate < 1.0:
            for it in inp.new_items:
                new_text = await asyncio.to_thread(self._compress_one, it.text, rate)
                if new_text and new_text != it.text:
                    rewritten[it.item_id] = new_text
                compressed_chars[it.item_id] = (len(it.text), len(new_text))

        after = sum(estimate_tokens(rewritten.get(it.item_id, it.text)) for it in inp.pool)
        return OrchestrationDecision(
            terminate=False,
            kept_ids={it.item_id for it in inp.pool},
            branch_allocation=uniform_allocation(inp.frontier),
            rewritten=rewritten,
            meta={
                "budget": budget,
                "rate": round(rate, 4),
                "tokens_before": total_now,
                "tokens_after": after,
                "compressed_chars": compressed_chars,
                "model": self.model_name,
            },
        )


LLMCall = Callable[[list[dict[str, str]]], Awaitable[str]]

_KEEP_RE = re.compile(r"^\s*KEEP\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_ALLOC_RE = re.compile(r"^\s*ALLOC\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_DECISION_RE = re.compile(r"^\s*DECISION\s*:\s*(CONTINUE|TERMINATE)", re.IGNORECASE | re.MULTILINE)
_PAIR_RE = re.compile(r"([A-Za-z0-9_\-]+)\s*=\s*([0-9]*\.?[0-9]+)")


class PromptedPolicy(OrchestrationPolicy):
    """Row 5. One LLM call per round sees the raw pool (ids, similarity,
    tokens, depth, age, a 200-char snippet), the open frontier, and the
    remaining budget, and answers KEEP / ALLOC / DECISION. Parse failures fall
    back to keep-all / uniform / continue and are flagged in ``meta``."""

    name = "prompted"
    needs_embeddings = True

    SYSTEM = (
        "You are the orchestrator of a deep research agent. After each research "
        "round you decide (1) which evidence items to keep in the working context, "
        "(2) how to split the next round's search effort across the open research "
        "branches, and (3) whether to stop researching and write the report. "
        "Answer ONLY in the exact three-line format requested."
    )

    def __init__(self, llm_call: LLMCall, budget: int | None = None) -> None:
        self.llm_call = llm_call
        self.budget = budget

    def build_prompt(self, inp: OrchestrationInput) -> str:
        budget = self.budget if self.budget is not None else inp.token_budget
        lines: list[str] = []
        lines.append(f"Root question: {inp.root_query.strip()}")
        if inp.subquestions:
            lines.append("Sub-questions the report must cover:")
            lines.extend(f"  - {s}" for s in inp.subquestions)
        lines.append(
            f"Round {inp.round_id}, tree depth {inp.tree_depth}. "
            f"Tokens spent so far: {inp.tokens_used}. "
            + (f"Context budget for retained evidence: {budget} tokens." if budget else "No hard context budget.")
        )
        lines.append("")
        lines.append("Evidence pool (id | sim-to-question | tokens | depth | round | status | snippet):")
        for it in inp.pool:
            sim = cosine(inp.query_embedding, it.embedding)
            status = "new" if it.is_new else "retained"
            lines.append(
                f"  {it.item_id} | {sim:.2f} | {it.tokens} | d{it.tree_depth} | r{it.retrieval_round} "
                f"| {status} | {_snippet(it.text)}"
            )
        lines.append("")
        if inp.frontier:
            lines.append("Open branches (node_id | sub-query):")
            lines.extend(f"  {f.node_id} | {f.subquery}" for f in inp.frontier)
        else:
            lines.append("Open branches: none")
        lines.append("")
        lines.append(
            "Respond with exactly three lines:\n"
            "KEEP: <space-separated evidence ids to keep, or ALL>\n"
            "ALLOC: <node_id>=<weight> <node_id>=<weight> ... (weights are relative)\n"
            "DECISION: CONTINUE or TERMINATE"
        )
        return "\n".join(lines)

    def parse(self, text: str, inp: OrchestrationInput) -> OrchestrationDecision:
        pool_ids = {it.item_id for it in inp.pool}
        meta: dict[str, Any] = {"raw": text[:4000], "parse_failure": False}

        keep_m = _KEEP_RE.search(text or "")
        if not keep_m:
            meta["parse_failure"] = True
            kept = set(pool_ids)
        else:
            body = keep_m.group(1).strip()
            if body.upper() == "ALL":
                kept = set(pool_ids)
            else:
                tokens = re.findall(r"[A-Za-z0-9_\-]+", body)
                kept = {t for t in tokens if t in pool_ids}
                if not kept and pool_ids:
                    # The model answered but named nothing we know; keep all
                    # rather than wipe the context.
                    meta["parse_failure"] = True
                    kept = set(pool_ids)

        alloc = uniform_allocation(inp.frontier)
        alloc_m = _ALLOC_RE.search(text or "")
        if alloc_m:
            node_ids = {f.node_id for f in inp.frontier}
            parsed = {k: float(v) for k, v in _PAIR_RE.findall(alloc_m.group(1)) if k in node_ids}
            if parsed and sum(parsed.values()) > 0:
                s = sum(parsed.values())
                alloc = {k: v / s for k, v in parsed.items()}
                for nid in node_ids - parsed.keys():
                    alloc[nid] = 0.0

        dec_m = _DECISION_RE.search(text or "")
        terminate = bool(dec_m and dec_m.group(1).upper() == "TERMINATE")

        return OrchestrationDecision(
            terminate=terminate, kept_ids=kept, branch_allocation=alloc, meta=meta
        )

    async def decide(self, inp: OrchestrationInput) -> OrchestrationDecision:
        prompt = self.build_prompt(inp)
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": prompt},
        ]
        try:
            text = await self.llm_call(messages)
        except Exception as exc:  # the row must not crash the run
            logger.warning("Prompted orchestrator LLM call failed: %s", exc)
            dec = OrchestrationDecision(
                terminate=False,
                kept_ids={it.item_id for it in inp.pool},
                branch_allocation=uniform_allocation(inp.frontier),
                meta={"parse_failure": True, "error": str(exc)},
            )
            dec.meta["prompt_chars"] = len(prompt)
            return dec
        dec = self.parse(text, inp)
        dec.meta["prompt_chars"] = len(prompt)
        return dec


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def policy_name_from_env() -> str:
    name = os.environ.get(ENV_POLICY, DEFAULT_POLICY).strip().lower() or DEFAULT_POLICY
    if name not in POLICY_NAMES:
        raise ValueError(f"{ENV_POLICY}={name!r} is not one of {POLICY_NAMES}")
    return name


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def build_policy(
    name: str | None = None,
    *,
    llm_call: LLMCall | None = None,
    budget: int | None = None,
) -> OrchestrationPolicy | None:
    """Return the policy for ``name`` (default: from ``GR_ORCHESTRATOR``), or
    ``None`` for ``legacy`` so the caller keeps the pre-existing code path."""
    name = (name or policy_name_from_env()).lower()
    if budget is None:
        budget = _env_int(ENV_BUDGET)
    rate = _env_float(ENV_RATE, 0.5)

    if name == "legacy":
        return None
    if name == "none":
        return NoPruningPolicy()
    if name == "topk":
        return TopKPolicy(budget=budget, k=_env_int(ENV_TOPK_K))
    if name == "extractive":
        return ExtractivePolicy(budget=budget, rate=rate)
    if name == "llmlingua":
        return LLMLinguaPolicy(budget=budget, rate=rate)
    if name == "prompted":
        if llm_call is None:
            raise ValueError("prompted policy needs an llm_call")
        return PromptedPolicy(llm_call=llm_call, budget=budget)
    raise ValueError(f"Unknown orchestration policy {name!r}")

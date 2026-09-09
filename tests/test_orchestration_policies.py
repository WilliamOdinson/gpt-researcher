"""Orchestration policies on a fixed synthetic pool.

Embeddings are 3-d unit vectors so cosine to the query is known exactly:
  q = (1, 0, 0)
  P1 rho=1.00  P2 rho=0.80  P3 rho=0.60  P4 rho=0.00 (new)
  R1 rho=0.90                                       (retained from round 1)
Token estimate is len(text) // 4, so texts are sized to make budgets exact.
"""

from __future__ import annotations

import math

import pytest

from gpt_researcher.orchestration import (
    ExtractivePolicy,
    FrontierInfo,
    LLMLinguaPolicy,
    NoPruningPolicy,
    OrchestrationInput,
    PoolItem,
    PromptedPolicy,
    TopKPolicy,
    allocate_breadth,
    build_policy,
    estimate_tokens,
    policy_name_from_env,
)


def _vec(rho: float) -> list[float]:
    return [rho, math.sqrt(max(0.0, 1 - rho * rho)), 0.0]


def _text(tokens: int) -> str:
    return "x" * (tokens * 4)


def _item(iid: str, rho: float, tokens: int, *, new: bool = True, chunks: list[str] | None = None,
          chunk_rhos: list[float] | None = None, rnd: int = 2) -> PoolItem:
    text = "\n\n".join(chunks) if chunks else _text(tokens)
    return PoolItem(
        item_id=iid,
        source_url=f"https://example.org/{iid}",
        text=text,
        embedding=_vec(rho),
        tree_depth=1,
        retrieval_round=rnd,
        is_new=new,
        chunks=chunks or [],
        chunk_embeddings=[_vec(r) for r in (chunk_rhos or [])],
    )


FRONTIER = [FrontierInfo("nA", "branch A"), FrontierInfo("nB", "branch B"), FrontierInfo("nC", "branch C")]


def _inp(new, prev=(), budget=None):
    return OrchestrationInput(
        root_query="what is x",
        query_embedding=_vec(1.0),
        subquestions=["sub 1", "sub 2"],
        new_items=list(new),
        retained_prev=list(prev),
        frontier=FRONTIER,
        round_id=2,
        tree_depth=1,
        tokens_used=1234,
        token_budget=budget,
    )


POOL_NEW = [
    _item("P1", 1.0, 100),
    _item("P2", 0.8, 100),
    _item("P3", 0.6, 100),
    _item("P4", 0.0, 100),
]
POOL_PREV = [_item("R1", 0.9, 100, new=False, rnd=1)]


# ---------------------------------------------------------------- none


async def test_none_keeps_everything_uniform_continue():
    dec = await NoPruningPolicy().decide(_inp(POOL_NEW, POOL_PREV, budget=50))
    assert dec.kept_ids == {"P1", "P2", "P3", "P4", "R1"}
    assert dec.terminate is False
    assert dec.rewritten == {}
    assert dec.branch_allocation == pytest.approx({"nA": 1 / 3, "nB": 1 / 3, "nC": 1 / 3})


# ---------------------------------------------------------------- topk


async def test_topk_ranks_whole_pool_by_cosine_and_respects_budget():
    # Budget for exactly three 100-token items: P1 (1.0), R1 (0.9), P2 (0.8).
    dec = await TopKPolicy().decide(_inp(POOL_NEW, POOL_PREV, budget=300))
    assert dec.kept_ids == {"P1", "R1", "P2"}
    assert dec.meta["tokens_kept"] == 300
    assert dec.terminate is False


async def test_topk_supersedes_previously_retained_items():
    # R1 has lower similarity than every new item -> it is pruned this round.
    prev = [_item("R1", 0.1, 100, new=False, rnd=1)]
    dec = await TopKPolicy().decide(_inp(POOL_NEW, prev, budget=300))
    assert dec.kept_ids == {"P1", "P2", "P3"}
    assert "R1" not in dec.kept_ids


async def test_topk_k_without_budget():
    dec = await TopKPolicy(k=2).decide(_inp(POOL_NEW, POOL_PREV))
    assert dec.kept_ids == {"P1", "R1"}


async def test_topk_always_keeps_best_item_even_if_over_budget():
    dec = await TopKPolicy().decide(_inp(POOL_NEW, [], budget=10))
    assert dec.kept_ids == {"P1"}


async def test_topk_without_query_embedding_keeps_by_recency():
    inp = _inp(POOL_NEW, POOL_PREV, budget=200)
    inp.query_embedding = None
    dec = await TopKPolicy().decide(inp)
    # All scores tie at 0.0; newer rounds win the tie-break.
    assert dec.kept_ids <= {"P1", "P2", "P3", "P4"}
    assert len(dec.kept_ids) == 2


# ---------------------------------------------------------------- extractive


async def test_extractive_keeps_all_items_but_trims_chunks_to_budget():
    # Two items, each three 50-token chunks (150 tokens per item, 300 total).
    a = _item("A", 0.9, 0, chunks=["x" * 200, "y" * 200, "z" * 200], chunk_rhos=[1.0, 0.2, 0.5])
    b = _item("B", 0.7, 0, chunks=["p" * 200, "q" * 200, "r" * 200], chunk_rhos=[0.9, 0.1, 0.3])
    dec = await ExtractivePolicy().decide(_inp([a, b], [], budget=200))
    assert dec.kept_ids == {"A", "B"}  # nothing dropped
    # Mandatory top chunk of each (2 x 50 = 100), then best remaining chunks
    # until 200: A's 0.5 chunk (150) and B's 0.3 chunk (200). Stops there.
    assert estimate_tokens(dec.rewritten["A"]) == 100
    assert estimate_tokens(dec.rewritten["B"]) == 100
    assert dec.meta["tokens_after"] == 200
    # Kept chunks preserve original order within the item.
    assert dec.rewritten["A"].startswith("x") and dec.rewritten["A"].endswith("z")
    assert "y" not in dec.rewritten["A"]


async def test_extractive_uses_rate_when_no_budget():
    a = _item("A", 0.9, 0, chunks=["x" * 200, "y" * 200], chunk_rhos=[1.0, 0.2])
    dec = await ExtractivePolicy(rate=0.5).decide(_inp([a], []))
    assert dec.meta["budget"] == 50
    assert estimate_tokens(dec.rewritten["A"]) == 50


async def test_extractive_leaves_previously_retained_items_untouched():
    a = _item("A", 0.9, 0, chunks=["x" * 200, "y" * 200], chunk_rhos=[1.0, 0.2])
    dec = await ExtractivePolicy().decide(_inp([a], POOL_PREV, budget=150))
    assert "R1" in dec.kept_ids and "R1" not in dec.rewritten


# ---------------------------------------------------------------- llmlingua


class _FakeCompressor:
    def __init__(self):
        self.calls: list[float] = []

    def compress_prompt(self, text, rate, force_tokens=None):
        self.calls.append(rate)
        n = max(1, int(len(text) * rate))
        return {"compressed_prompt": text[:n]}


async def test_llmlingua_compresses_new_items_at_budget_rate():
    fake = _FakeCompressor()
    pol = LLMLinguaPolicy(compressor=fake)
    # prev 100 tokens, new 400 tokens, budget 300 -> rate for new = 0.5
    dec = await pol.decide(_inp(POOL_NEW, POOL_PREV, budget=300))
    assert dec.kept_ids == {"P1", "P2", "P3", "P4", "R1"}
    assert fake.calls and all(abs(r - 0.5) < 1e-9 for r in fake.calls)
    assert set(dec.rewritten) == {"P1", "P2", "P3", "P4"}
    assert dec.meta["tokens_after"] == 300


async def test_llmlingua_missing_package_fails_loudly(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "llmlingua":
            raise ImportError("nope")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pol = LLMLinguaPolicy()
    with pytest.raises(RuntimeError, match="pip install llmlingua"):
        await pol.decide(_inp(POOL_NEW, [], budget=100))


# ---------------------------------------------------------------- prompted


async def test_prompted_parses_keep_alloc_decision():
    seen: list[list[dict]] = []

    async def llm(messages):
        seen.append(messages)
        return "KEEP: P1 P2 R1\nALLOC: nA=2 nB=1 nC=1\nDECISION: TERMINATE\n"

    dec = await PromptedPolicy(llm).decide(_inp(POOL_NEW, POOL_PREV, budget=300))
    assert dec.kept_ids == {"P1", "P2", "R1"}
    assert dec.branch_allocation == pytest.approx({"nA": 0.5, "nB": 0.25, "nC": 0.25})
    assert dec.terminate is True
    assert dec.meta["parse_failure"] is False
    prompt = seen[0][1]["content"]
    # The raw pool, with snippets and similarity, is in the prompt.
    for iid in ("P1", "P2", "P3", "P4", "R1"):
        assert iid in prompt
    assert "Context budget for retained evidence: 300" in prompt
    assert "nA | branch A" in prompt


async def test_prompted_keep_all_and_unknown_ids():
    async def llm(messages):
        return "KEEP: ALL\nALLOC: zz=1\nDECISION: CONTINUE"

    dec = await PromptedPolicy(llm).decide(_inp(POOL_NEW, POOL_PREV))
    assert dec.kept_ids == {"P1", "P2", "P3", "P4", "R1"}
    # Unknown node ids are ignored -> uniform allocation.
    assert dec.branch_allocation == pytest.approx({"nA": 1 / 3, "nB": 1 / 3, "nC": 1 / 3})
    assert dec.terminate is False


async def test_prompted_parse_failure_falls_back_to_keep_all():
    async def llm(messages):
        return "I think we should keep the good ones."

    dec = await PromptedPolicy(llm).decide(_inp(POOL_NEW, POOL_PREV))
    assert dec.kept_ids == {"P1", "P2", "P3", "P4", "R1"}
    assert dec.meta["parse_failure"] is True
    assert dec.terminate is False


async def test_prompted_llm_error_does_not_crash():
    async def llm(messages):
        raise RuntimeError("api down")

    dec = await PromptedPolicy(llm).decide(_inp(POOL_NEW, POOL_PREV))
    assert dec.kept_ids == {"P1", "P2", "P3", "P4", "R1"}
    assert dec.meta["parse_failure"] is True
    assert "api down" in dec.meta["error"]


# ---------------------------------------------------------------- allocation


def test_allocate_breadth_uniform_reproduces_legacy():
    nodes = ["a", "b", "c"]
    out = allocate_breadth({n: 1 / 3 for n in nodes}, nodes, total=6)
    assert out == {"a": 2, "b": 2, "c": 2}


def test_allocate_breadth_skews_but_keeps_total_and_floor():
    nodes = ["a", "b", "c"]
    out = allocate_breadth({"a": 0.5, "b": 0.4, "c": 0.1}, nodes, total=6)
    assert sum(out.values()) == 6
    assert out["a"] >= out["b"] >= out["c"] >= 1
    # Zero weight still gets the floor.
    out2 = allocate_breadth({"a": 1.0}, nodes, total=6)
    assert out2["b"] == 1 and out2["c"] == 1 and out2["a"] == 4


def test_allocate_breadth_empty_weights_is_uniform():
    nodes = ["a", "b"]
    assert allocate_breadth({}, nodes, total=4) == {"a": 2, "b": 2}


# ---------------------------------------------------------------- selection


def test_policy_from_env(monkeypatch):
    monkeypatch.delenv("GR_ORCHESTRATOR", raising=False)
    assert policy_name_from_env() == "legacy"
    assert build_policy("legacy") is None
    monkeypatch.setenv("GR_ORCHESTRATOR", "topk")
    monkeypatch.setenv("GR_CONTEXT_BUDGET_TOKENS", "4200")
    pol = build_policy()
    assert isinstance(pol, TopKPolicy) and pol.budget == 4200
    monkeypatch.setenv("GR_ORCHESTRATOR", "bogus")
    with pytest.raises(ValueError):
        policy_name_from_env()


def test_prompted_requires_llm_call():
    with pytest.raises(ValueError):
        build_policy("prompted")

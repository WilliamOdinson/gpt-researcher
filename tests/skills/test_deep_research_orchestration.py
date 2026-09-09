"""The deep-research checkpoint acts on the orchestration policy's (u, m, w).

Sub-researchers are mocked; the embedding cache is fed directly with the
records the real ContextCompressor would produce. Query embedding is set by
hand because these tests call deep_research() rather than run().
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpt_researcher.context import _global_embedding_cache
from gpt_researcher.skills.deep_research import DeepResearchSkill
from gpt_researcher.utils.trajectory_logger import make_item_id


def _researcher(depth: int, breadth: int):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            deep_research_breadth=breadth,
            deep_research_depth=depth,
            deep_research_concurrency=2,
            strategic_llm_provider="openai",
            strategic_llm_model="gpt",
            reasoning_effort=None,
            llm_kwargs={},
            config_path=None,
        ),
        query="what is x",
        websocket=None,
        tone=None,
        headers={},
        visited_urls=set(),
        mcp_configs=None,
        mcp_strategy=None,
    )


def _chunk(url: str, emb: list[float], kept: bool, fill: str) -> dict:
    return {
        "chunk_id": f"{url}-{fill}",
        "content": fill * 200,  # 50 estimated tokens
        "source_url": url,
        "embedding": emb,
        "similarity": 0.0,
        "kept": kept,
    }


# Page A: on-topic, filter kept.  Page B: off-topic but filter kept.
# Page C: on-topic but the filter pruned it.
PAGE_A = ("https://a", [1.0, 0.0, 0.0], True, "a")
PAGE_B = ("https://b", [0.0, 1.0, 0.0], True, "b")
PAGE_C = ("https://c", [0.9, 0.4359, 0.0], False, "c")


def _feed_cache_for(query: str):
    """Sub-researcher side effect: pretend q1 saw pages A and B, q2 saw page C."""
    if query.endswith("1"):
        chunks = [_chunk(*PAGE_A), _chunk(*PAGE_B)]
    else:
        chunks = [_chunk(*PAGE_C)]
    _global_embedding_cache.records.append({"query": query, "chunks": chunks})


async def _run(skill: DeepResearchSkill, depth: int, breadth: int):
    async def fake_generate(query, num_queries=3):
        return [
            {"query": "q1", "researchGoal": "goal 1"},
            {"query": "q2", "researchGoal": "goal 2"},
        ]

    async def fake_process(query, context, num_learnings=3):
        return {"learnings": [f"learned {query}"], "followUpQuestions": ["why?"], "citations": {}}

    skill.generate_search_queries = fake_generate  # type: ignore
    skill.process_research_results = fake_process  # type: ignore

    with patch("gpt_researcher.GPTResearcher") as MockR:
        def make(query, **kwargs):
            inst = SimpleNamespace(visited_urls=set(), research_sources=[])

            async def conduct():
                _feed_cache_for(query)
                return f"ctx-{query}"

            inst.conduct_research = conduct
            return inst

        MockR.side_effect = make
        out = await skill.deep_research(query="what is x", breadth=breadth, depth=depth)
        return out, MockR


@pytest.fixture(autouse=True)
def _clean_cache():
    _global_embedding_cache.records = []
    yield
    _global_embedding_cache.records = []


# ------------------------------------------------------------------ legacy


async def test_legacy_records_filter_verdict_and_passes_context_through(monkeypatch):
    monkeypatch.delenv("GR_ORCHESTRATOR", raising=False)
    skill = DeepResearchSkill(_researcher(depth=1, breadth=2))
    assert skill.orchestrator is None

    out, _ = await _run(skill, depth=1, breadth=2)
    snap = skill.trajectory_logger.trajectory.rounds[0]
    assert snap.decision.policy == "legacy"
    assert snap.decision.type == "continue"
    assert snap.decision.branch_allocation == {}
    ids = {e.source_url: iid for iid, e in skill.trajectory_logger.get_all_evidence().items()}
    assert set(snap.decision.kept_item_ids) == {ids["https://a"], ids["https://b"]}
    assert set(snap.decision.pruned_item_ids) == {ids["https://c"]}
    # Context is whatever the sub-researchers returned.
    assert sorted(out["context"]) == ["ctx-q1", "ctx-q2"]


# ------------------------------------------------------------------ topk


async def test_topk_rebuilds_context_from_its_own_decision(monkeypatch):
    monkeypatch.setenv("GR_ORCHESTRATOR", "topk")
    monkeypatch.setenv("GR_CONTEXT_BUDGET_TOKENS", "100")  # two 50-token pages
    skill = DeepResearchSkill(_researcher(depth=1, breadth=2))
    skill.query_embedding = [1.0, 0.0, 0.0]

    out, _ = await _run(skill, depth=1, breadth=2)
    snap = skill.trajectory_logger.trajectory.rounds[0]
    ids = {e.source_url: iid for iid, e in skill.trajectory_logger.get_all_evidence().items()}

    assert snap.decision.policy == "topk"
    # A (1.0) and C (0.9) fit the budget; B (0.0) is pruned even though the
    # filter had kept it.
    assert set(snap.decision.kept_item_ids) == {ids["https://a"], ids["https://c"]}
    assert set(snap.decision.pruned_item_ids) == {ids["https://b"]}
    assert snap.decision.branch_allocation == pytest.approx(
        {make_item_id("goal 1", ""): 0.5, make_item_id("goal 2", ""): 0.5}
    )
    assert snap.decision.meta["context_tokens_before"] == 150
    assert snap.decision.meta["context_tokens_after"] == 100

    # The forward context is the kept pages, with their URLs, and not the
    # sub-researchers' strings.
    joined = "\n".join(out["context"])
    assert "Source: https://a" in joined and "Source: https://c" in joined
    assert "https://b" not in joined and "ctx-q1" not in joined

    # Retained set on the logger agrees.
    retained = {e.source_url for e in skill.trajectory_logger.get_retained_evidence().values()}
    assert retained == {"https://a", "https://c"}


# ------------------------------------------------------------------ none


async def test_none_keeps_filter_pruned_pages_too(monkeypatch):
    monkeypatch.setenv("GR_ORCHESTRATOR", "none")
    skill = DeepResearchSkill(_researcher(depth=1, breadth=2))
    out, _ = await _run(skill, depth=1, breadth=2)
    snap = skill.trajectory_logger.trajectory.rounds[0]
    assert snap.decision.policy == "none"
    assert snap.decision.pruned_item_ids == []
    assert len(snap.decision.kept_item_ids) == 3
    assert len(out["context"]) == 3


# ------------------------------------------------------------------ u and w


async def test_prompted_terminate_stops_recursion_and_alloc_sets_child_breadth(monkeypatch):
    monkeypatch.setenv("GR_ORCHESTRATOR", "prompted")
    skill = DeepResearchSkill(_researcher(depth=2, breadth=4))
    skill.query_embedding = [1.0, 0.0, 0.0]

    n1, n2 = make_item_id("goal 1", ""), make_item_id("goal 2", "")
    reply = f"KEEP: ALL\nALLOC: {n1}=3 {n2}=1\nDECISION: TERMINATE\n"
    with patch(
        "gpt_researcher.skills.deep_research.create_chat_completion",
        new=AsyncMock(return_value=reply),
    ):
        out, MockR = await _run(skill, depth=2, breadth=4)

    snap = skill.trajectory_logger.trajectory.rounds[0]
    assert snap.decision.policy == "prompted"
    assert snap.decision.type == "terminate"
    assert skill.stop_requested is True
    # Only the two top-level sub-researchers ran; no descent.
    assert MockR.call_count == 2
    assert len(skill.trajectory_logger.trajectory.rounds) == 1
    # w: total child breadth = 2 nodes x max(2, 4 // 2) = 4, split 3:1.
    assert snap.decision.meta["child_breadth"] == {n1: 3, n2: 1}


async def test_legacy_recursion_unchanged(monkeypatch):
    monkeypatch.delenv("GR_ORCHESTRATOR", raising=False)
    skill = DeepResearchSkill(_researcher(depth=2, breadth=4))
    out, MockR = await _run(skill, depth=2, breadth=4)
    # 2 top-level + 2 children each with 2 queries = 6 sub-researchers.
    assert MockR.call_count == 6
    assert len(skill.trajectory_logger.trajectory.rounds) == 3


async def test_none_recursion_matches_legacy_effort(monkeypatch):
    monkeypatch.setenv("GR_ORCHESTRATOR", "none")
    skill = DeepResearchSkill(_researcher(depth=2, breadth=4))
    out, MockR = await _run(skill, depth=2, breadth=4)
    assert MockR.call_count == 6
    for snap in skill.trajectory_logger.trajectory.rounds:
        assert set(snap.decision.meta["child_breadth"].values()) == {2}

"""Token stopping rule (GR_STOP_TOTAL_TOKENS) at its two check points.

Sub-researchers and the LLM steps are mocked; each mock charges a known number
of tokens to the TokenTracker so the rule's arithmetic is exact. Concurrency is
1, as in the pilot configs, so sub-queries run one after another.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpt_researcher.context import _global_embedding_cache
from gpt_researcher.skills import deep_research as dr
from gpt_researcher.skills.deep_research import DeepResearchSkill
from gpt_researcher.utils.token_tracker import TokenTracker

GEN_TOKENS = 100  # generate_search_queries
CONDUCT_TOKENS = 4000  # one sub-researcher's conduct_research
PROCESS_TOKENS = 1000  # process_research_results
SUBQUERY_TOKENS = CONDUCT_TOKENS + PROCESS_TOKENS


def _researcher(depth: int, breadth: int):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            deep_research_breadth=breadth,
            deep_research_depth=depth,
            deep_research_concurrency=1,
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


def _charge(n: int) -> None:
    TokenTracker.track_tokens("gpt", n, 0)


async def _run(skill: DeepResearchSkill, depth: int, breadth: int):
    async def fake_generate(query, num_queries=3):
        _charge(GEN_TOKENS)
        return [{"query": f"{query[:8]}-q{i}", "researchGoal": f"goal {i}"} for i in range(num_queries)]

    async def fake_process(query, context, num_learnings=3):
        _charge(PROCESS_TOKENS)
        return {"learnings": [f"learned {query}"], "followUpQuestions": ["why?"], "citations": {}}

    skill.generate_search_queries = fake_generate  # type: ignore
    skill.process_research_results = fake_process  # type: ignore

    with patch("gpt_researcher.GPTResearcher") as MockR:
        def make(query, **kwargs):
            inst = SimpleNamespace(visited_urls=set(), research_sources=[])

            async def conduct():
                _charge(CONDUCT_TOKENS)
                _global_embedding_cache.records.append({
                    "query": query,
                    "chunks": [{
                        "chunk_id": f"{query}-c",
                        "content": "word " * 50,
                        "source_url": f"https://{query}",
                        "embedding": [1.0, 0.0],
                        "similarity": 0.0,
                        "kept": True,
                    }],
                })
                return f"ctx-{query}"

            inst.conduct_research = conduct
            return inst

        MockR.side_effect = make
        out = await skill.deep_research(query="what is x", breadth=breadth, depth=depth)
        return out, MockR


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    TokenTracker.reset()
    _global_embedding_cache.records = []
    for name in ("GR_ORCHESTRATOR", dr.ENV_STOP_TOTAL_TOKENS, dr.ENV_REPORT_RESERVE_TOKENS):
        monkeypatch.delenv(name, raising=False)
    yield
    TokenTracker.reset()
    _global_embedding_cache.records = []


def _rounds(skill: DeepResearchSkill) -> int:
    return len(skill.trajectory_logger.trajectory.rounds)


# ------------------------------------------------------------------ no rule


async def test_no_stopping_rule_by_default_runs_the_full_tree():
    skill = DeepResearchSkill(_researcher(depth=2, breadth=2))
    assert skill.stop_total_tokens is None
    assert skill.token_stop_info["report_reserve"] is None

    _, MockR = await _run(skill, depth=2, breadth=2)

    # depth 2, breadth 2 (legacy keeps the child breadth): one root round + one
    # child round per result = 3 rounds of 2 sub-queries, 6 sub-researchers.
    assert _rounds(skill) == 3
    assert MockR.call_count == 6
    assert skill.token_stop_info["stopped"] is False
    assert skill.stop_requested is False


async def test_zero_target_means_no_stopping_rule(monkeypatch):
    monkeypatch.setenv(dr.ENV_STOP_TOTAL_TOKENS, "0")
    skill = DeepResearchSkill(_researcher(depth=2, breadth=2))
    assert skill.stop_total_tokens is None
    _, MockR = await _run(skill, depth=2, breadth=2)
    assert MockR.call_count == 6


# ------------------------------------------------------------------ check points


async def test_stops_before_next_round(monkeypatch):
    # Root round: 100 + 2 x 5000 = 10100 used. Child round check:
    # 10100 + (5000 + 100) + 500 = 15700 > 15000 -> stop before the round.
    monkeypatch.setenv(dr.ENV_STOP_TOTAL_TOKENS, "15000")
    monkeypatch.setenv(dr.ENV_REPORT_RESERVE_TOKENS, "500")
    skill = DeepResearchSkill(_researcher(depth=2, breadth=2))
    # First-round checks use the defaults (20000 + 4000), which would block at
    # once; start from measured costs so only the round check binds.
    skill._subquery_costs = [SUBQUERY_TOKENS]
    skill._round_overheads = [GEN_TOKENS]

    _, MockR = await _run(skill, depth=2, breadth=2)

    info = skill.token_stop_info
    assert info["stopped"] is True
    assert info["stopped_at"] == "round"
    assert info["tokens_at_stop"] == GEN_TOKENS + 2 * SUBQUERY_TOKENS
    assert info["report_reserve_at_stop"] == 500
    assert info["skipped_rounds"] == 1  # the second child returns on stop_requested
    assert _rounds(skill) == 1
    assert MockR.call_count == 2
    assert skill.stop_requested is True


async def test_stops_before_next_subquery(monkeypatch):
    # Round check: 0 + (5000 + 100) + 0 <= 9000. Sub-query 1: 100 + 5100 <= 9000.
    # Sub-query 2: 5100 + 5100 = 10200 > 9000 -> stop; the round still
    # checkpoints the one finished sub-query.
    monkeypatch.setenv(dr.ENV_STOP_TOTAL_TOKENS, "9000")
    monkeypatch.setenv(dr.ENV_REPORT_RESERVE_TOKENS, "0")
    skill = DeepResearchSkill(_researcher(depth=2, breadth=2))
    skill._subquery_costs = [SUBQUERY_TOKENS]
    skill._round_overheads = [GEN_TOKENS]

    _, MockR = await _run(skill, depth=2, breadth=2)

    info = skill.token_stop_info
    assert info["stopped"] is True
    assert info["stopped_at"] == "subquery"
    assert info["tokens_at_stop"] == GEN_TOKENS + SUBQUERY_TOKENS
    assert info["skipped_subqueries"] == 1
    assert MockR.call_count == 1
    assert _rounds(skill) == 1  # checkpointed, no descent after the stop


async def test_all_subqueries_blocked_returns_without_error(monkeypatch):
    # The round check passes on the defaults but the first sub-query does not.
    monkeypatch.setenv(dr.ENV_STOP_TOTAL_TOKENS, "24050")
    monkeypatch.setenv(dr.ENV_REPORT_RESERVE_TOKENS, "0")
    skill = DeepResearchSkill(_researcher(depth=1, breadth=2))

    out, MockR = await _run(skill, depth=1, breadth=2)

    assert skill.token_stop_info["stopped_at"] == "subquery"
    assert MockR.call_count == 0
    assert out["learnings"] == [] and out["context"] == []


async def test_measured_costs_replace_the_defaults(monkeypatch):
    monkeypatch.setenv(dr.ENV_STOP_TOTAL_TOKENS, "1000000")
    skill = DeepResearchSkill(_researcher(depth=1, breadth=2))
    await _run(skill, depth=1, breadth=2)
    assert skill._subquery_costs == [SUBQUERY_TOKENS, SUBQUERY_TOKENS]
    assert skill._round_overheads == [GEN_TOKENS]
    assert skill.token_stop_info["stopped"] is False


# ------------------------------------------------------------------ report reserve


def test_report_cost_model_matches_the_full_context_reserve():
    # A full 25k-word writer context costs about what the old fixed 45k held.
    full = dr.report_tokens_for_words(dr.MAX_CONTEXT_WORDS)
    assert 43000 <= full <= 46000
    assert dr.report_tokens_for_words(0) == dr.REPORT_PROMPT_OVERHEAD_TOKENS + dr.REPORT_OUTPUT_TOKENS


def test_auto_reserve_without_a_policy_assumes_the_full_context(monkeypatch):
    monkeypatch.setenv(dr.ENV_STOP_TOTAL_TOKENS, "150000")
    skill = DeepResearchSkill(_researcher(depth=1, breadth=1))
    assert skill.orchestrator is None
    reserve, words = skill._report_reserve()
    assert words == dr.MAX_CONTEXT_WORDS
    assert reserve == dr.report_tokens_for_words(dr.MAX_CONTEXT_WORDS)


def test_auto_reserve_tracks_retained_evidence_under_a_policy(monkeypatch):
    monkeypatch.setenv(dr.ENV_STOP_TOTAL_TOKENS, "150000")
    monkeypatch.setenv("GR_ORCHESTRATOR", "none")
    skill = DeepResearchSkill(_researcher(depth=1, breadth=1))
    assert skill.token_stop_info["report_reserve"] == "auto"

    # Nothing retained yet: one sub-query's default growth.
    reserve, words = skill._report_reserve()
    assert words == dr.DEFAULT_SUBQUERY_CONTEXT_WORDS
    assert reserve == dr.report_tokens_for_words(words)

    # A small measured growth (a pruning policy) gives a much smaller reserve.
    skill._subquery_context_words = [200.0]
    small, words = skill._report_reserve()
    assert words == 200
    assert small < dr.report_tokens_for_words(dr.MAX_CONTEXT_WORDS) // 3

    # Pending (not yet checkpointed) sub-queries count toward the projection.
    skill._pending_subqueries = 2
    _, words = skill._report_reserve()
    assert words == 600


def test_fixed_reserve_ignores_the_evidence(monkeypatch):
    monkeypatch.setenv(dr.ENV_STOP_TOTAL_TOKENS, "150000")
    monkeypatch.setenv(dr.ENV_REPORT_RESERVE_TOKENS, "45000")
    monkeypatch.setenv("GR_ORCHESTRATOR", "none")
    skill = DeepResearchSkill(_researcher(depth=1, breadth=1))
    assert skill._report_reserve() == (45000, None)
    assert skill.token_stop_info["report_reserve"] == 45000

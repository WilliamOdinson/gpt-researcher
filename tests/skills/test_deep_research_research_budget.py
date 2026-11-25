"""Research-only budget (GR_RESEARCH_BUDGET), writer context cap
(GR_MAX_CONTEXT_WORDS), fixed report length (GR_REPORT_WORDS) and the
score-based prompted orchestrator."""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

from gpt_researcher.orchestration import (
    FrontierInfo,
    OrchestrationInput,
    PoolItem,
    PromptedPolicy,
    PromptedScoresPolicy,
    build_policy,
)
from gpt_researcher.prompts import PromptFamily, report_body_words, report_word_target
from gpt_researcher.skills import deep_research as dr
from gpt_researcher.skills.deep_research import DeepResearchSkill
from gpt_researcher.utils.token_tracker import TokenTracker


# --------------------------------------------------------------------------
# research_spend
# --------------------------------------------------------------------------

PER_TAG = {
    "subquery_generation": {"input": 10_000, "output": 1_000},
    "deep_process_results": {"input": 20_000, "output": 4_000},
    "orchestrator": {"input": 5_000, "output": 500},
    "report_generation": {"input": 30_000, "output": 6_000},
}


def test_spend_leaves_out_the_writer():
    assert dr.research_spend(PER_TAG) == 10_000 + 1_000 + 20_000 + 4_000 + 5_000 + 500


def test_spend_leaves_out_a_local_orchestrator_and_weights_output():
    spend = dr.research_spend(PER_TAG, output_weight=4.0, exclude_tags=frozenset({"report_generation", "orchestrator"}))
    assert spend == 10_000 + 4 * 1_000 + 20_000 + 4 * 4_000


# --------------------------------------------------------------------------
# DeepResearchSkill budget state, on a stub (no researcher needed)
# --------------------------------------------------------------------------


def _skill(budget, exclude=("report_generation",), force_full=True):
    s = SimpleNamespace(
        research_budget=budget,
        budget_output_weight=1.0,
        budget_exclude_tags=frozenset(exclude),
        budget_force_full=force_full,
        stop_requested=False,
        research_budget_info={
            "exhausted": False, "exhausted_at_round": None, "spend_at_exhaustion": None,
            "cancelled_subqueries": 0, "terminate_ignored": 0,
        },
        trajectory_logger=SimpleNamespace(trajectory=SimpleNamespace(rounds=[])),
    )
    s._research_spend = lambda: DeepResearchSkill._research_spend(s)
    s._research_budget_exhausted = lambda: DeepResearchSkill._research_budget_exhausted(s)
    return s


@pytest.fixture(autouse=True)
def _clean_tracker():
    TokenTracker.reset()
    yield
    TokenTracker.reset()


def test_not_exhausted_below_budget_and_off_without_one():
    TokenTracker.track_tokens("m", 900, 50, usage_tag="subquery_generation")
    assert DeepResearchSkill._research_budget_exhausted(_skill(1_000)) is False
    assert DeepResearchSkill._research_budget_exhausted(_skill(None)) is False


def test_exhausted_at_budget_stops_research_and_records_spend():
    TokenTracker.track_tokens("m", 900, 100, usage_tag="subquery_generation")
    s = _skill(1_000)
    assert DeepResearchSkill._research_budget_exhausted(s) is True
    assert s.stop_requested is True
    assert s.research_budget_info["exhausted"] is True
    assert s.research_budget_info["spend_at_exhaustion"] == 1_000


def test_writer_tokens_never_exhaust_the_research_budget():
    TokenTracker.track_tokens("m", 500, 0, usage_tag="subquery_generation")
    TokenTracker.track_tokens("m", 50_000, 8_000, usage_tag="report_generation")
    assert DeepResearchSkill._research_budget_exhausted(_skill(1_000)) is False


def test_local_orchestrator_tokens_do_not_count():
    TokenTracker.track_tokens("m", 500, 0, usage_tag="subquery_generation")
    TokenTracker.track_tokens("qwen", 9_000, 500, usage_tag="orchestrator")
    assert DeepResearchSkill._research_budget_exhausted(
        _skill(1_000, exclude=("report_generation", "orchestrator"))) is False
    assert DeepResearchSkill._research_budget_exhausted(_skill(1_000)) is True


def test_gather_cancels_unfinished_work_once_the_budget_is_reached():
    s = _skill(1_000)
    finished = []

    async def cheap():
        TokenTracker.track_tokens("m", 1_200, 0, usage_tag="deep_process_results")
        finished.append("cheap")
        return "cheap"

    async def slow():
        await asyncio.sleep(5)
        finished.append("slow")
        return "slow"

    out = asyncio.run(DeepResearchSkill._gather_within_budget(s, [cheap(), slow()]))
    assert out == ["cheap", None]
    assert finished == ["cheap"]
    assert s.research_budget_info["cancelled_subqueries"] == 1
    assert s.stop_requested is True


def test_gather_is_plain_without_a_budget():
    s = _skill(None)

    async def one(x):
        return x

    assert asyncio.run(DeepResearchSkill._gather_within_budget(s, [one(1), one(2)])) == [1, 2]


def test_max_context_words_env(monkeypatch):
    monkeypatch.delenv("GR_MAX_CONTEXT_WORDS", raising=False)
    assert dr._max_context_words() == dr.MAX_CONTEXT_WORDS
    monkeypatch.setenv("GR_MAX_CONTEXT_WORDS", "120000")
    assert dr._max_context_words() == 120_000


# --------------------------------------------------------------------------
# Fixed report length
# --------------------------------------------------------------------------


def test_report_word_target(monkeypatch):
    monkeypatch.delenv("GR_REPORT_WORDS", raising=False)
    assert report_word_target() is None
    monkeypatch.setenv("GR_REPORT_WORDS", "2000")
    assert report_word_target() == (2000, 1800, 2200)
    monkeypatch.setenv("GR_REPORT_WORDS_TOL", "0.05")
    assert report_word_target() == (2000, 1900, 2100)


def test_body_words_stop_at_the_reference_list():
    report = "# Title\n" + "word " * 50 + "\n## References\n" + "ref " * 30
    assert report_body_words(report) == 52


def test_deep_prompt_asks_for_the_target_instead_of_a_minimum(monkeypatch):
    monkeypatch.setenv("GR_REPORT_WORDS", "2000")
    p = PromptFamily.generate_deep_research_prompt("q", "ctx", "web", total_words=1200)
    assert "between 1800 and 2200 words" in p and "minimum length" not in p
    monkeypatch.delenv("GR_REPORT_WORDS")
    p = PromptFamily.generate_deep_research_prompt("q", "ctx", "web", total_words=1200)
    assert "minimum length of 1200 words" in p


# --------------------------------------------------------------------------
# Prompted orchestration: budget line and scores + thresholds
# --------------------------------------------------------------------------


def _vec(rho):
    return [rho, math.sqrt(max(0.0, 1 - rho * rho)), 0.0]


def _item(iid, rho):
    return PoolItem(item_id=iid, source_url=f"https://e.org/{iid}", text="x" * 400, embedding=_vec(rho),
                    tree_depth=1, retrieval_round=1, is_new=True)


def _inp(**kw):
    return OrchestrationInput(
        root_query="q", query_embedding=_vec(1.0), subquestions=["s1"],
        new_items=[_item("a1", 0.9), _item("b2", 0.5), _item("c3", 0.1)], retained_prev=[],
        frontier=[FrontierInfo("nA", "A"), FrontierInfo("nB", "B")],
        round_id=1, tree_depth=1, tokens_used=10, token_budget=None, **kw,
    )


async def _never(messages):  # pragma: no cover - not called
    raise AssertionError


def test_prompt_shows_the_research_budget_only_when_set():
    pol = PromptedPolicy(llm_call=_never)
    assert "Research budget" not in pol.build_prompt(_inp())
    p = pol.build_prompt(_inp(research_budget=100_000, research_spend=40_000))
    assert "Research budget: 100000 tokens in total; spent 40000; remaining 60000." in p


def test_scores_policy_thresholds_scores(monkeypatch):
    monkeypatch.setenv("GR_PROMPTED_KEEP_THRESHOLD", "6")
    monkeypatch.setenv("GR_PROMPTED_STOP_THRESHOLD", "3")
    pol = PromptedScoresPolicy(llm_call=_never)
    dec = pol.parse("SCORES: a1=9 b2=6 c3=2\nBRANCHES: nA=8 nB=2\nCONTINUE: 7", _inp())
    assert dec.kept_ids == {"a1", "b2"}
    assert dec.branch_allocation == pytest.approx({"nA": 0.8, "nB": 0.2})
    assert dec.terminate is False
    dec = pol.parse("SCORES: a1=9 b2=6 c3=2\nBRANCHES: nA=8 nB=2\nCONTINUE: 1", _inp())
    assert dec.terminate is True


def test_scores_policy_fallbacks_keep_and_continue():
    pol = PromptedScoresPolicy(llm_call=_never)
    dec = pol.parse("SCORES: a1=1\n", _inp())
    # Unscored items are kept, unscored branches weighted as 10, no CONTINUE = continue.
    assert dec.kept_ids == {"b2", "c3"}
    assert dec.branch_allocation == pytest.approx({"nA": 0.5, "nB": 0.5})
    assert dec.terminate is False
    assert dec.meta["missing_item_scores"] == 2
    dec = pol.parse("garbage", _inp())
    assert dec.meta["parse_failure"] is True and dec.kept_ids == {"a1", "b2", "c3"}


def test_build_policy_knows_prompted_scores():
    async def call(messages):
        return ""

    assert isinstance(build_policy("prompted_scores", llm_call=call), PromptedScoresPolicy)
    with pytest.raises(ValueError):
        build_policy("prompted_scores")

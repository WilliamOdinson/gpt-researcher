"""The research-budget line in the learned policy's state is opt-in, so prompts
of adapters trained without it stay byte-identical."""

from gpt_researcher.orchestration import learned
from gpt_researcher.orchestration.serialize import serialize_state

BASE = dict(root_query="q", subquestions=["a"], round_id=1, tree_depth=1, tokens_used=10,
            token_budget=None, items=[], frontier=[], feature_names=["sim"])


def test_no_budget_line_by_default():
    assert serialize_state(**BASE) == serialize_state(**BASE, research_budget=None, research_spend=None)
    assert "Research budget" not in serialize_state(**BASE)


def test_budget_line_when_given():
    text = serialize_state(**BASE, research_budget=200000, research_spend=150000.4)
    assert "Research budget 200000 | spent 150000 | remaining 50000" in text


def test_env_flag(monkeypatch):
    monkeypatch.delenv("GR_LEARNED_BUDGET_IN_STATE", raising=False)
    assert learned.budget_in_state() is False
    monkeypatch.setenv("GR_LEARNED_BUDGET_IN_STATE", "1")
    assert learned.budget_in_state() is True

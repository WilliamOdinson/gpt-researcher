"""Hard cap (GR_STOP_HARD_CAP): the report is made to fit in what the token
target leaves, so a query cannot end above GR_STOP_TOTAL_TOKENS."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpt_researcher.skills import deep_research as dr
from gpt_researcher.skills.deep_research import DeepResearchSkill


def _skill(target: int | None = 150_000, smart_token_limit: int = 12_000):
    return SimpleNamespace(
        stop_total_tokens=target,
        token_stop_info={},
        researcher=SimpleNamespace(query="q", cfg=SimpleNamespace(smart_token_limit=smart_token_limit)),
    )


def _items(n_words: int, size: int = 100) -> list[str]:
    return [" ".join(["word"] * size) for _ in range(n_words // size)]


def _fit(skill, items, used):
    builds = []

    def build(max_words):
        builds.append(max_words)
        return dr.trim_context_to_word_limit(items, max_words)

    with patch.object(dr, "_tokens_used", return_value=used):
        out = DeepResearchSkill._fit_report_to_budget(skill, list(items), build)
    return out, builds


def _report_prompt_bound(skill, out):
    """Upper bound on the report prompt the cap assumes."""
    return dr.REPORT_PROMPT_FIXED_TOKENS + 2 * dr._count_tokens("q") + dr._count_tokens(dr._context_text(out))


def test_context_left_alone_when_the_report_fits():
    skill = _skill()
    items = _items(5_000)
    out, builds = _fit(skill, items, used=100_000)
    assert out == items and builds == []
    info = skill.token_stop_info
    assert info["report_context_trimmed"] is False and info["hard_cap_feasible"] is True
    # Plenty left: max_tokens stays at the configured limit.
    assert skill.researcher.cfg.smart_token_limit == 12_000


@pytest.mark.parametrize("used", [120_000, 135_000, 140_000])
def test_context_trimmed_and_output_capped_so_the_total_cannot_pass_the_target(used):
    skill = _skill()
    items = _items(25_000)
    out, builds = _fit(skill, items, used=used)
    info = skill.token_stop_info
    assert info["report_context_trimmed"] is True and builds
    assert len(dr._context_text(out).split()) < 25_000
    max_out = skill.researcher.cfg.smart_token_limit
    assert max_out >= dr.REPORT_MIN_OUTPUT_TOKENS
    # Research + the largest prompt the cap allows + max_tokens <= target.
    assert used + _report_prompt_bound(skill, out) + max_out <= 150_000


def test_output_capped_to_the_remainder_when_it_is_below_the_configured_limit():
    skill = _skill()
    items = _items(1_000)
    used = 150_000 - 10_000
    out, _ = _fit(skill, items, used=used)
    max_out = skill.researcher.cfg.smart_token_limit
    assert max_out < 12_000
    assert used + _report_prompt_bound(skill, out) + max_out == 150_000


def test_infeasible_when_research_left_too_little():
    skill = _skill()
    out, _ = _fit(skill, _items(5_000), used=149_000)
    assert out == []
    assert skill.token_stop_info["hard_cap_feasible"] is False


def test_hard_cap_on_by_default_with_a_target_and_can_be_turned_off(monkeypatch):
    monkeypatch.delenv(dr.ENV_STOP_HARD_CAP, raising=False)
    assert DeepResearchSkill._hard_cap_on(_skill()) is True
    assert DeepResearchSkill._hard_cap_on(_skill(target=None)) is False
    monkeypatch.setenv(dr.ENV_STOP_HARD_CAP, "0")
    assert DeepResearchSkill._hard_cap_on(_skill()) is False


def test_token_count_matches_o200k():
    tiktoken = pytest.importorskip("tiktoken")
    text = "Source: https://example.com/a?b=1 Content: 42.5% of 1,000 units"
    assert dr._count_tokens(text) == len(tiktoken.get_encoding("o200k_base").encode(text))

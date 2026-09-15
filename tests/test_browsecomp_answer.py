"""BrowseComp short-answer synthesis (not a 2000-word deep-research report)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpt_researcher.actions.report_generation import (
    SHORT_ANSWER_MAX_TOKENS,
    SHORT_ANSWER_SYSTEM,
    generate_report,
)
from gpt_researcher.prompts import PromptFamily
from gpt_researcher.utils.enum import Tone


def test_browsecomp_prompt_asks_for_exact_answer_not_a_report():
    prompt = PromptFamily.generate_browsecomp_answer_prompt(
        "Which river runs through Munich?",
        "Source: bcp://74874\nContent: The Isar flows through Munich.\n",
    )
    assert "Exact Answer:" in prompt
    assert "Confidence:" in prompt
    assert "Explanation:" in prompt
    assert "Which river runs through Munich?" in prompt
    assert "bcp://" in prompt
    assert "2000" not in prompt
    assert "comprehensive research report" not in prompt.lower()


def _cfg(**extra):
    return SimpleNamespace(
        smart_llm_model="gpt",
        smart_llm_provider="openai",
        smart_token_limit=8000,
        llm_kwargs={},
        report_format="apa",
        total_words=2000,
        language="english",
        **extra,
    )


async def _capture(**kwargs):
    seen = {}

    async def fake_chat(**kw):
        seen.update(kw)
        return "Explanation: The Isar.\nExact Answer: Isar\nConfidence: 90%"

    with patch(
        "gpt_researcher.actions.report_generation.create_chat_completion",
        new=AsyncMock(side_effect=fake_chat),
    ):
        out = await generate_report(
            query="Which river runs through Munich?",
            context="Source: bcp://74874\nContent: The Isar flows through Munich.",
            agent_role_prompt="You are a senior research analyst. Write a long APA report.",
            report_type="deep",
            tone=Tone.Objective,
            report_source="web",
            websocket=None,
            cfg=_cfg(),
            **kwargs,
        )
    return out, seen


async def test_answer_format_browsecomp_overrides_deep_report_prompt():
    out, seen = await _capture(answer_format="browsecomp")
    assert "Exact Answer: Isar" in out
    user = seen["messages"][1]["content"]
    system = seen["messages"][0]["content"]
    assert system == SHORT_ANSWER_SYSTEM
    assert "Exact Answer:" in user
    assert "comprehensive research report" not in user.lower()
    assert seen["max_tokens"] == SHORT_ANSWER_MAX_TOKENS
    assert seen["max_tokens"] < 8000
    assert seen["temperature"] == 0.0


async def test_env_gr_answer_format_is_enough(monkeypatch):
    monkeypatch.setenv("GR_ANSWER_FORMAT", "browsecomp_plus")
    out, seen = await _capture()
    assert "Exact Answer:" in seen["messages"][1]["content"]
    assert seen["max_tokens"] == SHORT_ANSWER_MAX_TOKENS
    assert "Isar" in out


async def test_deep_report_type_without_format_keeps_long_prompt():
    _, seen = await _capture()
    user = seen["messages"][1]["content"]
    assert "comprehensive research report" in user.lower() or "minimum length" in user.lower()
    assert seen["max_tokens"] == 8000
    assert seen["temperature"] == 0.35


async def test_custom_prompt_does_not_win_over_browsecomp():
    """Harness used to pass a one-sentence custom_prompt; that still left the
    researcher system role and an 8k token budget. answer_format wins."""
    _, seen = await _capture(
        answer_format="browsecomp",
        custom_prompt="Answer in one short sentence. No headers.",
    )
    user = seen["messages"][1]["content"]
    assert "Exact Answer:" in user
    assert "one short sentence" not in user
    assert seen["messages"][0]["content"] == SHORT_ANSWER_SYSTEM

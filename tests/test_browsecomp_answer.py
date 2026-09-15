"""BrowseComp synthesis uses the official QUERY_TEMPLATE_NO_GET_DOCUMENT."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gpt_researcher.actions.report_generation import (
    OFFICIAL_COMPLETION_MAX_TOKENS,
    generate_report,
)
from gpt_researcher.prompts import QUERY_TEMPLATE_NO_GET_DOCUMENT, PromptFamily
from gpt_researcher.utils.enum import Tone


def test_browsecomp_prompt_is_official_query_template():
    prompt = PromptFamily.generate_browsecomp_answer_prompt(
        "Which river runs through Munich?",
        "Source: bcp://74874\nContent: The Isar flows through Munich.\n",
    )
    official = QUERY_TEMPLATE_NO_GET_DOCUMENT.format(
        Question="Which river runs through Munich?"
    )
    assert prompt.startswith(official)
    assert "bcp://74874" in prompt
    assert "You are a deep research agent" in prompt
    assert "cite your evidence documents inline" in prompt
    assert "2000" not in prompt
    assert "comprehensive research report" not in prompt.lower()
    assert "You have completed research" not in prompt
    assert "not a paragraph" not in prompt


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
    assert len(seen["messages"]) == 1
    assert seen["messages"][0]["role"] == "user"
    user = seen["messages"][0]["content"]
    assert user.startswith(
        QUERY_TEMPLATE_NO_GET_DOCUMENT.format(Question="Which river runs through Munich?")
    )
    assert "comprehensive research report" not in user.lower()
    assert seen["max_tokens"] == OFFICIAL_COMPLETION_MAX_TOKENS
    assert seen["temperature"] == 0.0


async def test_env_gr_answer_format_is_enough(monkeypatch):
    monkeypatch.setenv("GR_ANSWER_FORMAT", "browsecomp_plus")
    out, seen = await _capture()
    assert "You are a deep research agent" in seen["messages"][0]["content"]
    assert seen["max_tokens"] == OFFICIAL_COMPLETION_MAX_TOKENS
    assert "Isar" in out


async def test_deep_report_type_without_format_keeps_long_prompt():
    _, seen = await _capture()
    user = seen["messages"][1]["content"]
    assert "comprehensive research report" in user.lower() or "minimum length" in user.lower()
    assert seen["max_tokens"] == 8000
    assert seen["temperature"] == 0.35
    assert seen["messages"][0]["role"] == "system"


async def test_custom_prompt_does_not_win_over_browsecomp():
    """Harness used to pass a one-sentence custom_prompt; that still left the
    researcher system role and an 8k token budget. answer_format wins."""
    _, seen = await _capture(
        answer_format="browsecomp",
        custom_prompt="Answer in one short sentence. No headers.",
    )
    assert seen["messages"][0]["role"] == "user"
    user = seen["messages"][0]["content"]
    assert "Exact Answer:" in user
    assert "one short sentence" not in user
    assert "Write a long APA report" not in str(seen["messages"])

"""Reasoning-family models, including Azure deployment names, get no temperature.

Azure deployments are named after the model plus a suffix
(gpt-6.1-sol-1gtjfzlv8i), which the exact-match model lists cannot contain;
sending temperature=0.4 to them fails with HTTP 400 on every call.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpt_researcher.llm_provider.generic.base import (
    is_reasoning_family,
    supports_reasoning_effort,
    supports_temperature,
)
from gpt_researcher.utils.llm import create_chat_completion


@pytest.mark.parametrize("model", [
    "gpt-6.1-sol-1gtjfzlv8i", "gpt-6-luna-t8zcuw231b", "gpt-6.1-terra-e9kcfzynjl",
    "gpt-5", "gpt-5-mini", "gpt-5.5-pro", "o3", "o4-mini", "o1-2024-12-17", "openai/gpt-6.1-sol",
])
def test_reasoning_families_take_no_temperature(model):
    assert is_reasoning_family(model)
    assert not supports_temperature(model)
    assert supports_reasoning_effort(model)


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o-series-x", "gpt-50x", ""])
def test_other_openai_models_keep_temperature(model):
    assert not is_reasoning_family(model)
    assert supports_temperature(model)


def test_listed_non_openai_models_still_follow_the_lists():
    assert not supports_temperature("claude-opus-4-6")
    assert not supports_reasoning_effort("claude-opus-4-6")
    assert supports_temperature("claude-3-5-sonnet-20241022")


@pytest.mark.asyncio
async def test_azure_deployment_call_sends_no_temperature():
    provider = MagicMock()
    provider.get_chat_response = AsyncMock(return_value="ok")
    with patch("gpt_researcher.utils.llm.get_llm", return_value=provider) as get_llm:
        out = await create_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-6.1-sol-1gtjfzlv8i",
            temperature=0.4,
            llm_provider="openai",
            reasoning_effort="low",
        )
    assert out == "ok"
    kwargs = get_llm.call_args.kwargs
    assert kwargs["temperature"] is None
    assert kwargs["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_gpt_4o_call_keeps_temperature():
    provider = MagicMock()
    provider.get_chat_response = AsyncMock(return_value="ok")
    with patch("gpt_researcher.utils.llm.get_llm", return_value=provider) as get_llm:
        await create_chat_completion(
            messages=[{"role": "user", "content": "hi"}], model="gpt-4o", temperature=0.4, llm_provider="openai",
        )
    kwargs = get_llm.call_args.kwargs
    assert kwargs["temperature"] == 0.4
    assert "reasoning_effort" not in kwargs

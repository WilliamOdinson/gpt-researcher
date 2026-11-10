from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpt_researcher.utils.llm import create_chat_completion


SOL_DEPLOYMENT = "gpt-6.1-sol-1gtjfzlv8i"
LEARNED_ADAPTERS = ("pilot-bc-teacher", "pilot-dpo-teacher")


def _provider(response: str = "ok") -> MagicMock:
    provider = MagicMock()
    provider.get_chat_response = AsyncMock(return_value=response)
    provider.last_response_metadata = {}
    provider.last_usage_metadata = {"input_tokens": 1, "output_tokens": 1}
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ("strategic", "smart", "fast"))
async def test_sol_always_uses_flex_after_llm_kwargs_merge(role):
    provider = _provider(role)

    with patch("gpt_researcher.utils.llm.get_llm", return_value=provider) as get_llm:
        result = await create_chat_completion(
            messages=[{"role": "user", "content": role}],
            model=SOL_DEPLOYMENT,
            llm_provider="openai",
            llm_kwargs={"service_tier": "default"},
        )

    assert result == role
    assert get_llm.call_args.kwargs["service_tier"] == "flex"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", LEARNED_ADAPTERS)
async def test_learned_adapters_never_receive_service_tier(model):
    provider = _provider()

    with patch("gpt_researcher.utils.llm.get_llm", return_value=provider) as get_llm:
        await create_chat_completion(
            messages=[{"role": "user", "content": "choose evidence"}],
            model=model,
            llm_provider="openai",
            llm_kwargs={
                "service_tier": "flex",
                "openai_api_base": "https://adapter.invalid/v1",
                "openai_api_key": "EMPTY",
            },
        )

    kwargs = get_llm.call_args.kwargs
    assert "service_tier" not in kwargs
    assert kwargs["openai_api_base"] == "https://adapter.invalid/v1"


@pytest.mark.asyncio
async def test_other_models_keep_existing_service_tier_behavior():
    provider = _provider()

    with patch("gpt_researcher.utils.llm.get_llm", return_value=provider) as get_llm:
        await create_chat_completion(
            messages=[{"role": "user", "content": "x"}],
            model="gpt-4o-mini",
            llm_provider="openai",
            llm_kwargs={"service_tier": "default"},
        )

    assert get_llm.call_args.kwargs["service_tier"] == "default"

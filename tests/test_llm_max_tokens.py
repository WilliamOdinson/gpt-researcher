from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpt_researcher.utils.llm import create_chat_completion


@pytest.mark.asyncio
async def test_create_chat_completion_accepts_max_tokens_above_old_32k_cap():
    provider = MagicMock()
    provider.get_chat_response = AsyncMock(side_effect=["ok-64000", "ok-128000"])

    with patch("gpt_researcher.utils.llm.get_llm", return_value=provider) as mock_get_llm:
        accepted_values = (
            (64_000, "ok-64000"),
            (128_000, "ok-128000"),
        )

        for max_tokens, expected in accepted_values:
            result = await create_chat_completion(
                messages=[{"role": "user", "content": "Generate a report"}],
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                llm_provider="anthropic",
            )

            assert result == expected

    assert [call.kwargs["max_tokens"] for call in mock_get_llm.call_args_list] == [64_000, 128_000]


@pytest.mark.asyncio
async def test_create_chat_completion_rejects_absurd_max_tokens():
    with patch("gpt_researcher.utils.llm.get_llm") as mock_get_llm:
        with pytest.raises(ValueError, match="env vars|typos"):
            await create_chat_completion(
                messages=[{"role": "user", "content": "Generate a report"}],
                model="claude-sonnet-4-6",
                max_tokens=1_000_000,
                llm_provider="anthropic",
            )

    mock_get_llm.assert_not_called()


@pytest.mark.asyncio
async def test_create_chat_completion_does_not_clobber_explicit_openai_base(monkeypatch):
    """Learned orchestrator sets GR_ORCH_LLM_BASE_URL via llm_kwargs; the
    process still has Azure OPENAI_BASE_URL for researcher/embeddings."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.openai.azure.com/openai/v1")
    provider = MagicMock()
    provider.get_chat_response = AsyncMock(return_value="KEEP: ALL")
    provider.last_response_metadata = {}
    provider.last_usage_metadata = {"input_tokens": 1, "output_tokens": 1}

    with patch("gpt_researcher.utils.llm.get_llm", return_value=provider) as mock_get_llm:
        result = await create_chat_completion(
            messages=[{"role": "user", "content": "x"}],
            model="pilot-bc",
            max_tokens=16,
            temperature=0.0,
            llm_provider="openai",
            llm_kwargs={
                "openai_api_base": "https://tunnel.example/v1",
                "base_url": "https://tunnel.example/v1",
                "openai_api_key": "EMPTY",
            },
        )

    assert result == "KEEP: ALL"
    kwargs = mock_get_llm.call_args.kwargs
    assert kwargs["openai_api_base"] == "https://tunnel.example/v1"
    assert kwargs["base_url"] == "https://tunnel.example/v1"
    assert kwargs["openai_api_key"] == "EMPTY"


@pytest.mark.asyncio
async def test_create_chat_completion_uses_env_openai_base_when_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.openai.azure.com/openai/v1")
    provider = MagicMock()
    provider.get_chat_response = AsyncMock(return_value="ok")
    provider.last_response_metadata = {}
    provider.last_usage_metadata = {"input_tokens": 1, "output_tokens": 1}

    with patch("gpt_researcher.utils.llm.get_llm", return_value=provider) as mock_get_llm:
        await create_chat_completion(
            messages=[{"role": "user", "content": "x"}],
            model="gpt-4o-mini",
            max_tokens=16,
            temperature=0.0,
            llm_provider="openai",
        )

    assert mock_get_llm.call_args.kwargs["openai_api_base"] == (
        "https://example.openai.azure.com/openai/v1"
    )

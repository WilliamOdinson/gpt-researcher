"""Azure Flex handling in create_chat_completion: 429 no_capacity retries that
do not use up the normal attempts, a per-request timeout, and the per-call log."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpt_researcher.utils import llm as llm_mod
from gpt_researcher.utils.llm import create_chat_completion, is_capacity_error
from gpt_researcher.utils.llm_call_log import LLMCallLog
from gpt_researcher.utils.token_tracker import TokenTracker

SOL = "gpt-6.1-sol-1gtjfzlv8i"


class FakeAPIError(Exception):
    """Shaped like openai.APIStatusError: status_code, code, body."""

    def __init__(self, status_code: int, code: str | None, message: str):
        super().__init__(f"Error code: {status_code} - {message}")
        self.status_code = status_code
        self.code = code
        self.body = {"code": code, "message": message}


def capacity_error() -> FakeAPIError:
    return FakeAPIError(429, "no_capacity", "No capacity available for the flex tier")


def rate_limit_error() -> FakeAPIError:
    return FakeAPIError(429, "rate_limit_exceeded", "Rate limit reached")


def _provider(side_effect) -> MagicMock:
    provider = MagicMock()
    provider.get_chat_response = AsyncMock(side_effect=side_effect)
    provider.last_response_metadata = {"service_tier": "flex"}
    provider.last_usage_metadata = {"input_tokens": 11, "output_tokens": 7}
    return provider


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    LLMCallLog.reset()
    TokenTracker.reset()
    monkeypatch.delenv(llm_mod.ENV_REQUEST_TIMEOUT_S, raising=False)
    monkeypatch.delenv(llm_mod.ENV_CAPACITY_MAX_WAIT_S, raising=False)
    monkeypatch.delenv("LLM_KWARGS", raising=False)
    # Backoff sleeps are instant; the request timeout uses asyncio.wait_for,
    # not sleep, so it still applies.
    monkeypatch.setattr(llm_mod.asyncio, "sleep", AsyncMock(return_value=None))
    yield
    LLMCallLog.reset()
    TokenTracker.reset()


async def _call(provider, model=SOL):
    with patch("gpt_researcher.utils.llm.get_llm", return_value=provider) as get_llm:
        out = await create_chat_completion(
            messages=[{"role": "user", "content": "x"}],
            model=model,
            llm_provider="openai",
            usage_tag="t",
        )
    return out, get_llm


# ------------------------------------------------------------------ classifier


def test_is_capacity_error():
    assert is_capacity_error(capacity_error())
    assert is_capacity_error(Exception("Error code: 429 - {'error': {'code': 'no_capacity'}}"))
    assert not is_capacity_error(rate_limit_error())
    assert not is_capacity_error(FakeAPIError(500, "no_capacity", "server error"))
    assert not is_capacity_error(RuntimeError("boom"))


# ------------------------------------------------------------------ capacity retries


async def test_capacity_retries_do_not_use_normal_attempts():
    # More capacity rejections than there are normal attempts, then success.
    n = llm_mod.LLM_MAX_ATTEMPTS + 5
    provider = _provider([capacity_error()] * n + ["answer"])

    out, _ = await _call(provider)

    assert out == "answer"
    (rec,) = LLMCallLog.get_records()
    assert rec["status"] == "ok"
    assert rec["attempts"] == 1
    assert rec["capacity_retries"] == n
    assert rec["capacity_wait_s"] >= 0.0
    # Tokens are charged once, for the successful request.
    assert TokenTracker.get_totals()["input_tokens"] == 11


async def test_capacity_wait_budget_spent_falls_back_to_normal_attempts(monkeypatch):
    monkeypatch.setenv(llm_mod.ENV_CAPACITY_MAX_WAIT_S, "0")
    provider = _provider([capacity_error()] * llm_mod.LLM_MAX_ATTEMPTS)

    with pytest.raises(RuntimeError):
        await _call(provider)

    (rec,) = LLMCallLog.get_records()
    assert rec["status"] == "failed"
    assert rec["attempts"] == llm_mod.LLM_MAX_ATTEMPTS
    assert rec["capacity_retries"] == 0
    assert "no_capacity" in rec["error"] or "429" in rec["error"]


async def test_ordinary_rate_limit_uses_normal_attempts():
    provider = _provider([rate_limit_error(), rate_limit_error(), "answer"])
    out, _ = await _call(provider)
    assert out == "answer"
    (rec,) = LLMCallLog.get_records()
    assert rec["attempts"] == 3
    assert rec["capacity_retries"] == 0


# ------------------------------------------------------------------ timeout


async def test_hung_request_times_out_and_counts_as_an_attempt(monkeypatch):
    monkeypatch.setenv(llm_mod.ENV_REQUEST_TIMEOUT_S, "0.05")
    calls = {"n": 0}

    async def respond(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.Event().wait()  # never set: a hung request
        return "answer"

    provider = _provider(None)
    provider.get_chat_response = respond

    out, get_llm = await _call(provider)

    assert out == "answer"
    (rec,) = LLMCallLog.get_records()
    assert rec["timeouts"] == 1
    assert rec["attempts"] == 2
    # The client gets the same timeout, and its own silent retries are off.
    kwargs = get_llm.call_args.kwargs
    assert kwargs["timeout"] == 0.05
    assert kwargs["max_retries"] == 0


async def test_timeout_zero_disables_it(monkeypatch):
    monkeypatch.setenv(llm_mod.ENV_REQUEST_TIMEOUT_S, "0")
    _, get_llm = await _call(_provider(["answer"]))
    assert "timeout" not in get_llm.call_args.kwargs


async def test_client_timeout_error_is_counted_as_a_timeout():
    class APITimeoutError(Exception):
        pass

    provider = _provider([APITimeoutError("Request timed out."), "answer"])
    await _call(provider)
    (rec,) = LLMCallLog.get_records()
    assert rec["timeouts"] == 1 and rec["attempts"] == 2


# ------------------------------------------------------------------ call log


async def test_call_log_records_timing_tokens_and_tier():
    await _call(_provider(["answer"]))
    (rec,) = LLMCallLog.get_records()
    assert rec["model"] == SOL
    assert rec["usage_tag"] == "t"
    assert rec["tier_requested"] == "flex"  # forced for Sol
    assert rec["tier_served"] == "flex"
    assert rec["input_tokens"] == 11 and rec["output_tokens"] == 7
    assert rec["end"] >= rec["start"]
    assert rec["duration_s"] >= 0.0
    assert rec["status"] == "ok" and rec["error"] is None

    summary = LLMCallLog.summary()
    assert summary["n_calls"] == 1 and summary["n_failed"] == 0
    assert summary["capacity_wait_s"] == 0.0


async def test_explicit_llm_kwargs_override_retry_and_timeout_defaults():
    provider = _provider(["answer"])
    with patch("gpt_researcher.utils.llm.get_llm", return_value=provider) as get_llm:
        await create_chat_completion(
            messages=[{"role": "user", "content": "x"}],
            model=SOL,
            llm_provider="openai",
            llm_kwargs={"max_retries": 3, "timeout": 30},
        )
    kwargs = get_llm.call_args.kwargs
    assert kwargs["max_retries"] == 3 and kwargs["timeout"] == 30

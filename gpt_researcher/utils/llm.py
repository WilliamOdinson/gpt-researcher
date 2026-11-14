"""LLM utilities for GPT Researcher.

This module provides utility functions for interacting with various
LLM providers through a unified interface.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any
import asyncio

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from gpt_researcher.llm_provider.generic.base import (
    NO_SUPPORT_TEMPERATURE_MODELS,
    SUPPORT_REASONING_EFFORT_MODELS,
    ReasoningEfforts,
)

from ..prompts import PromptFamily
from .costs import calculate_llm_cost
from .latency_tracker import LatencyTracker
from .llm_call_log import LLMCallLog
from .token_tracker import TokenTracker
from .validators import Subtopics


_AZURE_FLEX_MODELS = {"gpt-6.1-sol-1gtjfzlv8i"}
_LEARNED_ORCHESTRATOR_MODELS = {"pilot-bc-teacher", "pilot-dpo-teacher"}

# Normal attempts per call (errors, empty responses, timeouts).
LLM_MAX_ATTEMPTS = 10
# GR_LLM_REQUEST_TIMEOUT_S: seconds before one request is abandoned and counted
# as a failed attempt, so a hung call cannot stall a run. 0 = no timeout.
ENV_REQUEST_TIMEOUT_S = "GR_LLM_REQUEST_TIMEOUT_S"
DEFAULT_REQUEST_TIMEOUT_S = 300.0
# GR_LLM_CAPACITY_MAX_WAIT_S: total seconds one call may spend waiting out
# Azure Flex "429 no_capacity" rejections. These retries do not use up the
# normal attempts; once the wait budget is spent the call fails.
ENV_CAPACITY_MAX_WAIT_S = "GR_LLM_CAPACITY_MAX_WAIT_S"
DEFAULT_CAPACITY_MAX_WAIT_S = 3600.0
CAPACITY_BACKOFF_START_S = 5.0
CAPACITY_BACKOFF_MAX_S = 60.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logging.getLogger(__name__).warning(f"{name}={raw!r} is not a number; using {default}")
        return default


def is_capacity_error(exc: BaseException) -> bool:
    """True for a Flex capacity rejection: HTTP 429 whose error code or message
    says ``no_capacity``. Ordinary 429 rate limits are not capacity errors and
    use the normal attempts."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    text = str(exc)
    if status != 429 and "Error code: 429" not in text:
        return False
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code.lower() == "no_capacity":
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and str(err.get("code", "")).lower() == "no_capacity":
            return True
    return "no_capacity" in text.lower()


def get_llm(llm_provider: str, **kwargs):
    """Get an LLM provider instance.

    Args:
        llm_provider: The name of the LLM provider (e.g., 'openai', 'anthropic').
        **kwargs: Additional keyword arguments passed to the provider.

    Returns:
        A GenericLLMProvider instance configured for the specified provider.
    """
    from gpt_researcher.llm_provider import GenericLLMProvider
    return GenericLLMProvider.from_provider(llm_provider, **kwargs)


async def create_chat_completion(
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = 0.4,
        max_tokens: int | None = 4000,
        llm_provider: str | None = None,
        stream: bool = False,
        websocket: Any | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        cost_callback: callable = None,
        reasoning_effort: str | None = ReasoningEfforts.Medium.value,
        usage_tag: str | None = None,
        **kwargs
) -> str:
    """Create a chat completion using the OpenAI API
    Args:
        messages (list[dict[str, str]]): The messages to send to the chat completion.
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.4.
        max_tokens (int, optional): The max tokens to use. Defaults to 4000.
        llm_provider (str, optional): The LLM Provider to use.
        stream (bool): Whether to stream the response. Defaults to False.
        webocket (WebSocket): The websocket used in the currect request,
        llm_kwargs (dict[str, Any], optional): Additional LLM keyword arguments. Defaults to None.
        cost_callback: Callback function for updating cost.
        reasoning_effort (str, optional): Reasoning effort for OpenAI's reasoning models. Defaults to 'low'.
        **kwargs: Additional keyword arguments.
    Returns:
        str: The response from the chat completion.
    """
    # validate input
    if model is None:
        raise ValueError("Model cannot be None")
    # Sanity guard against absurd values (e.g., env var typos). The actual
    # per-model output limits are enforced by the upstream provider.
    if max_tokens is not None and max_tokens > 200_000:
        raise ValueError(
            f"max_tokens={max_tokens} exceeds the largest output limit of "
            "any currently available model (128k as of late 2025). "
            "Check your FAST_TOKEN_LIMIT / SMART_TOKEN_LIMIT / "
            "STRATEGIC_TOKEN_LIMIT env vars for typos."
        )

    # Get the provider from supported providers
    provider_kwargs = {'model': model}

    if llm_kwargs:
        provider_kwargs.update(llm_kwargs)
    elif os.environ.get("LLM_KWARGS"):
        import json
        try:
            provider_kwargs.update(json.loads(os.environ["LLM_KWARGS"]))
        except json.JSONDecodeError:
            pass

    # The DRB Sol deployment must always request Azure Flex, even if a stale
    # LLM_KWARGS value asks for another tier. Conversely, the learned BC/DPO
    # adapters are served by a separate OpenAI-compatible vLLM endpoint and
    # must never receive Azure's service_tier request parameter.
    if llm_provider == "openai" and model in _AZURE_FLEX_MODELS:
        provider_kwargs["service_tier"] = "flex"
    elif model in _LEARNED_ORCHESTRATOR_MODELS:
        provider_kwargs.pop("service_tier", None)

    if model in SUPPORT_REASONING_EFFORT_MODELS:
        provider_kwargs['reasoning_effort'] = reasoning_effort

    if model not in NO_SUPPORT_TEMPERATURE_MODELS:
        provider_kwargs['temperature'] = temperature
    else:
        # These models enforce their default temperature, but output limits
        # still apply (langchain-openai maps max_tokens to the API's
        # max_completion_tokens). Note that for reasoning models the limit
        # covers reasoning tokens too, so budgets need extra headroom.
        provider_kwargs['temperature'] = None
    provider_kwargs['max_tokens'] = max_tokens

    if llm_provider == "openai":
        # Do not clobber an explicit base (GR_ORCH_LLM_BASE_URL / llm_kwargs).
        # The process also has OPENAI_BASE_URL for Azure researcher/embeddings;
        # overwriting here sent the learned orchestrator to Azure with key EMPTY.
        if not provider_kwargs.get("openai_api_base") and not provider_kwargs.get("base_url"):
            base_url = os.environ.get("OPENAI_BASE_URL", None)
            if base_url:
                provider_kwargs["openai_api_base"] = base_url

    request_timeout_s = _env_float(ENV_REQUEST_TIMEOUT_S, DEFAULT_REQUEST_TIMEOUT_S)
    capacity_max_wait_s = _env_float(ENV_CAPACITY_MAX_WAIT_S, DEFAULT_CAPACITY_MAX_WAIT_S)
    if llm_provider == "openai":
        # Retries happen in the loop below, where each attempt is counted and
        # Flex capacity rejections are told apart; the client's own silent
        # retries would hide both. An explicit llm_kwargs value still wins.
        provider_kwargs.setdefault("max_retries", 0)
        if request_timeout_s > 0:
            provider_kwargs.setdefault("timeout", request_timeout_s)

    provider = get_llm(llm_provider, **provider_kwargs)
    response = ""
    _llm_start = time.time()
    max_attempts = 1 if (stream and websocket is not None) else LLM_MAX_ATTEMPTS
    last_exception: Exception | None = None
    log = logging.getLogger(__name__)
    call = {
        "model": model,
        "usage_tag": usage_tag,
        "tier_requested": provider_kwargs.get("service_tier"),
        "tier_served": None,
        "start": _llm_start,
        "end": None,
        "duration_s": None,
        "attempts": 0,
        "capacity_retries": 0,
        "capacity_wait_s": 0.0,
        "timeouts": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "status": "failed",
        "error": None,
    }

    def _finish(status: str, error: BaseException | None = None) -> None:
        end = time.time()
        call.update(
            end=end,
            duration_s=round(end - _llm_start, 4),
            capacity_wait_s=round(call["capacity_wait_s"], 4),
            status=status,
            error=None if error is None else f"{type(error).__name__}: {error}"[:500],
        )
        LLMCallLog.record(call)

    attempt = 0
    capacity_backoff = CAPACITY_BACKOFF_START_S
    while attempt < max_attempts:
        attempt += 1
        call["attempts"] = attempt
        t_req = time.time()
        try:
            request = provider.get_chat_response(messages, stream, websocket, **kwargs)
            if request_timeout_s > 0:
                response = await asyncio.wait_for(request, timeout=request_timeout_s)
            else:
                response = await request
        except asyncio.TimeoutError as exc:
            last_exception = exc
            call["timeouts"] += 1
            log.warning(
                f"LLM request timed out after {request_timeout_s:.0f}s "
                f"(attempt {attempt}/{max_attempts})"
            )
            if attempt < max_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
            continue
        except Exception as exc:
            last_exception = exc
            if is_capacity_error(exc) and call["capacity_wait_s"] < capacity_max_wait_s:
                # Flex has no capacity right now: wait and retry without using
                # up a normal attempt, until the wait budget is spent.
                attempt -= 1
                sleep_s = min(capacity_backoff, max(0.0, capacity_max_wait_s - call["capacity_wait_s"]))
                call["capacity_retries"] += 1
                log.warning(
                    f"LLM capacity unavailable (429 no_capacity, retry {call['capacity_retries']}); "
                    f"waiting {sleep_s:.0f}s"
                )
                await asyncio.sleep(sleep_s)
                call["capacity_wait_s"] += time.time() - t_req
                capacity_backoff = min(capacity_backoff * 2, CAPACITY_BACKOFF_MAX_S)
                continue
            if "timeout" in type(exc).__name__.lower():
                # The client's own timeout (e.g. openai.APITimeoutError).
                call["timeouts"] += 1
            log.warning(f"LLM request failed (attempt {attempt}/{max_attempts}): {exc}")
            if attempt < max_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
            continue

        if not response:
            last_exception = RuntimeError("Empty response from LLM provider")
            log.warning(f"LLM returned empty response (attempt {attempt}/{max_attempts})")
            if attempt < max_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
            continue

        llm_costs = calculate_llm_cost(
            llm_provider=llm_provider,
            model=model,
            input_content=str(messages),
            output_content=response,
            response_metadata=provider.last_response_metadata,
            usage_metadata=provider.last_usage_metadata,
            request_options=provider_kwargs,
        )
        if cost_callback:
            cost_callback(llm_costs)

        _input_tokens = 0
        _output_tokens = 0
        usage_meta = provider.last_usage_metadata
        if isinstance(usage_meta, dict):
            _input_tokens = usage_meta.get("input_tokens", 0) or usage_meta.get("prompt_tokens", 0) or 0
            _output_tokens = usage_meta.get("output_tokens", 0) or usage_meta.get("completion_tokens", 0) or 0
        TokenTracker.track_tokens(model, _input_tokens, _output_tokens, llm_costs, usage_tag=usage_tag)
        LatencyTracker.track_latency("llm", time.time() - _llm_start, source=model)
        meta = provider.last_response_metadata
        call.update(
            input_tokens=int(_input_tokens),
            output_tokens=int(_output_tokens),
            tier_served=meta.get("service_tier") if isinstance(meta, dict) else None,
        )
        _finish("ok")

        return response

    _finish("failed", last_exception)
    logging.error(f"Failed to get response from {llm_provider} API")
    raise RuntimeError(f"Failed to get response from {llm_provider} API") from last_exception


async def construct_subtopics(
    task: str,
    data: str,
    config,
    subtopics: list = [],
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> list:
    """
    Construct subtopics based on the given task and data.

    Args:
        task (str): The main task or topic.
        data (str): Additional data for context.
        config: Configuration settings.
        subtopics (list, optional): Existing subtopics. Defaults to [].
        prompt_family (PromptFamily): Family of prompts
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of constructed subtopics.
    """
    try:
        parser = PydanticOutputParser(pydantic_object=Subtopics)

        prompt = PromptTemplate(
            template=prompt_family.generate_subtopics_prompt(),
            input_variables=["task", "data", "subtopics", "max_subtopics"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()},
        )

        provider_kwargs = {'model': config.smart_llm_model}

        if config.llm_kwargs:
            provider_kwargs.update(config.llm_kwargs)

        if config.smart_llm_model in SUPPORT_REASONING_EFFORT_MODELS:
            provider_kwargs['reasoning_effort'] = ReasoningEfforts.High.value
        else:
            provider_kwargs['temperature'] = config.temperature
        provider_kwargs['max_tokens'] = config.smart_token_limit

        provider = get_llm(config.smart_llm_provider, **provider_kwargs)

        model = provider.llm

        chain = prompt | model | parser

        output = await chain.ainvoke({
            "task": task,
            "data": data,
            "subtopics": subtopics,
            "max_subtopics": config.max_subtopics
        }, **kwargs)

        return output

    except Exception as e:
        logging.getLogger(__name__).error(
            "Exception in parsing subtopics: %s", e, exc_info=True
        )
        return subtopics

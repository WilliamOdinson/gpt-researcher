"""Orchestration policies for deep research.

A policy is called once per deep-research round (one tree level finishing) and
returns the orchestration action ``(u, m, w)``:

  u  terminate       stop descending and write the report now
  m  kept_ids        which evidence items (new and previously retained) survive
  w  branch_allocation  how the next level's search breadth is split across
                        the open frontier nodes

Select a policy with the ``GR_ORCHESTRATOR`` environment variable:

  unset / legacy   pre-existing behaviour: the checkpoint only records what the
                   per-sub-query EmbeddingsFilter kept; nothing is re-decided
  none             keep everything; uniform allocation; never terminate early
  topk             rank items by cosine to the root query, keep under budget
  extractive       keep every item, trim each to its most on-topic chunks
  llmlingua        keep every item, compress each with LLMLingua-2
  prompted         one LLM call decides KEEP / ALLOC / DECISION

``GR_CONTEXT_BUDGET_TOKENS`` (optional) caps the retained evidence per round,
in approximate tokens (``len(text) // 4``). Used by topk / extractive /
llmlingua and shown to the prompted policy.
"""

from .policies import (  # noqa: F401
    DEFAULT_POLICY,
    ENV_BUDGET as ENV_BUDGET_NAME,
    ENV_POLICY as ENV_POLICY_NAME,
    POLICY_NAMES,
    ExtractivePolicy,
    FrontierInfo,
    LLMLinguaPolicy,
    NoPruningPolicy,
    OrchestrationDecision,
    OrchestrationInput,
    OrchestrationPolicy,
    PoolItem,
    PromptedPolicy,
    TopKPolicy,
    allocate_breadth,
    build_policy,
    estimate_tokens,
    policy_name_from_env,
)

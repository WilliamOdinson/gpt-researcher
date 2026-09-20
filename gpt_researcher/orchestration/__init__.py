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
  random           data-collection policy for behavior cloning: samples keep
                   ratio, scoring weights, allocation concentration and stop
                   rule once per query (``GR_ORCH_SEED``) and logs per-item
                   features into the trajectory

``GR_CONTEXT_BUDGET_TOKENS`` (optional) caps the retained evidence per round,
in approximate tokens (``len(text) // 4``). Used by topk / extractive /
llmlingua / random and shown to the prompted policy.
"""

from .features import (  # noqa: F401
    FEATURE_NAMES,
    FeatureBundle,
    FrontierRow,
    ItemRow,
    compute_features,
    coverage_potential_for,
)
from .randomized import PARAM_SPACE, RandomizedPolicy, derive_seed, sample_params  # noqa: F401
from .serialize import (  # noqa: F401
    SYSTEM_PROMPT,
    ParsedAction,
    StateFrontier,
    StateItem,
    parse_action,
    serialize_action,
    serialize_state,
)
from .policies import (  # noqa: F401
    DEFAULT_POLICY,
    ENV_BUDGET as ENV_BUDGET_NAME,
    ENV_RANDOM_PARAMS as ENV_RANDOM_PARAMS_NAME,
    ENV_SEED as ENV_SEED_NAME,
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

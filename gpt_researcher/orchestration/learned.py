"""LearnedPolicy: the behavior-cloned orchestrator (PILOT §4.6, Stage 1 → online).

Online counterpart of the harness's ``bc_pairs`` builder. Each round it

  1. computes the same per-item features the RandomizedPolicy logged
     (``features.compute_features``),
  2. serializes the state with ``serialize.serialize_state`` — byte-identical
     to the prompt the model was fine-tuned on,
  3. sends ``[SYSTEM_PROMPT, state]`` to the fine-tuned model through the
     callable it is given, and
  4. parses the three-line KEEP / ALLOC / DECISION answer with
     ``serialize.parse_action`` (shared with ``PromptedPolicy``).

Parse failures and transport errors degrade to keep-all / uniform / continue
and are flagged in ``decision.meta`` so a run never crashes on a bad sample.
Everything the BC builder needs (features, sources, frontier stats, Phi) is
logged to ``meta`` too, so rollouts under this policy can be re-scored and
re-cloned (DAgger-style) with the same tooling as the random corpus.

The model endpoint is chosen by ``DeepResearchSkill._orchestrator_llm_call``;
see ``GR_ORCH_LLM_*`` there.
"""

from __future__ import annotations

import logging
from typing import Any

from .features import FrontierRow, ItemRow, compute_features, coverage_potential_for
from .policies import (
    LLMCall,
    OrchestrationDecision,
    OrchestrationInput,
    OrchestrationPolicy,
    PoolItem,
    uniform_allocation,
)
from .serialize import SYSTEM_PROMPT, StateFrontier, StateItem, parse_action, serialize_state

logger = logging.getLogger(__name__)


def _rows(items: list[PoolItem]) -> list[ItemRow]:
    return [
        ItemRow(
            item_id=it.item_id,
            source_url=it.source_url,
            text=it.text,
            embedding=it.embedding,
            tree_depth=it.tree_depth,
            retrieval_round=it.retrieval_round,
            is_new=it.is_new,
            source_subquery=it.source_subquery or "",
        )
        for it in items
    ]


class LearnedPolicy(OrchestrationPolicy):
    name = "learned"
    needs_embeddings = True
    needs_frontier_embeddings = True

    def __init__(
        self,
        llm_call: LLMCall,
        budget: int | None = None,
        *,
        snippet_chars: int | None = None,
    ) -> None:
        self.llm_call = llm_call
        self.budget = budget
        self.snippet_chars = snippet_chars

    # -- prompt --------------------------------------------------------------

    def build_messages(self, inp: OrchestrationInput) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Return the chat messages and the feature bundle meta for this round."""
        pool = inp.pool
        rows = _rows(pool)
        frontier_rows = [FrontierRow(f.node_id, f.subquery, f.embedding) for f in inp.frontier]
        bundle = compute_features(
            query_embedding=inp.query_embedding,
            subquestion_embeddings=inp.subquestion_embeddings,
            items=rows,
            frontier=frontier_rows,
            round_id=inp.round_id,
        )
        items = [
            StateItem(
                item_id=it.item_id,
                features=bundle.features[it.item_id],
                is_new=it.is_new,
                text=it.text,
                source=bundle.sources.get(it.item_id, ""),
            )
            for it in pool
        ]
        frontier = [
            StateFrontier(
                node_id=f.node_id,
                subquery=f.subquery,
                gap=(bundle.frontier.get(f.node_id) or {}).get("gap"),
            )
            for f in inp.frontier
        ]
        budget = self.budget if self.budget is not None else inp.token_budget
        kwargs: dict[str, Any] = dict(
            root_query=inp.root_query,
            subquestions=list(inp.subquestions),
            round_id=inp.round_id,
            tree_depth=inp.tree_depth,
            tokens_used=inp.tokens_used,
            token_budget=budget,
            items=items,
            frontier=frontier,
            feature_names=bundle.names,
        )
        if self.snippet_chars is not None:
            kwargs["snippet_chars"] = self.snippet_chars
        prompt = serialize_state(**kwargs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        meta: dict[str, Any] = {
            "policy": self.name,
            "n_pool": len(pool),
            "n_new": len(inp.new_items),
            "tokens_used": inp.tokens_used,
            "token_budget": budget,
            "prompt_chars": len(prompt),
        }
        meta.update(bundle.as_meta())
        meta["_rows"] = rows  # consumed by decide(); stripped before return
        return messages, meta

    # -- decision -------------------------------------------------------------

    async def decide(self, inp: OrchestrationInput) -> OrchestrationDecision:
        messages, meta = self.build_messages(inp)
        rows = meta.pop("_rows")
        pool_ids = [it.item_id for it in inp.pool]
        node_ids = [f.node_id for f in inp.frontier]

        try:
            text = await self.llm_call(messages)
        except Exception as exc:  # the row must not crash the run
            logger.warning("Learned orchestrator LLM call failed: %s", exc)
            kept = set(pool_ids)
            meta.update({"parse_failure": True, "error": str(exc), "raw": ""})
            meta["phi_after"] = round(float(coverage_potential_for(kept, rows, inp.subquestion_embeddings)), 4)
            return OrchestrationDecision(
                terminate=False,
                kept_ids=kept,
                branch_allocation=uniform_allocation(inp.frontier),
                meta=meta,
            )

        parsed = parse_action(text, pool_ids, node_ids)
        meta.update(parsed.meta)
        meta["phi_after"] = round(
            float(coverage_potential_for(parsed.kept_ids, rows, inp.subquestion_embeddings)), 4
        )
        meta["terminate_reason"] = "model" if parsed.terminate else ""
        return OrchestrationDecision(
            terminate=parsed.terminate,
            kept_ids=parsed.kept_ids,
            branch_allocation=parsed.branch_allocation,
            meta=meta,
        )

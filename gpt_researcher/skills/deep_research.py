from typing import List, Dict, Any, Optional, Set
import asyncio
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta

import json_repair
import numpy as np

from gpt_researcher.llm_provider.generic.base import ReasoningEfforts
from ..utils.latency_tracker import LatencyTracker
from ..utils.llm import create_chat_completion
from ..utils.enum import ReportType, ReportSource, Tone
from ..utils.scraper_cache import _global_scraper_cache
from ..utils.search_cache import _global_search_cache
from ..utils.token_tracker import TokenTracker
from ..utils.llm_call_log import LLMCallLog
from ..utils.trajectory_logger import (
    TrajectoryLogger, FrontierNode, RoundCost, make_item_id,
)
from ..context import _global_embedding_cache
from ..actions.query_processing import get_search_results
from ..orchestration import (
    ENV_BUDGET_NAME,
    FrontierInfo,
    OrchestrationInput,
    PoolItem,
    allocate_breadth,
    build_policy,
    estimate_tokens,
    policy_name_from_env,
)

logger = logging.getLogger(__name__)


def _unit_mean(vectors: list[list[float]]) -> list[float]:
    """Mean of vectors, renormalized to unit length."""
    m = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
    n = np.linalg.norm(m)
    return (m / n if n > 0 else m).tolist()

# Maximum words allowed in context (25k words for safety margin)
MAX_CONTEXT_WORDS = 25000

JSON_BLOCK_PATTERNS = [
    re.compile(
        r"```(?:json)?\s*(?P<payload>[\s\S]*?)```",
        re.IGNORECASE,
    ),
    re.compile(r"(?P<payload>\[[\s\S]*\])"),
    re.compile(r"(?P<payload>\{[\s\S]*\})"),
]

QUERY_LINE_PATTERN = re.compile(
    r"^(?:[-*]|\d+[.)])?\s*Query:\s*(?P<query>.+)$",
    re.IGNORECASE,
)
GOAL_LINE_PATTERN = re.compile(
    r"^(?:[-*]|\d+[.)])?\s*(?:Goal|Research Goal):\s*(?P<goal>.+)$",
    re.IGNORECASE,
)
QUESTION_LINE_PATTERN = re.compile(
    r"^(?:[-*]|\d+[.)])?\s*(?:Question:\s*)?(?P<question>.+\?)$",
    re.IGNORECASE,
)
LEARNING_LINE_PATTERN = re.compile(
    r"^(?:[-*]|\d+[.)])?\s*Learning(?:\s*\[(?P<citation>[^\]]+)\])?:\s*(?P<learning>.+)$",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://[^\s\]\)>\",;]+")


def _extract_json_payloads(response: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    for pattern in JSON_BLOCK_PATTERNS:
        for match in pattern.finditer(response):
            candidate = match.group("payload").strip()
            if candidate and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

    return candidates


def _load_repaired_json(response: str) -> Any:
    for candidate in [response.strip(), *_extract_json_payloads(response)]:
        if not candidate:
            continue
        try:
            return json_repair.loads(candidate)
        except Exception as exc:
            logger.debug(
                "json_repair failed on candidate (%d chars): %s",
                len(candidate), exc,
            )
            continue
    return None


def parse_search_queries_response(response: str, num_queries: int) -> List[Dict[str, str]]:
    parsed = _load_repaired_json(response)
    candidate_queries = parsed
    if isinstance(parsed, dict):
        candidate_queries = parsed.get("queries") or parsed.get("searchQueries") or parsed.get("items")

    if isinstance(candidate_queries, list):
        queries = [
            {
                "query": item["query"].strip(),
                "researchGoal": item["researchGoal"].strip(),
            }
            for item in candidate_queries
            if isinstance(item, dict) and item.get("query") and item.get("researchGoal")
        ]
        if queries:
            return queries[:num_queries]

    queries: List[Dict[str, str]] = []
    current_query: Dict[str, str] = {}

    for raw_line in response.replace("```json", "").replace("```", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        query_match = QUERY_LINE_PATTERN.match(line)
        goal_match = GOAL_LINE_PATTERN.match(line)

        if query_match:
            if current_query.get("query") and current_query.get("researchGoal"):
                queries.append(current_query)
            current_query = {"query": query_match.group("query").strip()}
        elif goal_match and current_query.get("query"):
            current_query["researchGoal"] = goal_match.group("goal").strip()

    if current_query.get("query") and current_query.get("researchGoal"):
        queries.append(current_query)

    return queries[:num_queries]


def parse_follow_up_questions_response(response: str, num_questions: int) -> List[str]:
    parsed = _load_repaired_json(response)
    candidate_questions = parsed
    if isinstance(parsed, dict):
        candidate_questions = parsed.get("questions") or parsed.get("followUpQuestions") or parsed.get("items")

    if isinstance(candidate_questions, list):
        questions = [str(item).strip() for item in candidate_questions if str(item).strip()]
        if questions:
            return questions[:num_questions]

    questions: List[str] = []
    for raw_line in response.replace("```json", "").replace("```", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        question_match = QUESTION_LINE_PATTERN.match(line)
        if question_match:
            questions.append(question_match.group("question").strip())

    return questions[:num_questions]


def parse_research_results_response(response: str, num_learnings: int) -> Dict[str, Any]:
    parsed = _load_repaired_json(response)

    if isinstance(parsed, dict):
        learnings_payload = parsed.get("learnings", [])
        follow_up_payload = parsed.get("followUpQuestions") or parsed.get("questions") or []
        learnings: List[str] = []
        citations: Dict[str, str] = {}

        if isinstance(learnings_payload, list):
            for item in learnings_payload:
                if isinstance(item, dict):
                    learning = str(item.get("insight") or item.get("learning") or "").strip()
                    citation = str(item.get("sourceUrl") or item.get("citation") or "").strip()
                else:
                    learning = str(item).strip()
                    citation = ""

                if learning:
                    learnings.append(learning)
                    if citation:
                        citations[learning] = citation

        questions = [str(item).strip() for item in follow_up_payload if str(item).strip()]
        if learnings or questions:
            return {
                "learnings": learnings[:num_learnings],
                "followUpQuestions": questions[:num_learnings],
                "citations": citations,
            }

    learnings: List[str] = []
    questions: List[str] = []
    citations: Dict[str, str] = {}

    for raw_line in response.replace("```json", "").replace("```", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        learning_match = LEARNING_LINE_PATTERN.match(line)
        question_match = QUESTION_LINE_PATTERN.match(line)

        if learning_match:
            learning = learning_match.group("learning").strip()
            citation = (learning_match.group("citation") or "").strip()
            if not citation:
                url_match = URL_PATTERN.search(learning)
                if url_match:
                    citation = url_match.group(0)
                    learning = learning.replace(citation, "").strip(" -")
            if learning:
                learnings.append(learning)
                if citation:
                    citations[learning] = citation
        elif question_match:
            questions.append(question_match.group("question").strip())

    return {
        "learnings": learnings[:num_learnings],
        "followUpQuestions": questions[:num_learnings],
        "citations": citations,
    }

def count_words(text) -> int:
    """Count words in a text string. Handles both strings and lists."""
    if isinstance(text, list):
        text = " ".join(str(item) for item in text)
    return len(str(text).split())

def trim_context_to_word_limit(context_list: List[str], max_words: int = MAX_CONTEXT_WORDS) -> List[str]:
    """Trim context list to stay within word limit while preserving most recent/relevant items"""
    total_words = 0
    trimmed_context = []

    # Process in reverse to keep most recent items
    for item in reversed(context_list):
        text = " ".join(str(part) for part in item) if isinstance(item, list) else str(item)
        words = count_words(item)
        if total_words + words <= max_words:
            trimmed_context.insert(0, item)  # Insert at start to maintain original order
            total_words += words
        elif not trimmed_context:
            trimmed_context.insert(0, " ".join(text.split()[:max_words]))
            break
        else:
            break

    return trimmed_context


# Final synthesis context under an orchestration policy. ``ranked`` (default)
# reserves a share of the word cap for the distilled learnings and fills the
# rest with retained evidence ranked by relevance, round-robin across
# sub-questions, so the policy's keep mask decides what the writer sees.
# ``recency`` is the old behavior: learnings first, then the retained set in
# retrieval order, trimmed from the end by trim_context_to_word_limit, which
# drops the learnings and the earliest evidence once the cap binds.
ENV_FINAL_CONTEXT = "GR_FINAL_CONTEXT"
ENV_FINAL_LEARNINGS_SHARE = "GR_FINAL_LEARNINGS_SHARE"
ENV_FINAL_ITEM_MAX_WORDS = "GR_FINAL_ITEM_MAX_WORDS"
FINAL_CONTEXT_MODES = ("ranked", "recency")
DEFAULT_LEARNINGS_SHARE = 0.2
DEFAULT_ITEM_MAX_WORDS = 1500

# Token stopping rule (GR_STOP_TOTAL_TOKENS). Before each new round and each
# sub-query, research stops if that next step, at the largest cost seen so far,
# plus the tokens the final report is expected to need, would take the query's
# total (input + output of every LLM call: the TokenTracker total that the
# harness reports as the run's tokens) past the target. This is a stopping
# rule, not a hard cap: a step costlier than any seen before, or a report
# longer than estimated, can still end the run above the target; the harness
# records whether it did. Unset / 0 = no stopping rule (default).
ENV_STOP_TOTAL_TOKENS = "GR_STOP_TOTAL_TOKENS"
# Tokens reserved for the report: ``auto`` (default) estimates them from the
# writer context the retained evidence and learnings would build; an integer
# reserves that many tokens regardless of the evidence ("equal research
# budget" rather than "equal total budget").
ENV_REPORT_RESERVE_TOKENS = "GR_REPORT_RESERVE_TOKENS"
REPORT_RESERVE_AUTO = "auto"
# Report-cost model for ``auto``, fit on 103 GPT-6.1 Sol DeepResearchGym
# reports (17k-25k context words): prompt tokens were 1.34-1.43 x context
# words, completion tokens at most 6.9k. Rounded up so the estimate errs high.
REPORT_TOKENS_PER_WORD = 1.45
REPORT_PROMPT_OVERHEAD_TOKENS = 1500
REPORT_OUTPUT_TOKENS = 7000
# Cost guesses used until the first sub-query / round has been measured.
DEFAULT_SUBQUERY_COST_TOKENS = 20000
DEFAULT_ROUND_OVERHEAD_TOKENS = 4000
# Writer-context words a sub-query adds, until the first round is measured.
DEFAULT_SUBQUERY_CONTEXT_WORDS = 4000
# Words of "Source: <url>" / "Content:" framing per evidence item.
_ITEM_FRAME_WORDS = 3


def report_tokens_for_words(words: int) -> int:
    """Expected report cost (prompt + completion) for a writer context of ``words``."""
    return int(REPORT_TOKENS_PER_WORD * max(0, words)) + REPORT_PROMPT_OVERHEAD_TOKENS + REPORT_OUTPUT_TOKENS


def _tokens_used() -> int:
    snap = TokenTracker.snapshot()
    return int(snap.get("input_tokens", 0)) + int(snap.get("output_tokens", 0))


def _cosine(a: Optional[list[float]], b: Optional[list[float]]) -> float:
    if a is None or b is None:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb) / denom if denom > 0 else 0.0


def build_final_context(
    learnings: List[str],
    evidence: List[Any],
    *,
    query_embedding: Optional[list[float]] = None,
    subquestion_embeddings: Optional[List[list[float]]] = None,
    max_words: int = MAX_CONTEXT_WORDS,
    learnings_share: float = DEFAULT_LEARNINGS_SHARE,
    item_max_words: int = DEFAULT_ITEM_MAX_WORDS,
) -> tuple[List[str], Dict[str, Any]]:
    """Writer context from the learnings and the retained evidence K_T.

    ``evidence`` items need ``source_url``, ``content``, ``embedding``,
    ``retrieval_round`` and ``tree_depth`` (EvidenceItem). Learnings take up
    to ``learnings_share`` of ``max_words``; evidence gets the rest. Each item
    is assigned to its closest sub-question and the groups are drained
    round-robin, most query-relevant first, so no sub-question is crowded out.
    Items longer than ``item_max_words`` (0 = no cap) are cut to their head so
    a few long pages cannot fill the cap. Selected items are emitted in
    retrieval order. Returns the context list and a small stats dict.
    """
    learn_cap = int(max_words * max(0.0, min(1.0, learnings_share)))
    learn_out: List[str] = []
    learn_words = 0
    for line in learnings:
        w = count_words(line)
        if learn_words + w > learn_cap:
            continue
        learn_out.append(line)
        learn_words += w

    subqs = [v for v in (subquestion_embeddings or []) if v is not None]
    groups: Dict[int, List[tuple[float, int, Any]]] = defaultdict(list)
    for idx, e in enumerate(evidence):
        rel = _cosine(query_embedding, e.embedding)
        if subqs and e.embedding is not None:
            g = int(np.argmax([_cosine(s, e.embedding) for s in subqs]))
        else:
            g = 0
        groups[g].append((rel, idx, e))
    for g in groups:
        groups[g].sort(key=lambda t: (-t[0], t[1]))

    order: List[tuple[int, Any]] = []
    queues = [groups[g] for g in sorted(groups)]
    while any(queues):
        for q in queues:
            if q:
                _, idx, e = q.pop(0)
                order.append((idx, e))

    remaining = max_words - learn_words
    picked: List[tuple[int, str]] = []
    for idx, e in order:
        words = str(e.content).split()
        if item_max_words and len(words) > item_max_words:
            words = words[:item_max_words]
        text = DeepResearchSkill._format_item(e.source_url, " ".join(words))
        w = count_words(text)
        if w > remaining:
            continue
        picked.append((idx, text))
        remaining -= w

    picked.sort(key=lambda t: (evidence[t[0]].retrieval_round, evidence[t[0]].tree_depth, t[0]))
    context = learn_out + [text for _, text in picked]
    stats = {
        "mode": "ranked",
        "learnings_in": len(learnings),
        "learnings_used": len(learn_out),
        "evidence_in": len(evidence),
        "evidence_used": len(picked),
        "words": max_words - remaining,
    }
    return context, stats


class ResearchProgress:
    def __init__(self, total_depth: int, total_breadth: int):
        self.current_depth = 1  # Start from 1 and increment up to total_depth
        self.total_depth = total_depth
        self.current_breadth = 0  # Start from 0 and count up to total_breadth as queries complete
        self.total_breadth = total_breadth
        self.current_query: Optional[str] = None
        self.total_queries = 0
        self.completed_queries = 0


class DeepResearchSkill:
    def __init__(self, researcher):
        self.researcher = researcher
        self.breadth = getattr(researcher.cfg, 'deep_research_breadth', 4)
        self.depth = getattr(researcher.cfg, 'deep_research_depth', 2)
        self.concurrency_limit = getattr(researcher.cfg, 'deep_research_concurrency', 2)
        self.websocket = researcher.websocket
        self.tone = researcher.tone
        self.config_path = researcher.cfg.config_path if hasattr(researcher.cfg, 'config_path') else None
        self.headers = researcher.headers or {}
        self.visited_urls = researcher.visited_urls
        self.learnings = []
        self.research_sources = []  # Track all research sources
        self.context = []  # Track all context
        self.trajectory_logger = TrajectoryLogger(researcher.query)

        # Orchestration policy (GR_ORCHESTRATOR). ``None`` means legacy: the
        # checkpoint only records what the per-sub-query EmbeddingsFilter kept
        # and the recursion is fixed. Any other value makes the checkpoint act
        # on the policy's (u, m, w).
        self.policy_name = policy_name_from_env()
        self.orchestrator = build_policy(self.policy_name, llm_call=self._orchestrator_llm_call)
        raw_budget = os.environ.get(ENV_BUDGET_NAME, "").strip()
        self.token_budget: int | None = int(raw_budget) if raw_budget else None
        self.query_embedding: list[float] | None = None
        self.subq_embeddings: list[list[float]] = []
        # Set when a policy returns u=terminate; every pending recursion stops.
        self.stop_requested = False
        # Token stopping rule (GR_STOP_TOTAL_TOKENS); see _token_stop_blocks.
        raw_target = os.environ.get(ENV_STOP_TOTAL_TOKENS, "").strip()
        self.stop_total_tokens: int | None = int(raw_target) if raw_target and int(raw_target) > 0 else None
        raw_reserve = os.environ.get(ENV_REPORT_RESERVE_TOKENS, "").strip().lower() or REPORT_RESERVE_AUTO
        self.report_reserve_fixed: int | None = None if raw_reserve == REPORT_RESERVE_AUTO else int(raw_reserve)
        self._subquery_costs: list[int] = []
        self._round_overheads: list[int] = []
        # Writer-context growth per sub-query (words), measured at checkpoints.
        self._subquery_context_words: list[float] = []
        # Sub-queries finished since the last checkpoint (their evidence is not
        # in the retained set yet).
        self._pending_subqueries = 0
        self._learnings_seen: dict[str, None] = {}
        self.token_stop_info: Dict[str, Any] = {
            "stop_total_tokens": self.stop_total_tokens,
            "report_reserve": (
                None if self.stop_total_tokens is None
                else (REPORT_RESERVE_AUTO if self.report_reserve_fixed is None else self.report_reserve_fixed)
            ),
            "stopped": False,
            "stopped_at": None,
            "tokens_at_stop": None,
            "next_step_estimate": None,
            "report_reserve_at_stop": None,
            "writer_words_projected_at_stop": None,
            "skipped_subqueries": 0,
            "skipped_rounds": 0,
        }
        logger.info(
            f"DeepResearchSkill initialized: depth={self.depth}, breadth={self.breadth}, "
            f"orchestrator={self.policy_name}, context_budget={self.token_budget}, "
            f"stop_total_tokens={self.stop_total_tokens}, "
            f"report_reserve={self.token_stop_info['report_reserve']}"
        )

    def _writer_context_words(self) -> int:
        """Words the final writer context would hold if research stopped now.

        Mirrors build_final_context (ranked mode under a policy): learnings up
        to their share of the word cap, plus each retained item cut to
        GR_FINAL_ITEM_MAX_WORDS, all within MAX_CONTEXT_WORDS. Without a policy
        or in recency mode the writer gets the trimmed raw context, so the
        estimate is the full word cap."""
        final_mode = os.environ.get(ENV_FINAL_CONTEXT, "").strip().lower() or "ranked"
        if self.orchestrator is None or final_mode != "ranked":
            return MAX_CONTEXT_WORDS
        share_raw = os.environ.get(ENV_FINAL_LEARNINGS_SHARE, "").strip()
        share = float(share_raw) if share_raw else DEFAULT_LEARNINGS_SHARE
        cap_raw = os.environ.get(ENV_FINAL_ITEM_MAX_WORDS, "").strip()
        item_max = int(cap_raw) if cap_raw else DEFAULT_ITEM_MAX_WORDS
        learn_cap = int(MAX_CONTEXT_WORDS * max(0.0, min(1.0, share)))
        learn_words = min(learn_cap, sum(count_words(l) for l in self._learnings_seen))
        ev_words = 0
        for e in self.trajectory_logger.get_retained_evidence().values():
            w = len(str(e.content).split())
            ev_words += (min(w, item_max) if item_max else w) + _ITEM_FRAME_WORDS
        return min(MAX_CONTEXT_WORDS, learn_words + ev_words)

    def _report_reserve(self) -> tuple[int, int | None]:
        """Tokens to hold back for the report if one more sub-query runs, and
        the projected writer-context words behind them (None for a fixed reserve)."""
        if self.report_reserve_fixed is not None:
            return self.report_reserve_fixed, None
        growth = max(self._subquery_context_words, default=DEFAULT_SUBQUERY_CONTEXT_WORDS)
        projected = min(
            MAX_CONTEXT_WORDS,
            int(self._writer_context_words() + growth * (self._pending_subqueries + 1)),
        )
        return report_tokens_for_words(projected), projected

    def _token_stop_blocks(self, unit: str) -> bool:
        """True when starting one more ``unit`` (``round`` | ``subquery``) would,
        by the current estimates, take the query past GR_STOP_TOTAL_TOKENS once
        the report is written. The first time it blocks, research stops for
        every pending branch. A stopping rule, not a hard cap: the estimates are
        the largest costs seen so far, so an unusually costly step or report can
        still overshoot."""
        if self.stop_total_tokens is None:
            return False
        # Either unit is followed by at least one round checkpoint (and a round
        # must fit at least one sub-query), so both estimates are one sub-query
        # plus one round overhead, at the largest cost seen so far.
        est = (max(self._subquery_costs, default=DEFAULT_SUBQUERY_COST_TOKENS)
               + max(self._round_overheads, default=DEFAULT_ROUND_OVERHEAD_TOKENS))
        reserve, projected_words = self._report_reserve()
        used = _tokens_used()
        if used + est + reserve <= self.stop_total_tokens:
            return False
        info = self.token_stop_info
        info["skipped_rounds" if unit == "round" else "skipped_subqueries"] += 1
        if not info["stopped"]:
            info.update(
                stopped=True,
                stopped_at=unit,
                tokens_at_stop=used,
                next_step_estimate=est,
                report_reserve_at_stop=reserve,
                writer_words_projected_at_stop=projected_words,
            )
            self.stop_requested = True
            logger.info(
                f"Token stopping rule: stopping research before next {unit} "
                f"(used={used}, est_next={est}, report_reserve={reserve}, "
                f"target={self.stop_total_tokens})"
            )
            print(
                f"\nTOKEN STOP: {used} used + next {unit} ~{est} + report ~{reserve} "
                f"> {self.stop_total_tokens}; stopping research.",
                flush=True,
            )
        return True

    async def _orchestrator_llm_call(self, messages: List[Dict[str, str]]) -> str:
        """LLM call used by the prompted / learned policies; metered like every
        other call.

        Defaults to the strategic model. ``GR_ORCH_LLM_MODEL`` points the
        orchestrator at a different model (e.g. a fine-tuned checkpoint);
        ``GR_ORCH_LLM_BASE_URL`` / ``GR_ORCH_LLM_API_KEY`` route it to any
        OpenAI-compatible server (vLLM, a Colab tunnel, ...) without touching
        the rest of the run's LLM configuration.
        """
        model = os.environ.get("GR_ORCH_LLM_MODEL", "").strip() or self.researcher.cfg.strategic_llm_model
        provider = os.environ.get("GR_ORCH_LLM_PROVIDER", "").strip() or (
            "openai" if os.environ.get("GR_ORCH_LLM_BASE_URL", "").strip() else self.researcher.cfg.strategic_llm_provider
        )
        llm_kwargs: Dict[str, Any] = dict(self.researcher.cfg.llm_kwargs or {})
        base_url = os.environ.get("GR_ORCH_LLM_BASE_URL", "").strip()
        if base_url:
            # Set both names: langchain-openai accepts either; the OpenAI SDK
            # reads `base_url`. create_chat_completion must not overwrite these
            # with process-level OPENAI_BASE_URL (Azure).
            llm_kwargs["openai_api_base"] = base_url
            llm_kwargs["base_url"] = base_url
            llm_kwargs["openai_api_key"] = os.environ.get("GR_ORCH_LLM_API_KEY", "").strip() or "EMPTY"
            llm_kwargs["api_key"] = llm_kwargs["openai_api_key"]
        return await create_chat_completion(
            messages=messages,
            llm_provider=provider,
            model=model,
            temperature=0.0,
            max_tokens=2000,
            llm_kwargs=llm_kwargs,
            usage_tag="orchestrator",
        )

    @staticmethod
    def _format_item(source_url: str, text: str) -> str:
        """Context line for one retained page; keeps the URL so the report can cite it."""
        return f"Source: {source_url}\nContent: {text}\n"

    def _retained_context(self) -> List[str]:
        """The authoritative K_T: every retained item's (possibly compressed)
        text, in retrieval order. Used for the final synthesis context when an
        orchestration policy is active."""
        items = sorted(
            self.trajectory_logger.get_retained_evidence().values(),
            key=lambda e: (e.retrieval_round, e.tree_depth),
        )
        return [self._format_item(e.source_url, e.content) for e in items]

    async def _embed_query_and_subquestions(self, subquestions: List[str]) -> None:
        """Embed the root query and sub-questions once so policies can score
        evidence against them. Non-fatal: policies degrade to score 0."""
        try:
            emb_model = self.researcher.memory.get_embeddings()
            texts = [self.researcher.query] + list(subquestions)
            vecs = await asyncio.to_thread(emb_model.embed_documents, texts)
            self.query_embedding = list(vecs[0])
            self.subq_embeddings = [list(v) for v in vecs[1:]]
        except Exception as e:
            logger.warning(f"Query embedding for orchestrator failed (non-fatal): {e}")

    async def _embed_texts(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed short strings (frontier sub-queries) for per-branch coverage
        gaps. Non-fatal: returns ``None`` per text when the embedder is
        unavailable, and policies fall back to a gap of 1.0."""
        if not texts:
            return []
        try:
            emb_model = self.researcher.memory.get_embeddings()
            vecs = await asyncio.to_thread(emb_model.embed_documents, list(texts))
            return [list(v) for v in vecs]
        except Exception as e:
            logger.warning(f"Frontier embedding for orchestrator failed (non-fatal): {e}")
            return [None for _ in texts]

    async def generate_search_queries(self, query: str, num_queries: int = 3) -> List[Dict[str, str]]:
        """Generate SERP queries for research"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert researcher generating search queries. "
                    "Each query must be a short keyword phrase under 80 characters, suitable for a web search engine. "
                    "Do not copy the user's full prompt into the query. "
                    "Return valid JSON only. Do not include markdown, code fences, bullets, numbering, or prose."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Given the following prompt, generate {num_queries} unique search queries to research the topic thoroughly. "
                    "For each query, provide a research goal.\n\n"
                    "Return ONLY a JSON array of objects using this exact schema:\n"
                    '[{"query": "<search query>", "researchGoal": "<research goal>"}]\n\n'
                    f"Prompt: {query}"
                ),
            },
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            reasoning_effort=self.researcher.cfg.reasoning_effort,
            temperature=0.4,
            max_tokens=4000,
            llm_kwargs=self.researcher.cfg.llm_kwargs,
            usage_tag="deep_search_query_gen",
        )

        return parse_search_queries_response(response, num_queries)

    async def generate_research_plan(self, query: str, num_questions: int = 3) -> List[str]:
        """Generate follow-up questions to clarify research direction"""
        # Get initial search results from all retrievers to inform query generation
        all_search_results = []
        for retriever in self.researcher.retrievers:
            try:
                results = await get_search_results(
                    query,
                    retriever,
                    researcher=self.researcher
                )
                all_search_results.extend(results)
            except Exception as e:
                logger.warning(f"Error with retriever {retriever.__name__}: {e}")
        search_results = all_search_results
        logger.info(f"Initial web knowledge obtained: {len(search_results)} results")

        # Get current time for context
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert researcher. Your task is to analyze the original query and search results, "
                    "then generate targeted questions that explore different aspects and time periods of the topic. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user",
             "content": f"""Original query: {query}

Current time: {current_time}

Search results:
{search_results}

Based on these results, the original query, and the current time, generate {num_questions} unique questions. Each question should explore a different aspect or time period of the topic, considering recent developments up to {current_time}.

Return ONLY a JSON object using this exact schema:
{{"questions": ["<question 1>", "<question 2>"]}}"""}
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            reasoning_effort=ReasoningEfforts.High.value,
            temperature=0.4,
            llm_kwargs=self.researcher.cfg.llm_kwargs,
            usage_tag="deep_research_plan",
        )

        return parse_follow_up_questions_response(response, num_questions)

    async def process_research_results(self, query: str, context: str, num_learnings: int = 3) -> Dict[str, List[str]]:
        """Process research results to extract learnings and follow-up questions"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert researcher analyzing search results. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user",
             "content": (
                 f"Given the following research results for the query '{query}', extract key learnings and suggest "
                 "follow-up questions. For each learning, include a citation to the source URL if available.\n\n"
                 "Return ONLY a JSON object using this exact schema:\n"
                 '{"learnings": [{"insight": "<insight>", "sourceUrl": "<url or empty string>"}], '
                 '"followUpQuestions": ["<question 1>", "<question 2>"]}\n\n'
                 f"Research results:\n{context}"
             )}
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            temperature=0.4,
            reasoning_effort=ReasoningEfforts.Low.value,
            # Needs headroom for reasoning tokens on reasoning models
            max_tokens=16000,
            llm_kwargs=self.researcher.cfg.llm_kwargs,
            usage_tag="deep_process_results",
        )

        return parse_research_results_response(response, num_learnings)

    async def deep_research(
            self,
            query: str,
            breadth: int,
            depth: int,
            learnings: List[str] = None,
            citations: Dict[str, str] = None,
            visited_urls: Set[str] = None,
            on_progress=None
    ) -> Dict[str, Any]:
        """Conduct deep iterative research"""
        print(f"\n📊 DEEP RESEARCH: depth={depth}, breadth={breadth}, query={query[:100]}...", flush=True)
        if learnings is None:
            learnings = []
        if citations is None:
            citations = {}
        if visited_urls is None:
            visited_urls = set()

        progress = ResearchProgress(depth, breadth)

        if on_progress:
            on_progress(progress)

        # Snapshot counters before any work so the round cost captures the full delta
        tokens_before = TokenTracker.snapshot()
        latency_counts_before = LatencyTracker.snapshot()
        round_start = time.time()

        all_learnings = learnings.copy()
        all_visited_urls = set(visited_urls) if visited_urls else set()
        all_citations = dict(citations) if citations else {}
        all_context = []
        all_sources = []

        if self.stop_requested or self._token_stop_blocks("round"):
            return {
                'learnings': all_learnings,
                'visited_urls': all_visited_urls,
                'citations': all_citations,
                'context': all_context,
                'sources': all_sources,
            }

        print(f"🔎 Generating {breadth} search queries...", flush=True)
        # Round overhead = everything in the round outside its sub-queries
        # (search-query generation, the policy's checkpoint call); measured
        # just before descending.
        overhead_start = _tokens_used()
        n_costs_before = len(self._subquery_costs)
        writer_words_before = self._writer_context_words() if self.stop_total_tokens else 0
        serp_queries = await self.generate_search_queries(query, num_queries=breadth)
        print(f"✅ Generated {len(serp_queries)} queries: {[q['query'] for q in serp_queries]}", flush=True)
        progress.total_queries = len(serp_queries)
        if not serp_queries:
            logger.warning("Deep research generated zero search queries; stopping descent.")
            return {
                'learnings': all_learnings,
                'visited_urls': all_visited_urls,
                'citations': all_citations,
                'context': all_context,
                'sources': all_sources,
            }

        # Process queries with concurrency limit
        semaphore = asyncio.Semaphore(self.concurrency_limit)

        async def process_query(serp_query: Dict[str, str]) -> Optional[Dict[str, Any]]:
            async with semaphore:
                if self._token_stop_blocks("subquery"):
                    return None
                subquery_start = _tokens_used()
                try:
                    progress.current_query = serp_query['query']
                    if on_progress:
                        on_progress(progress)

                    from .. import GPTResearcher
                    researcher = GPTResearcher(
                        query=serp_query['query'],
                        report_type=ReportType.ResearchReport.value,
                        report_source=ReportSource.Web.value,
                        tone=self.tone,
                        websocket=self.websocket,
                        config_path=self.config_path,
                        headers=self.headers,
                        visited_urls=self.visited_urls,
                        # Propagate MCP configuration to nested researchers
                        mcp_configs=self.researcher.mcp_configs,
                        mcp_strategy=self.researcher.mcp_strategy
                    )

                    # Conduct research
                    context = await researcher.conduct_research()

                    # Get results and visited URLs
                    visited = researcher.visited_urls
                    sources = researcher.research_sources

                    # Process results to extract learnings and citations
                    results = await self.process_research_results(
                        query=serp_query['query'],
                        context=context
                    )

                    # Exact with concurrency 1; an over-estimate otherwise
                    # (overlapping sub-queries are counted together).
                    self._subquery_costs.append(_tokens_used() - subquery_start)
                    self._pending_subqueries += 1
                    for learning in results['learnings']:
                        self._learnings_seen.setdefault(learning, None)

                    # Update progress
                    progress.completed_queries += 1
                    progress.current_breadth += 1
                    if on_progress:
                        on_progress(progress)

                    return {
                        'learnings': results['learnings'],
                        'visited_urls': list(visited),
                        'followUpQuestions': results['followUpQuestions'],
                        'researchGoal': serp_query['researchGoal'],
                        'citations': results['citations'],
                        'context': "\n".join(context) if isinstance(context, list) else (context or ""),
                        'sources': sources if sources else []
                    }

                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.error(f"Error processing query '{serp_query['query']}': {str(e)}")
                    print(f"\n❌ DEEP RESEARCH ERROR: {str(e)}\n{error_details}", flush=True)
                    return None

        # Process queries concurrently with limit
        tasks = [process_query(query) for query in serp_queries]
        results = await asyncio.gather(*tasks)
        results = [r for r in results if r is not None]

        # Update breadth progress based on successful queries
        progress.current_breadth = len(results)
        if on_progress:
            on_progress(progress)

        # #1579: if every branch at this level failed (bad API key, offline
        # retriever, etc.), stop instead of endlessly generating follow-ups
        # from empty goals / empty learnings.
        if not results and self.token_stop_info["stopped"]:
            return {
                'learnings': all_learnings,
                'visited_urls': all_visited_urls,
                'citations': all_citations,
                'context': all_context,
                'sources': all_sources,
            }
        if not results:
            logger.warning(
                "Deep research produced no successful query results at depth=%s; stopping descent.",
                depth,
            )
            print(
                f"\nDEEP RESEARCH: no successful results at depth={depth}; stopping to avoid infinite work.",
                flush=True,
            )
            return {
                'learnings': all_learnings,
                'visited_urls': all_visited_urls,
                'citations': all_citations,
                'context': all_context,
                'sources': all_sources,
            }

        # -- Orchestration checkpoint: aggregate this layer's evidence --
        round_id = self.trajectory_logger.begin_round()

        current_tree_depth = self.depth - depth + 1
        assert current_tree_depth >= 1, f"Invalid tree_depth={current_tree_depth} (self.depth={self.depth}, depth={depth})"

        frontier_nodes: list[FrontierNode] = []

        # Collect all results
        for result in results:
            all_learnings.extend(result['learnings'])
            all_visited_urls.update(result['visited_urls'])
            all_citations.update(result['citations'])
            if result['context']:
                # Use extend, not append: when CURATE_SOURCES=True, result['context'] is
                # a List[dict]. append() nests it as a single item, which causes
                # "\n".join() to crash later with "expected str instance, dict found".
                ctx = result['context']
                if isinstance(ctx, list):
                    all_context.extend(ctx)
                else:
                    all_context.append(ctx)
            if result['sources']:
                all_sources.extend(result['sources'])

            node_id = make_item_id(result['researchGoal'], '')
            frontier_nodes.append(FrontierNode(
                node_id=node_id,
                subquery=result['researchGoal'],
                parent_subquery=query[:200],
                status="open",
            ))

        # Page-level evidence from EmbeddingsFilter keep/prune decisions.
        # Each sub-researcher's conduct_research -> compression pipeline populates
        # _global_embedding_cache.record(); drain() collects all chunks from this
        # round's parallel queries. Chunks are aggregated by source URL into one
        # evidence item per page: content is the deduplicated chunk text joined,
        # embedding is the mean over all chunks, and the page is kept if any
        # sub-query's filter kept any of its chunks.
        chunk_records = _global_embedding_cache.drain()
        pages: dict[str, dict] = defaultdict(
            lambda: {"chunks": [], "embs": [], "chunk_kept": [], "kept": False, "subquery": None}
        )
        for rec in chunk_records:
            for ch in rec["chunks"]:
                p = pages[ch["source_url"]]
                if p["subquery"] is None:
                    p["subquery"] = rec["query"]
                p["chunks"].append(ch["content"])
                p["embs"].append(ch["embedding"])
                p["chunk_kept"].append(bool(ch["kept"]))
                p["kept"] = p["kept"] or ch["kept"]

        # One evidence item per page. Chunks are deduplicated with their
        # embeddings and filter verdicts kept aligned so policies can score
        # (or reproduce the filter) at chunk granularity. A chunk seen under
        # several sub-queries is kept if any of them kept it, matching the
        # page-level ``kept`` rule above.
        page_items: list[dict] = []
        for url, p in pages.items():
            seen: dict[str, int] = {}
            unique_chunks: list[str] = []
            unique_embs: list[list[float]] = []
            unique_kept: list[bool] = []
            for chunk, emb, kept_flag in zip(p["chunks"], p["embs"], p["chunk_kept"]):
                idx = seen.get(chunk)
                if idx is not None:
                    unique_kept[idx] = unique_kept[idx] or kept_flag
                    continue
                seen[chunk] = len(unique_chunks)
                unique_chunks.append(chunk)
                unique_embs.append(emb)
                unique_kept.append(kept_flag)
            content = "\n\n".join(unique_chunks)
            item_id = self.trajectory_logger.add_evidence(
                content=content,
                source_url=url,
                source_subquery=p["subquery"],
                tree_depth=current_tree_depth,
                source_type="page",
                embedding=_unit_mean(unique_embs) if unique_embs else None,
            )
            page_items.append({
                "item_id": item_id,
                "url": url,
                "content": content,
                "chunks": unique_chunks,
                "embs": unique_embs,
                "chunk_kept": unique_kept,
                "subquery": p["subquery"],
                "filter_kept": p["kept"],
            })

        kept_ids: list[str] = []
        pruned_ids: list[str] = []
        decision_type = "continue"
        branch_allocation: dict[str, float] | None = None
        decision_meta: dict[str, Any] = {}
        result_node_ids = [make_item_id(r['researchGoal'], '') for r in results]
        default_child_breadth = max(2, breadth // 2)
        child_breadth: dict[str, int] = {nid: default_child_breadth for nid in result_node_ids}

        if self.orchestrator is None:
            # Legacy: record the EmbeddingsFilter's verdict, change nothing.
            for pi in page_items:
                (kept_ids if pi["filter_kept"] else pruned_ids).append(pi["item_id"])
        else:
            new_ids = {pi["item_id"] for pi in page_items}
            new_items = [
                PoolItem(
                    item_id=pi["item_id"],
                    source_url=pi["url"],
                    text=pi["content"],
                    embedding=_unit_mean(pi["embs"]) if pi["embs"] else None,
                    tree_depth=current_tree_depth,
                    retrieval_round=round_id,
                    is_new=True,
                    source_subquery=pi["subquery"] or "",
                    chunks=pi["chunks"],
                    chunk_embeddings=pi["embs"],
                    filter_kept=bool(pi["filter_kept"]),
                    chunk_kept=pi["chunk_kept"],
                )
                for pi in page_items
            ]
            retained_prev = [
                PoolItem(
                    item_id=e.item_id,
                    source_url=e.source_url,
                    text=e.content,
                    embedding=e.embedding,
                    tree_depth=e.tree_depth,
                    retrieval_round=e.retrieval_round,
                    is_new=False,
                    source_subquery=e.source_subquery,
                )
                for e in self.trajectory_logger.get_retained_evidence().values()
                if e.item_id not in new_ids
            ]
            tokens_now = TokenTracker.snapshot()
            # Branch sub-query vectors let policies score the coverage gap of
            # each open branch (allocation prior). Only policies that opt in
            # pay for it, so the Table-4 baselines' cost profile is unchanged.
            frontier_embs: list[Optional[list[float]]] = [None] * len(frontier_nodes)
            if getattr(self.orchestrator, "needs_frontier_embeddings", False) and frontier_nodes:
                frontier_embs = await self._embed_texts([n.subquery for n in frontier_nodes])
            inp = OrchestrationInput(
                root_query=self.researcher.query,
                query_embedding=self.query_embedding,
                subquestions=list(self.trajectory_logger.trajectory.subquestions),
                new_items=new_items,
                retained_prev=retained_prev,
                frontier=[
                    FrontierInfo(n.node_id, n.subquery, emb)
                    for n, emb in zip(frontier_nodes, frontier_embs)
                ],
                round_id=round_id,
                tree_depth=current_tree_depth,
                tokens_used=int(tokens_now.get("input_tokens", 0)) + int(tokens_now.get("output_tokens", 0)),
                token_budget=self.token_budget,
                subquestion_embeddings=[list(v) for v in self.subq_embeddings],
            )
            decision = await self.orchestrator.decide(inp)

            # m: apply retention and any compression to the logger's evidence.
            for iid, text in decision.rewritten.items():
                self.trajectory_logger.set_content(iid, text)
            for it in inp.pool:
                (kept_ids if it.item_id in decision.kept_ids else pruned_ids).append(it.item_id)

            # Forward context for this level is rebuilt from the decision, not
            # from what the sub-researchers' filters happened to return.
            all_context = [
                self._format_item(it.source_url, decision.text_for(it))
                for it in new_items
                if it.item_id in decision.kept_ids
            ]

            # u: stop every pending recursion once any round says terminate.
            if decision.terminate:
                self.stop_requested = True
            decision_type = "terminate" if decision.terminate else "continue"

            # w: split the same total child breadth the fork would have spent.
            branch_allocation = dict(decision.branch_allocation)
            child_breadth = allocate_breadth(
                branch_allocation, result_node_ids, default_child_breadth * len(result_node_ids)
            )

            decision_meta = dict(decision.meta)
            decision_meta["context_tokens_before"] = sum(it.tokens for it in inp.pool)
            decision_meta["context_tokens_after"] = sum(
                estimate_tokens(decision.text_for(it)) for it in inp.pool if it.item_id in decision.kept_ids
            )
            decision_meta["child_breadth"] = child_breadth

        tokens_after = TokenTracker.snapshot()
        latency_counts_after = LatencyTracker.snapshot()
        round_search_calls = (
            latency_counts_after.get("search", 0)
            - latency_counts_before.get("search", 0)
        )
        round_llm_calls = (
            latency_counts_after.get("llm", 0)
            - latency_counts_before.get("llm", 0)
        )
        round_cost = RoundCost(
            tokens_input=tokens_after["input_tokens"] - tokens_before["input_tokens"],
            tokens_output=tokens_after["output_tokens"] - tokens_before["output_tokens"],
            latency_seconds=time.time() - round_start,
            llm_calls=round_llm_calls,
            search_calls=round_search_calls,
        )

        self.trajectory_logger.record_round(
            kept_item_ids=kept_ids,
            pruned_item_ids=pruned_ids,
            frontier=frontier_nodes,
            round_cost=round_cost,
            decision_type=decision_type,
            branch_allocation=branch_allocation,
            policy=self.policy_name,
            meta=decision_meta,
        )

        logger.info(
            f"Round {round_id} checkpoint [{self.policy_name}]: depth={depth}, "
            f"kept={len(kept_ids)}, pruned={len(pruned_ids)}, "
            f"frontier_nodes={len(frontier_nodes)}, decision={decision_type}, "
            f"child_breadth={child_breadth}"
        )

        self._round_overheads.append(max(
            0,
            _tokens_used() - overhead_start - sum(self._subquery_costs[n_costs_before:]),
        ))
        # Writer-context growth per sub-query this round, after the policy's
        # keep/compress decision (so pruning policies grow slowly).
        if self.stop_total_tokens and results:
            grown = self._writer_context_words() - writer_words_before
            self._subquery_context_words.append(max(0.0, grown / len(results)))
        self._pending_subqueries = 0

        for result, node_id in zip(results, result_node_ids):
            # Continue deeper if needed, unless a policy asked to stop.
            if depth > 1 and not self.stop_requested:
                new_breadth = child_breadth.get(node_id, default_child_breadth)
                new_depth = depth - 1
                progress.current_depth += 1

                # Create next query from research goal and follow-up questions
                next_query = f"""
                Previous research goal: {result['researchGoal']}
                Follow-up questions: {' '.join(result['followUpQuestions'])}
                """

                # Recursive research
                deeper_results = await self.deep_research(
                    query=next_query,
                    breadth=new_breadth,
                    depth=new_depth,
                    learnings=all_learnings,
                    citations=all_citations,
                    visited_urls=all_visited_urls,
                    on_progress=on_progress
                )

                all_learnings = deeper_results['learnings']
                all_visited_urls.update(deeper_results['visited_urls'])
                all_citations.update(deeper_results['citations'])
                if deeper_results.get('context'):
                    all_context.extend(deeper_results['context'])
                if deeper_results.get('sources'):
                    all_sources.extend(deeper_results['sources'])

        # Update class tracking
        self.context.extend(all_context)
        self.research_sources.extend(all_sources)

        # Trim context to stay within word limits
        trimmed_context = trim_context_to_word_limit(all_context)
        logger.info(f"Trimmed context from {len(all_context)} items to {len(trimmed_context)} items to stay within word limit")

        return {
            'learnings': list(set(all_learnings)),
            'visited_urls': list(all_visited_urls),
            'citations': all_citations,
            'context': trimmed_context,
            'sources': all_sources
        }

    async def run(self, on_progress=None) -> str:
        """Run the deep research process and generate final report"""
        print(f"\n🔍 DEEP RESEARCH: Starting with breadth={self.breadth}, depth={self.depth}, concurrency={self.concurrency_limit}", flush=True)
        start_time = time.time()

        TokenTracker.reset()
        LLMCallLog.reset()
        LatencyTracker.reset()

        # Log initial costs
        initial_costs = self.researcher.get_costs()

        follow_up_questions = await self.generate_research_plan(self.researcher.query)
        self.trajectory_logger.set_subquestions(follow_up_questions)
        if self.orchestrator is not None and self.orchestrator.needs_embeddings:
            await self._embed_query_and_subquestions(follow_up_questions)
        answers = ["Automatically proceeding with research"] * len(follow_up_questions)

        qa_pairs = [f"Q: {q}\nA: {a}" for q, a in zip(follow_up_questions, answers)]
        combined_query = f"""
        Initial Query: {self.researcher.query}\nFollow - up Questions and Answers:\n
        """ + "\n".join(qa_pairs)

        results = await self.deep_research(
            query=combined_query,
            breadth=self.breadth,
            depth=self.depth,
            on_progress=on_progress
        )

        # Get costs after deep research
        research_costs = self.researcher.get_costs() - initial_costs

        # Log research costs if we have a log handler
        if self.researcher.log_handler:
            await self.researcher._log_event("research", step="deep_research_costs", details={
                "research_costs": research_costs,
                "total_costs": self.researcher.get_costs()
            })

        # Prepare context with citations
        context_with_citations = []
        for learning in results['learnings']:
            citation = results['citations'].get(learning, '')
            if citation:
                context_with_citations.append(f"{learning} [Source: {citation}]")
            else:
                context_with_citations.append(learning)

        # With a policy active the final evidence context is the retained set
        # K_T as decided round by round (including supersession of earlier
        # items and any compression), not the per-level lists.
        final_mode = os.environ.get(ENV_FINAL_CONTEXT, "").strip().lower() or "ranked"
        if final_mode not in FINAL_CONTEXT_MODES:
            raise ValueError(f"{ENV_FINAL_CONTEXT}={final_mode!r} is not one of {FINAL_CONTEXT_MODES}")
        if self.orchestrator is not None and final_mode == "ranked":
            share_raw = os.environ.get(ENV_FINAL_LEARNINGS_SHARE, "").strip()
            cap_raw = os.environ.get(ENV_FINAL_ITEM_MAX_WORDS, "").strip()
            final_context, final_stats = build_final_context(
                context_with_citations,
                list(self.trajectory_logger.get_retained_evidence().values()),
                query_embedding=self.query_embedding,
                subquestion_embeddings=self.subq_embeddings,
                learnings_share=float(share_raw) if share_raw else DEFAULT_LEARNINGS_SHARE,
                item_max_words=int(cap_raw) if cap_raw else DEFAULT_ITEM_MAX_WORDS,
            )
            logger.info(f"Final context: {final_stats}")
        else:
            if self.orchestrator is not None:
                results['context'] = self._retained_context()
            if results.get('context'):
                context_with_citations.extend(results['context'])
            final_context = trim_context_to_word_limit(context_with_citations)

        # Set enhanced context and visited URLs
        self.researcher.context = "\n".join(
            item if isinstance(item, str)
            else item.get("Content", str(item)) if isinstance(item, dict)
            else str(item)
            for item in final_context
        )
        self.researcher.visited_urls = results['visited_urls']

        # Set research sources
        if results.get('sources'):
            self.researcher.research_sources = results['sources']

        # Auxiliary vectors for feature computation (rho_i, kappa_i, Phi):
        # query, subquestions, and every frontier node subquery seen.
        try:
            emb_model = self.researcher.memory.get_embeddings()
            node_ids: list[str] = []
            node_texts: list[str] = []
            for snap in self.trajectory_logger.trajectory.rounds:
                for fn in snap.frontier:
                    if fn.node_id not in node_ids:
                        node_ids.append(fn.node_id)
                        node_texts.append(fn.subquery)
            subqs = list(self.trajectory_logger.trajectory.subquestions)
            texts = [self.researcher.query] + subqs + node_texts
            vecs = await asyncio.to_thread(emb_model.embed_documents, texts)
            self.trajectory_logger.set_aux_vectors("query", [vecs[0]])
            self.trajectory_logger.set_aux_vectors("subquestions", vecs[1:1 + len(subqs)])
            self.trajectory_logger.set_aux_vectors("nodes", vecs[1 + len(subqs):], ids=node_ids)
        except Exception as e:
            logger.warning(f"Aux vector embedding failed (non-fatal): {e}")

        self.token_stop_info["research_tokens"] = _tokens_used()
        token_totals = TokenTracker.get_totals()
        token_totals["tool_calls"] = sum(
            d.get("count", 0) for d in LatencyTracker.per_type_latencies.values()
        )
        final_ctx = self.researcher.context if isinstance(self.researcher.context, str) else str(self.researcher.context)
        self.trajectory_logger.finalize(final_ctx, token_totals)
        trajectory_path = self.trajectory_logger.save()
        _global_search_cache.save()
        _global_scraper_cache.save()

        # Log total execution time
        end_time = time.time()
        execution_time = timedelta(seconds=end_time - start_time)
        logger.info(f"Total research execution time: {execution_time}")
        logger.info(f"Total research costs: ${research_costs:.2f}")
        logger.info(f"Trajectory saved to: {trajectory_path}")

        # Return the context - don't generate report here as it will be done by the main agent
        return self.researcher.context

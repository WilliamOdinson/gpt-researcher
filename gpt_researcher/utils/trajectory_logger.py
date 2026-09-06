import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    item_id: str
    content: str
    word_count: int
    source_url: str
    source_subquery: str
    tree_depth: int
    retrieval_round: int
    embedding: list[float] | None = None
    source_type: str = "web"
    was_retained: bool = True


@dataclass
class RoundDecision:
    # continue / prune / expand / terminate / reallocate
    type: str
    # kept_item_ids / pruned_item_ids cover only items that received a decision
    # in this round (i.e. items produced by this round's sub-queries). They are
    # not a full-pool mask; see RoundSnapshot.retained_ids for the full K_t.
    kept_item_ids: list[str] = field(default_factory=list)
    pruned_item_ids: list[str] = field(default_factory=list)
    # Maps subquery node_id to allocated breadth fraction.
    # Currently empty because deep research uses fixed breadth // 2;
    # populated when a learned orchestrator provides reallocation decisions.
    branch_allocation: dict[str, float] = field(default_factory=dict)


@dataclass
class FrontierNode:
    node_id: str
    subquery: str
    parent_subquery: str
    status: str  # open / completed / pruned


@dataclass
class RoundCost:
    tokens_input: int = 0
    tokens_output: int = 0
    latency_seconds: float = 0.0
    llm_calls: int = 0
    search_calls: int = 0


@dataclass
class RoundSnapshot:
    round_id: int
    timestamp: float
    # Items first retrieved in this round. Full pool C_t = all items in
    # Trajectory.evidence with retrieval_round <= round_id.
    new_item_ids: list[str] = field(default_factory=list)
    # Full retained set K_t after this round's decision.
    retained_ids: list[str] = field(default_factory=list)
    decision: RoundDecision | None = None
    frontier: list[FrontierNode] = field(default_factory=list)
    round_cost: RoundCost = field(default_factory=RoundCost)


@dataclass
class Trajectory:
    query: str
    query_id: str
    subquestions: list[str] = field(default_factory=list)
    num_rounds: int = 0
    rounds: list[RoundSnapshot] = field(default_factory=list)
    # Single copy of every evidence item. Embeddings are stripped on save
    # and written to trajectory_{query_id}_emb.npz instead.
    evidence: dict[str, EvidenceItem] = field(default_factory=dict)
    final_context: str = ""
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    total_tool_calls: int = 0
    peak_context_length: int = 0
    evidence_total: int = 0
    evidence_retained_final: int = 0
    fraction_pruned: float = 0.0
    orchestration_steps: int = 0
    evidence_items_per_trajectory: int = 0


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def make_item_id(content: str, source_url: str) -> str:
    raw = f"{source_url}:{_content_hash(content)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


class TrajectoryLogger:
    def __init__(self, query: str, output_dir: str | None = None):
        self.trajectory = Trajectory(
            query=query,
            query_id=uuid.uuid4().hex[:12],
        )
        self._start_time = time.time()
        self._round_counter = 0
        self._current_round_start: float | None = None
        self._global_evidence: dict[str, EvidenceItem] = {}
        self._output_dir = output_dir or os.environ.get(
            "TRAJECTORY_OUTPUT_DIR",
            os.path.join(os.getcwd(), "trajectory_logs"),
        )

    def set_subquestions(self, subquestions: list[str]):
        self.trajectory.subquestions = subquestions

    def begin_round(self) -> int:
        self._round_counter += 1
        self._current_round_start = time.time()
        return self._round_counter

    def add_evidence(
        self,
        content: str,
        source_url: str,
        source_subquery: str,
        tree_depth: int,
        source_type: str = "web",
        embedding: list[float] | None = None,
    ) -> str:
        item_id = make_item_id(content, source_url)
        # First insertion wins: tree_depth and retrieval_round must reflect
        # when the item was first retrieved, not the last time it was seen.
        if item_id in self._global_evidence:
            return item_id
        self._global_evidence[item_id] = EvidenceItem(
            item_id=item_id,
            content=content,
            word_count=len(content.split()),
            source_url=source_url,
            source_subquery=source_subquery,
            tree_depth=tree_depth,
            retrieval_round=self._round_counter,
            embedding=embedding,
            source_type=source_type,
        )
        return item_id

    def record_round(
        self,
        kept_item_ids: list[str],
        pruned_item_ids: list[str],
        frontier: list[FrontierNode],
        round_cost: RoundCost,
        decision_type: str = "continue",
        branch_allocation: dict[str, float] | None = None,
    ):
        kept = set(kept_item_ids)
        pruned = set(pruned_item_ids) - kept

        for iid in kept:
            if iid in self._global_evidence:
                self._global_evidence[iid].was_retained = True
        for iid in pruned:
            if iid in self._global_evidence:
                self._global_evidence[iid].was_retained = False

        new_ids = [
            iid for iid, e in self._global_evidence.items()
            if e.retrieval_round == self._round_counter
        ]
        retained_ids = [
            iid for iid, e in self._global_evidence.items()
            if e.was_retained
        ]

        self.trajectory.rounds.append(RoundSnapshot(
            round_id=self._round_counter,
            timestamp=time.time(),
            new_item_ids=new_ids,
            retained_ids=retained_ids,
            decision=RoundDecision(
                type=decision_type,
                kept_item_ids=sorted(kept),
                pruned_item_ids=sorted(pruned),
                branch_allocation=branch_allocation or {},
            ),
            frontier=list(frontier),
            round_cost=round_cost,
        ))

    def get_retained_evidence(self) -> dict[str, EvidenceItem]:
        return {k: v for k, v in self._global_evidence.items() if v.was_retained}

    def get_all_evidence(self) -> dict[str, EvidenceItem]:
        return dict(self._global_evidence)

    def finalize(self, final_context: str, token_totals: dict[str, Any] | None = None):
        if self.trajectory.rounds:
            self.trajectory.rounds[-1].decision.type = "terminate"
            for fn in self.trajectory.rounds[-1].frontier:
                fn.status = "completed"

        self.trajectory.evidence = dict(self._global_evidence)
        self.trajectory.final_context = final_context
        self.trajectory.num_rounds = self._round_counter
        self.trajectory.total_latency = time.time() - self._start_time
        self.trajectory.orchestration_steps = self._round_counter

        all_items = list(self._global_evidence.values())
        retained = [i for i in all_items if i.was_retained]
        self.trajectory.evidence_total = len(all_items)
        self.trajectory.evidence_retained_final = len(retained)
        self.trajectory.evidence_items_per_trajectory = len(all_items)
        if all_items:
            self.trajectory.fraction_pruned = 1.0 - len(retained) / len(all_items)

        if token_totals:
            self.trajectory.total_tokens = (
                token_totals.get("input_tokens", 0) + token_totals.get("output_tokens", 0)
            )
            self.trajectory.total_cost = float(token_totals.get("cost", 0.0))
            self.trajectory.total_tool_calls = token_totals.get("tool_calls", 0)
            self.trajectory.peak_context_length = token_totals.get("peak_input_tokens", 0)

    def save(self) -> str:
        import numpy as np

        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        qid = self.trajectory.query_id

        ids: list[str] = []
        vecs: list[list[float]] = []
        for iid, e in self.trajectory.evidence.items():
            if e.embedding is not None:
                ids.append(iid)
                vecs.append(e.embedding)
        if ids:
            emb_path = out_dir / f"trajectory_{qid}_emb.npz"
            np.savez_compressed(
                emb_path,
                ids=np.array(ids),
                vectors=np.array(vecs, dtype=np.float32),
            )
            logger.info(f"Embeddings saved to {emb_path} ({len(ids)} vectors)")

        data = asdict(self.trajectory)
        for e in data["evidence"].values():
            e["embedding"] = None

        path = out_dir / f"trajectory_{qid}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Trajectory saved to {path}")
        return str(path)

import copy
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
    evidence_pool: list[EvidenceItem] = field(default_factory=list)
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
    final_context: str = ""
    total_tokens: int = 0
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
        word_count = len(content.split())
        item = EvidenceItem(
            item_id=item_id,
            content=content,
            word_count=word_count,
            source_url=source_url,
            source_subquery=source_subquery,
            tree_depth=tree_depth,
            retrieval_round=self._round_counter,
            embedding=embedding,
            source_type=source_type,
        )
        self._global_evidence[item_id] = item
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
        for iid in kept_item_ids:
            if iid in self._global_evidence:
                self._global_evidence[iid].was_retained = True
        for iid in pruned_item_ids:
            if iid in self._global_evidence:
                self._global_evidence[iid].was_retained = False

        snapshot = RoundSnapshot(
            round_id=self._round_counter,
            timestamp=time.time(),
            evidence_pool=copy.deepcopy(list(self._global_evidence.values())),
            decision=RoundDecision(
                type=decision_type,
                kept_item_ids=list(kept_item_ids),
                pruned_item_ids=list(pruned_item_ids),
                branch_allocation=branch_allocation or {},
            ),
            frontier=list(frontier),
            round_cost=round_cost,
        )
        self.trajectory.rounds.append(snapshot)

        if decision_type == "terminate":
            for iid in pruned_item_ids:
                self._global_evidence.pop(iid, None)

    def get_retained_evidence(self) -> dict[str, EvidenceItem]:
        return {k: v for k, v in self._global_evidence.items() if v.was_retained}

    def get_all_evidence(self) -> dict[str, EvidenceItem]:
        return dict(self._global_evidence)

    def finalize(self, final_context: str, token_totals: dict[str, Any] | None = None):
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
            self.trajectory.total_tokens = token_totals.get("input_tokens", 0) + token_totals.get("output_tokens", 0)
            self.trajectory.total_tool_calls = token_totals.get("tool_calls", 0)

        self.trajectory.peak_context_length = max(
            (sum(e.word_count for e in snap.evidence_pool) for snap in self.trajectory.rounds),
            default=0,
        )

    def save(self) -> str:
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"trajectory_{self.trajectory.query_id}.json"
        path = out_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.trajectory), f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Trajectory saved to {path}")
        return str(path)

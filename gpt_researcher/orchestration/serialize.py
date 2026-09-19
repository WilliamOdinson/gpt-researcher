"""Text serialization of the orchestration state and action (PILOT §4.3, App. A).

The learned orchestrator is a small LM that reads a serialized state and
emits a structured decision string. This module is the single definition of
both formats so that:

  * the online policy (fork) and the offline BC-data builder (harness) produce
    byte-identical prompts from the same round, and
  * a trained model's output can be parsed back into ``(u, m, w)`` with
    ``parse_action`` — the same parser ``PromptedPolicy`` uses.

State: one line per evidence item with its id, the eight features from
``features.FEATURE_NAMES`` rounded to two digits, status, and a short snippet;
then the open frontier with each branch's coverage gap; then the budget line.

Action (three lines, order fixed)::

    KEEP: <id> <id> ...        (or ALL)
    ALLOC: <node_id>=<w> ...   (weights normalised to sum to 1, 2 digits)
    DECISION: CONTINUE | TERMINATE
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_SNIPPET_CHARS = 160  # ~40 tokens

SYSTEM_PROMPT = (
    "You are the orchestrator of a deep research agent. After each research round "
    "you decide (1) which evidence items to keep in the working context, (2) how to "
    "split the next round's search effort across the open research branches, and "
    "(3) whether to stop researching and write the report. Answer ONLY in the exact "
    "three-line format: KEEP / ALLOC / DECISION."
)

_WS = re.compile(r"\s+")


def snippet(text: str, n: int = DEFAULT_SNIPPET_CHARS) -> str:
    return _WS.sub(" ", text or "").strip()[:n]


@dataclass
class StateItem:
    item_id: str
    features: Sequence[float]  # FEATURE_NAMES order
    is_new: bool
    text: str
    source: str = ""


@dataclass
class StateFrontier:
    node_id: str
    subquery: str
    gap: float | None = None


def _fmt_feature(name: str, value: float) -> str:
    if name in ("tok", "depth", "age", "src"):
        return str(int(round(value)))
    return f"{value:.2f}"


def serialize_state(
    *,
    root_query: str,
    subquestions: Sequence[str],
    round_id: int,
    tree_depth: int,
    tokens_used: int,
    token_budget: int | None,
    items: Sequence[StateItem],
    frontier: Sequence[StateFrontier],
    feature_names: Sequence[str],
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
) -> str:
    lines: list[str] = []
    lines.append(f"Root question: {root_query.strip()}")
    if subquestions:
        lines.append("Sub-questions:")
        lines.extend(f"  - {s.strip()}" for s in subquestions)
    budget_txt = f"{int(token_budget)}" if token_budget else "none"
    lines.append(
        f"Round {int(round_id)} | depth {int(tree_depth)} | tokens used {int(tokens_used)} "
        f"| context budget {budget_txt} | open branches {len(frontier)}"
    )
    lines.append("")
    lines.append(f"Evidence (id | {' '.join(feature_names)} | status | source | snippet):")
    for it in items:
        feats = " ".join(_fmt_feature(n, float(v)) for n, v in zip(feature_names, it.features))
        status = "new" if it.is_new else "retained"
        lines.append(f"  {it.item_id} | {feats} | {status} | {it.source or '-'} | {snippet(it.text, snippet_chars)}")
    lines.append("")
    if frontier:
        lines.append("Open branches (node_id | gap | sub-query):")
        for f in frontier:
            gap = "-" if f.gap is None else f"{float(f.gap):.2f}"
            lines.append(f"  {f.node_id} | {gap} | {snippet(f.subquery, 200)}")
    else:
        lines.append("Open branches: none")
    lines.append("")
    lines.append(
        "Respond with exactly three lines:\n"
        "KEEP: <space-separated evidence ids to keep, or ALL>\n"
        "ALLOC: <node_id>=<weight> <node_id>=<weight> ... (weights are relative)\n"
        "DECISION: CONTINUE or TERMINATE"
    )
    return "\n".join(lines)


def serialize_action(
    *,
    kept_ids: Iterable[str],
    pool_ids: Sequence[str],
    branch_allocation: Mapping[str, float],
    terminate: bool,
) -> str:
    kept = [i for i in pool_ids if i in set(kept_ids)]  # pool order, deterministic
    keep_txt = "ALL" if pool_ids and len(kept) == len(pool_ids) else " ".join(kept)
    total = sum(max(0.0, float(v)) for v in branch_allocation.values())
    if branch_allocation and total > 0:
        alloc_txt = " ".join(
            f"{nid}={max(0.0, float(w)) / total:.2f}" for nid, w in branch_allocation.items()
        )
    else:
        alloc_txt = "-"
    return f"KEEP: {keep_txt}\nALLOC: {alloc_txt}\nDECISION: {'TERMINATE' if terminate else 'CONTINUE'}"


_KEEP_RE = re.compile(r"^\s*KEEP\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_ALLOC_RE = re.compile(r"^\s*ALLOC\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_DECISION_RE = re.compile(r"^\s*DECISION\s*:\s*(CONTINUE|TERMINATE)", re.IGNORECASE | re.MULTILINE)
_PAIR_RE = re.compile(r"([A-Za-z0-9_\-]+)\s*=\s*([0-9]*\.?[0-9]+)")


@dataclass
class ParsedAction:
    terminate: bool
    kept_ids: set[str]
    branch_allocation: dict[str, float]
    meta: dict[str, Any]


def parse_action(text: str, pool_ids: Iterable[str], node_ids: Iterable[str]) -> ParsedAction:
    """Parse the three-line decision. Failures degrade to keep-all / uniform /
    continue and are flagged in ``meta["parse_failure"]``."""
    pool = set(pool_ids)
    nodes = list(dict.fromkeys(node_ids))
    meta: dict[str, Any] = {"raw": (text or "")[:4000], "parse_failure": False}

    keep_m = _KEEP_RE.search(text or "")
    if not keep_m:
        meta["parse_failure"] = True
        kept = set(pool)
    else:
        body = keep_m.group(1).strip()
        if body.upper() == "ALL":
            kept = set(pool)
        else:
            toks = re.findall(r"[A-Za-z0-9_\-]+", body)
            kept = {t for t in toks if t in pool}
            if not kept and pool:
                meta["parse_failure"] = True
                kept = set(pool)

    alloc: dict[str, float] = {nid: (1.0 / len(nodes)) for nid in nodes} if nodes else {}
    alloc_m = _ALLOC_RE.search(text or "")
    if alloc_m and nodes:
        node_set = set(nodes)
        parsed = {k: float(v) for k, v in _PAIR_RE.findall(alloc_m.group(1)) if k in node_set}
        if parsed and sum(parsed.values()) > 0:
            s = sum(parsed.values())
            alloc = {k: v / s for k, v in parsed.items()}
            for nid in node_set - parsed.keys():
                alloc[nid] = 0.0

    dec_m = _DECISION_RE.search(text or "")
    terminate = bool(dec_m and dec_m.group(1).upper() == "TERMINATE")
    return ParsedAction(terminate=terminate, kept_ids=kept, branch_allocation=alloc, meta=meta)

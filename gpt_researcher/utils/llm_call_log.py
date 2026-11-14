"""Per-call LLM log: one record per create_chat_completion call.

Each record has the call's start and end (epoch seconds), the requested and
served service tier, how many normal attempts it used, how many Flex
``429 no_capacity`` retries it waited through and for how long, timeouts, and
the tokens it was charged. ``capacity_wait_s`` lets Flex queueing be taken out
of latency. Process-global like TokenTracker; reset it at the same points.
"""
from __future__ import annotations

from typing import Any, Dict, List


class LLMCallLog:
    records: List[Dict[str, Any]] = []

    @staticmethod
    def record(entry: Dict[str, Any]) -> None:
        LLMCallLog.records.append(dict(entry))

    @staticmethod
    def get_records() -> List[Dict[str, Any]]:
        return [dict(r) for r in LLMCallLog.records]

    @staticmethod
    def summary() -> Dict[str, Any]:
        recs = LLMCallLog.records
        return {
            "n_calls": len(recs),
            "n_failed": sum(1 for r in recs if r.get("status") != "ok"),
            "attempts": sum(int(r.get("attempts", 0)) for r in recs),
            "capacity_retries": sum(int(r.get("capacity_retries", 0)) for r in recs),
            "capacity_wait_s": round(sum(float(r.get("capacity_wait_s", 0.0)) for r in recs), 4),
            "timeouts": sum(int(r.get("timeouts", 0)) for r in recs),
            "call_s": round(sum(float(r.get("duration_s", 0.0)) for r in recs), 4),
        }

    @staticmethod
    def reset() -> None:
        LLMCallLog.records.clear()

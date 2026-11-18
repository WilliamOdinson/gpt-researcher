from typing import Dict, List, Optional, Tuple


class LatencyTracker:
    """Global latency tracker for API call profiling.

    ``total_latency`` is the sum of call durations. Calls of one type can run
    concurrently (searches for parallel sub-queries are run in threads), so that
    sum can exceed wall time. When the caller passes ``start``, the call's
    ``(start, end)`` interval is kept in ``intervals`` and ``active_seconds``
    returns the time during which at least one call of that type was running,
    which never exceeds wall time.
    """

    per_type_latencies: Dict[str, Dict] = {}
    per_source_latencies: Dict[str, Dict] = {}

    @staticmethod
    def _entry() -> Dict:
        return {"count": 0, "total_latency": 0.0, "calls": [], "intervals": []}

    @staticmethod
    def track_latency(
        call_type: str, latency: float, source: str = None, start: Optional[float] = None
    ):
        buckets = [LatencyTracker.per_type_latencies.setdefault(call_type, LatencyTracker._entry())]
        if source:
            buckets.append(LatencyTracker.per_source_latencies.setdefault(source, LatencyTracker._entry()))
        for b in buckets:
            b["count"] += 1
            b["total_latency"] += latency
            b["calls"].append(latency)
            if start is not None:
                b.setdefault("intervals", []).append((float(start), float(start) + float(latency)))

    @staticmethod
    def union_seconds(intervals: List[Tuple[float, float]]) -> float:
        """Length of the union of ``(start, end)`` intervals."""
        total, cur_s, cur_e = 0.0, None, None
        for s, e in sorted(intervals):
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    total += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            total += cur_e - cur_s
        return total

    @staticmethod
    def active_seconds(call_type: str) -> Optional[float]:
        """Wall time during which at least one call of ``call_type`` was running,
        or None when no intervals were recorded for it."""
        entry = LatencyTracker.per_type_latencies.get(call_type) or {}
        intervals = entry.get("intervals") or []
        if not intervals:
            return None
        return LatencyTracker.union_seconds(intervals)

    @staticmethod
    def snapshot() -> Dict[str, int]:
        """Return frozen per-type call counts for before/after diffing."""
        return {
            call_type: d.get("count", 0)
            for call_type, d in LatencyTracker.per_type_latencies.items()
        }

    @staticmethod
    def reset():
        LatencyTracker.per_type_latencies.clear()
        LatencyTracker.per_source_latencies.clear()

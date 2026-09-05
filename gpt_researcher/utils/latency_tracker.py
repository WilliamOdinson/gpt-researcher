from typing import Dict


class LatencyTracker:
    """Global latency tracker for API call profiling."""

    per_type_latencies: Dict[str, Dict] = {}
    per_source_latencies: Dict[str, Dict] = {}

    @staticmethod
    def track_latency(call_type: str, latency: float, source: str = None):
        if call_type not in LatencyTracker.per_type_latencies:
            LatencyTracker.per_type_latencies[call_type] = {
                "count": 0, "total_latency": 0.0, "calls": []
            }
        LatencyTracker.per_type_latencies[call_type]["count"] += 1
        LatencyTracker.per_type_latencies[call_type]["total_latency"] += latency
        LatencyTracker.per_type_latencies[call_type]["calls"].append(latency)

        if source:
            if source not in LatencyTracker.per_source_latencies:
                LatencyTracker.per_source_latencies[source] = {
                    "count": 0, "total_latency": 0.0, "calls": []
                }
            LatencyTracker.per_source_latencies[source]["count"] += 1
            LatencyTracker.per_source_latencies[source]["total_latency"] += latency
            LatencyTracker.per_source_latencies[source]["calls"].append(latency)

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

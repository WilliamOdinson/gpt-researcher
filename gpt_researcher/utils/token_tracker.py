from typing import Dict


class TokenTracker:
    """Global token tracker that aggregates by model and usage tag."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0

    per_model_totals: Dict[str, Dict[str, float | int]] = {}
    per_usage_totals: Dict[str, Dict[str, float | int]] = {}

    @staticmethod
    def track_tokens(model: str, input_tokens: int, output_tokens: int, cost: float = 0.0, usage_tag: str | None = None):
        TokenTracker.total_input_tokens += int(input_tokens or 0)
        TokenTracker.total_output_tokens += int(output_tokens or 0)
        TokenTracker.total_cost += float(cost or 0.0)

        if model not in TokenTracker.per_model_totals:
            TokenTracker.per_model_totals[model] = {"input": 0, "output": 0, "cost": 0.0}
        TokenTracker.per_model_totals[model]["input"] += int(input_tokens or 0)
        TokenTracker.per_model_totals[model]["output"] += int(output_tokens or 0)
        TokenTracker.per_model_totals[model]["cost"] += float(cost or 0.0)

        tag = usage_tag or "unspecified"
        if tag not in TokenTracker.per_usage_totals:
            TokenTracker.per_usage_totals[tag] = {"input": 0, "output": 0, "cost": 0.0}
        TokenTracker.per_usage_totals[tag]["input"] += int(input_tokens or 0)
        TokenTracker.per_usage_totals[tag]["output"] += int(output_tokens or 0)
        TokenTracker.per_usage_totals[tag]["cost"] += float(cost or 0.0)

    @staticmethod
    def get_totals() -> Dict[str, float | int]:
        return {
            "input_tokens": TokenTracker.total_input_tokens,
            "output_tokens": TokenTracker.total_output_tokens,
            "cost": TokenTracker.total_cost,
        }

    @staticmethod
    def snapshot() -> Dict[str, float | int]:
        """Return a frozen copy of current totals for before/after diffing."""
        return {
            "input_tokens": TokenTracker.total_input_tokens,
            "output_tokens": TokenTracker.total_output_tokens,
            "cost": TokenTracker.total_cost,
        }

    @staticmethod
    def get_per_usage_totals() -> Dict[str, Dict[str, float | int]]:
        return TokenTracker.per_usage_totals

    @staticmethod
    def reset():
        TokenTracker.total_input_tokens = 0
        TokenTracker.total_output_tokens = 0
        TokenTracker.total_cost = 0.0
        TokenTracker.per_model_totals.clear()
        TokenTracker.per_usage_totals.clear()

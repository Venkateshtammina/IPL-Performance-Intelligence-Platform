"""Compatibility facade for strategy APIs."""

from src.bowling_optimizer import optimize_bowling_plan
from src.match_simulation import simulate_dynamic_over, simulate_over_outcomes

__all__ = [
    "optimize_bowling_plan",
    "simulate_dynamic_over",
    "simulate_over_outcomes",
]


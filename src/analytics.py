"""Backward-compatible public analytics API.

New code should import from the focused domain modules:
``feature_context``, ``model_components``, ``probability``, and ``strategy``.
This facade remains stable for existing callers and serialized model artifacts.
"""

from src.feature_context import (
    build_event_feature_frame,
    build_win_feature_frame,
    canonicalize_venue_name,
)
from src.model_components import (
    EventProbabilityBundle,
    HistoricalEventRateTransformer,
    HistoricalRunOutcomeBaseline,
    OutcomeDriftAdapter,
)
from src.probability import (
    blend_mean_with_matchup_evidence,
    blend_model_with_matchup_evidence,
    estimate_chase_win_percentage,
    estimate_high_pressure_from_history,
    matchup_confidence_label,
)
from src.strategy import (
    optimize_bowling_plan,
    simulate_dynamic_over,
    simulate_over_outcomes,
)

__all__ = [
    "EventProbabilityBundle",
    "HistoricalEventRateTransformer",
    "HistoricalRunOutcomeBaseline",
    "OutcomeDriftAdapter",
    "blend_mean_with_matchup_evidence",
    "blend_model_with_matchup_evidence",
    "build_event_feature_frame",
    "build_win_feature_frame",
    "canonicalize_venue_name",
    "estimate_chase_win_percentage",
    "estimate_high_pressure_from_history",
    "matchup_confidence_label",
    "optimize_bowling_plan",
    "simulate_dynamic_over",
    "simulate_over_outcomes",
]

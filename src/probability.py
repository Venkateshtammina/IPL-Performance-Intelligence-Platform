"""Historical evidence blending and chase probability helpers."""

import numpy as np
import pandas as pd

from src.feature_context import canonicalize_venue_name


def blend_model_with_matchup_evidence(
    model_probability,
    event_count,
    ball_count,
    prior_strength=24.0,
):
    balls = max(float(ball_count), 0.0)
    events = min(max(float(event_count), 0.0), balls)
    prior = float(np.clip(model_probability, 0.0, 1.0))
    return (events + prior_strength * prior) / (balls + prior_strength)


def blend_mean_with_matchup_evidence(
    model_mean,
    observed_total,
    ball_count,
    prior_strength=24.0,
):
    balls = max(float(ball_count), 0.0)
    total = max(float(observed_total), 0.0)
    prior = max(float(model_mean), 0.0)
    return (total + prior_strength * prior) / (balls + prior_strength)


def matchup_confidence_label(ball_count):
    balls = max(float(ball_count), 0.0)
    if balls >= 60:
        return "High"
    if balls >= 24:
        return "Medium"
    return "Low"

def estimate_high_pressure_from_history(
    historical_states,
    balls_remaining,
    runs_needed,
    wickets_lost,
    striker,
    non_striker,
    venue,
):
    if historical_states is None or historical_states.empty:
        return None

    required_run_rate = runs_needed * 6.0 / max(float(balls_remaining), 1.0)
    required_columns = {
        "MATCH_ID",
        "BALLS_REMAINING",
        "RUNS_NEEDED",
        "CURRENT_WICKETS_LOST",
        "REQUIRED_RUN_RATE",
        "CHASE_WON",
    }
    if not required_columns.issubset(historical_states.columns):
        return None

    states = historical_states.dropna(subset=list(required_columns)).copy()
    states["distance"] = np.sqrt(
        ((states["BALLS_REMAINING"] - balls_remaining) / 4.0) ** 2
        + ((states["RUNS_NEEDED"] - runs_needed) / 8.0) ** 2
        + ((states["CURRENT_WICKETS_LOST"] - wickets_lost) / 1.5) ** 2
        + ((states["REQUIRED_RUN_RATE"] - required_run_rate) / 4.0) ** 2
    )
    if "STRIKER" in states.columns:
        states["distance"] += np.where(states["STRIKER"] == striker, 0.0, 0.35)
    if "NON_STRIKER" in states.columns:
        states["distance"] += np.where(
            states["NON_STRIKER"] == non_striker, 0.0, 0.15
        )
    if "VENUE" in states.columns:
        canonical_venues = states["VENUE"].map(canonicalize_venue_name)
        states["distance"] += np.where(
            canonical_venues == canonicalize_venue_name(venue), 0.0, 0.10
        )

    nearest_matches = (
        states.sort_values("distance")
        .drop_duplicates("MATCH_ID")
        .head(300)
    )
    if nearest_matches.empty:
        return None

    distances = nearest_matches["distance"].to_numpy()
    weights = np.exp(-0.5 * distances ** 2)
    if weights.sum() == 0:
        weights = 1.0 / (distances + 0.25)
    return float(
        np.average(nearest_matches["CHASE_WON"].to_numpy(), weights=weights) * 100
    )


def _apply_wicket_resource_adjustment(probability, wickets_in_hand):
    bounded_probability = np.clip(probability / 100.0, 0.01, 0.99)
    log_odds = np.log(bounded_probability / (1.0 - bounded_probability))
    adjusted_log_odds = log_odds + 0.18 * (wickets_in_hand - 5)
    return 100.0 / (1.0 + np.exp(-adjusted_log_odds))


def estimate_chase_win_percentage(
    balls_left,
    wickets_lost,
    current_score,
    target_score,
    current_run_rate,
    historical_states=None,
    striker_name=None,
):
    runs_needed = target_score - current_score
    if runs_needed <= 0:
        return 100.0
    if balls_left <= 0 or wickets_lost >= 10:
        return 0.0

    required_run_rate = runs_needed * 6.0 / balls_left
    wickets_in_hand = 10 - wickets_lost

    z_score = (
        0.35 * (current_run_rate - required_run_rate)
        + 0.20 * (wickets_in_hand - 5)
        + 0.003 * (balls_left - 60)
    )
    fallback_probability = 100.0 / (1.0 + np.exp(-z_score))

    if historical_states is None or historical_states.empty:
        return float(np.clip(fallback_probability, 1.0, 99.0))

    states = historical_states.dropna(
        subset=[
            "MATCH_ID",
            "BALLS_REMAINING",
            "RUNS_NEEDED",
            "CURRENT_WICKETS_LOST",
            "REQUIRED_RUN_RATE",
            "CHASE_WON",
        ]
    ).copy()
    if states.empty:
        return float(np.clip(fallback_probability, 1.0, 99.0))

    states["distance"] = np.sqrt(
        ((states["BALLS_REMAINING"] - balls_left) / 12.0) ** 2
        + ((states["RUNS_NEEDED"] - runs_needed) / 15.0) ** 2
        + ((states["CURRENT_WICKETS_LOST"] - wickets_lost) / 1.5) ** 2
        + ((states["REQUIRED_RUN_RATE"] - required_run_rate) / 2.0) ** 2
    )

    nearest_per_match = (
        states.sort_values("distance")
        .drop_duplicates("MATCH_ID")
        .head(100)
    )
    if len(nearest_per_match) < 10:
        return float(np.clip(fallback_probability, 1.0, 99.0))

    weights = 1.0 / (nearest_per_match["distance"].to_numpy() + 0.25)
    historical_probability = (
        np.average(nearest_per_match["CHASE_WON"].to_numpy(), weights=weights) * 100.0
    )
    history_weight = min(len(nearest_per_match) / 50.0, 1.0) * 0.8
    situation_probability = (
        history_weight * historical_probability
        + (1.0 - history_weight) * fallback_probability
    )

    player_adjusted_probability = situation_probability
    if striker_name is not None and "STRIKER" in states.columns:
        striker_states = states[states["STRIKER"] == striker_name]
        striker_neighbors = (
            striker_states.sort_values("distance")
            .drop_duplicates("MATCH_ID")
            .head(40)
        )
        if len(striker_neighbors) >= 5:
            striker_weights = 1.0 / (
                striker_neighbors["distance"].to_numpy() + 0.25
            )
            striker_probability = (
                np.average(
                    striker_neighbors["CHASE_WON"].to_numpy(),
                    weights=striker_weights,
                )
                * 100.0
            )
            striker_weight = min(len(striker_neighbors) / 25.0, 1.0) * 0.35
            player_adjusted_probability = (
                striker_weight * striker_probability
                + (1.0 - striker_weight) * situation_probability
            )

    resource_adjusted_probability = _apply_wicket_resource_adjustment(
        player_adjusted_probability, wickets_in_hand
    )
    return float(np.clip(resource_adjusted_probability, 1.0, 99.0))

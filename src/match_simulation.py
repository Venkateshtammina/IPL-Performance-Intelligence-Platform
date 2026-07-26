"""Standalone delivery and over simulation helpers."""

import numpy as np
import pandas as pd

from src.feature_context import build_event_feature_frame, canonicalize_venue_name

def simulate_over_outcomes(
    event_probabilities,
    safe_runs,
    boundary_runs,
    simulations=5000,
    seed=42,
):
    probabilities = np.asarray(event_probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    safe_values = np.asarray(safe_runs if safe_runs else [0, 1, 2, 3])
    boundary_values = np.asarray(boundary_runs if boundary_runs else [4, 6])
    rng = np.random.default_rng(seed)

    events = rng.choice(3, size=(simulations, 6), p=probabilities)
    runs = np.zeros_like(events, dtype=float)
    safe_mask = events == 0
    boundary_mask = events == 1
    runs[safe_mask] = rng.choice(safe_values, size=safe_mask.sum())
    runs[boundary_mask] = rng.choice(
        boundary_values, size=boundary_mask.sum()
    )

    over_runs = runs.sum(axis=1)
    return {
        "expected_runs": float(over_runs.mean()),
        "median_runs": float(np.median(over_runs)),
        "low_runs": float(np.percentile(over_runs, 10)),
        "high_runs": float(np.percentile(over_runs, 90)),
        "wicket_probability": float((events == 2).any(axis=1).mean()),
        "boundary_probability": float((events == 1).any(axis=1).mean()),
    }


def simulate_dynamic_over(
    model,
    striker,
    non_striker,
    bowler,
    balls_remaining,
    wickets_lost,
    current_score,
    target_score,
    current_run_rate,
    required_run_rate,
    safe_runs,
    boundary_runs,
    innings=2,
    batting_hands=None,
    bowling_types=None,
    simulations=1500,
    seed=42,
    replacement_batter="Replacement Batter",
    max_deliveries=6,
    milestones=(160, 180, 200),
    striker_recent_boundary_rate=0.0,
    bowler_recent_boundary_rate=0.0,
    bowler_recent_dismissal_rate=0.0,
    striker_recent_run_rate=0.0,
    bowler_recent_run_rate=0.0,
    venue="Unknown",
    venue_recent_run_rate=0.0,
):
    batting_hands = batting_hands or {}
    bowling_types = bowling_types or {}
    rng = np.random.default_rng(seed)
    scores = np.full(simulations, float(current_score))
    wickets = np.full(simulations, int(wickets_lost))
    balls = np.full(simulations, int(balls_remaining))
    striker_names = np.full(simulations, striker, dtype=object)
    non_striker_names = np.full(simulations, non_striker, dtype=object)
    wicket_in_over = np.zeros(simulations, dtype=bool)
    boundary_in_over = np.zeros(simulations, dtype=bool)
    safe_values = np.asarray(safe_runs if safe_runs else [0, 1, 2, 3])
    boundary_values = np.asarray(boundary_runs if boundary_runs else [4, 6])
    initial_score = float(current_score)

    deliveries_to_simulate = min(
        max(int(max_deliveries), 0), max(int(balls_remaining), 0)
    )
    for delivery_index in range(deliveries_to_simulate):
        active = (balls > 0) & (wickets < 10)
        if innings == 2:
            active &= scores < target_score
        if not active.any():
            break

        completed_balls = 120 - balls[active]
        simulated_crr = np.where(
            completed_balls > 0,
            scores[active] * 6.0 / completed_balls,
            current_run_rate,
        )
        simulated_runs_needed = np.maximum(target_score - scores[active], 0)
        simulated_rrr = np.where(
            (innings == 2) & (balls[active] > 0),
            simulated_runs_needed * 6.0 / balls[active],
            0.0,
        )
        active_strikers = striker_names[active]
        active_non_strikers = non_striker_names[active]
        phase = np.where(
            balls[active] > 84,
            "powerplay",
            np.where(balls[active] > 30, "middle", "death"),
        )
        features = pd.DataFrame(
            {
                "striker": active_strikers,
                "bowler": bowler,
                "venue": canonicalize_venue_name(venue),
                "striker_bowler": [
                    f"{active_striker}|{bowler}"
                    for active_striker in active_strikers
                ],
                "innings_context": (
                    "batting_first" if innings == 1 else "chasing"
                ),
                "innings_phase": phase,
                "batting_hand": [
                    batting_hands.get(active_striker, "Unknown")
                    for active_striker in active_strikers
                ],
                "bowling_type": bowling_types.get(bowler, "Unknown"),
                "season_context": "current",
                "balls_remaining": balls[active],
                "current_team_score": scores[active],
                "current_wickets_lost": wickets[active],
                "current_run_rate": simulated_crr,
                "required_run_rate": simulated_rrr,
                "striker_recent_boundary_rate": striker_recent_boundary_rate,
                "bowler_recent_boundary_rate": bowler_recent_boundary_rate,
                "bowler_recent_dismissal_rate": bowler_recent_dismissal_rate,
                "striker_recent_run_rate": striker_recent_run_rate,
                "bowler_recent_run_rate": bowler_recent_run_rate,
                "venue_recent_run_rate": venue_recent_run_rate,
            }
        )
        if (
            hasattr(model, "predict_outcome_proba")
            and getattr(model, "run_model", None) is not None
        ):
            outcome_probabilities = model.predict_outcome_proba(features)
            cumulative_probability = np.cumsum(
                outcome_probabilities, axis=1
            )
            outcome_indices = (
                rng.random(len(features))[:, None] > cumulative_probability
            ).sum(axis=1)
            outcome_indices = np.minimum(
                outcome_indices, len(model.outcome_classes_) - 1
            )
            events = model.outcome_classes_[outcome_indices]
            delivery_runs = model.outcome_run_values_[outcome_indices].astype(
                int
            )
            boundary_mask = np.isin(events, [4, 6])
            wicket_mask = events == 7
        else:
            raw_probabilities = model.predict_proba(features)
            class_indices = {
                label: index for index, label in enumerate(model.classes_)
            }
            probabilities = raw_probabilities[
                :, [class_indices[0], class_indices[1], class_indices[2]]
            ]
            draws = rng.random(len(features))
            events = np.where(
                draws < probabilities[:, 0],
                0,
                np.where(
                    draws < probabilities[:, 0] + probabilities[:, 1], 1, 2
                ),
            )
            delivery_runs = np.zeros(len(features), dtype=int)
            safe_mask = events == 0
            boundary_mask = events == 1
            delivery_runs[safe_mask] = rng.choice(
                safe_values, size=safe_mask.sum()
            )
            delivery_runs[boundary_mask] = rng.choice(
                boundary_values, size=boundary_mask.sum()
            )
            wicket_mask = events == 2

        active_indices = np.flatnonzero(active)
        scores[active_indices] += delivery_runs
        balls[active_indices] -= 1
        wicket_indices = active_indices[wicket_mask]
        wickets[wicket_indices] += 1
        wicket_in_over[wicket_indices] = True
        striker_names[wicket_indices] = replacement_batter
        boundary_in_over[active_indices[boundary_mask]] = True

        odd_run_indices = active_indices[(delivery_runs % 2) == 1]
        previous_strikers = striker_names[odd_run_indices].copy()
        striker_names[odd_run_indices] = non_striker_names[odd_run_indices]
        non_striker_names[odd_run_indices] = previous_strikers
        if (delivery_index + 1) % 6 == 0:
            over_active_indices = np.flatnonzero(active)
            over_end_strikers = striker_names[over_active_indices].copy()
            striker_names[over_active_indices] = non_striker_names[
                over_active_indices
            ]
            non_striker_names[over_active_indices] = over_end_strikers

    over_runs = scores - initial_score
    result = {
        "expected_runs": float(over_runs.mean()),
        "median_runs": float(np.median(over_runs)),
        "low_runs": float(np.percentile(over_runs, 10)),
        "high_runs": float(np.percentile(over_runs, 90)),
        "wicket_probability": float(wicket_in_over.mean()),
        "boundary_probability": float(boundary_in_over.mean()),
        "expected_final_score": float(scores.mean()),
        "final_score_low": float(np.percentile(scores, 10)),
        "final_score_high": float(np.percentile(scores, 90)),
    }
    for milestone in milestones:
        result[f"reach_{milestone}_probability"] = float(
            (scores >= milestone).mean()
        )
    return result


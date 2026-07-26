import pandas as pd

import numpy as np

from src.analytics import (
    EventProbabilityBundle,
    HistoricalEventRateTransformer,
    HistoricalRunOutcomeBaseline,
    OutcomeDriftAdapter,
    build_event_feature_frame,
    build_win_feature_frame,
    blend_mean_with_matchup_evidence,
    blend_model_with_matchup_evidence,
    canonicalize_venue_name,
    estimate_high_pressure_from_history,
    estimate_chase_win_percentage,
    matchup_confidence_label,
    optimize_bowling_plan,
    simulate_dynamic_over,
    simulate_over_outcomes,
)
from src.feature_context import build_event_feature_frame as focused_event_builder
from src.strategy import optimize_bowling_plan as focused_plan_optimizer


def test_analytics_facade_preserves_focused_public_api():
    assert focused_event_builder is build_event_feature_frame
    assert focused_plan_optimizer is optimize_bowling_plan


class FixedProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self, positive_probability):
        self.positive_probability = positive_probability

    def predict_proba(self, features):
        positive = np.full(len(features), self.positive_probability)
        return np.column_stack([1.0 - positive, positive])


class FixedRunModel:
    classes_ = np.array([0, 1, 2, 3, 4, 6])

    def predict_proba(self, features):
        probabilities = np.array([0.40, 0.30, 0.10, 0.05, 0.10, 0.05])
        return np.tile(probabilities, (len(features), 1))


class BowlerSensitiveOutcomeModel:
    outcome_classes_ = np.array([0, 1, 2, 3, 4, 6, 7])
    outcome_run_values_ = np.array([0, 1, 2, 3, 4, 6, 0], dtype=float)
    run_model = object()

    def predict_outcome_proba(self, features):
        probabilities = []
        for bowler in features["bowler"]:
            if bowler == "Control":
                probabilities.append(
                    [0.62, 0.22, 0.05, 0.01, 0.05, 0.01, 0.04]
                )
            elif bowler == "Attack":
                probabilities.append(
                    [0.48, 0.24, 0.06, 0.01, 0.10, 0.03, 0.08]
                )
            else:
                probabilities.append(
                    [0.34, 0.25, 0.08, 0.01, 0.20, 0.08, 0.04]
                )
        return np.asarray(probabilities)


def test_bowling_optimizer_enforces_constraints_and_prefers_control():
    plan = optimize_bowling_plan(
        BowlerSensitiveOutcomeModel(),
        ["Control", "Attack", "Expensive"],
        "Batter A",
        "Batter B",
        balls_remaining=18,
        wickets_lost=4,
        current_score=120,
        current_run_rate=8.0,
        required_run_rate=12.0,
        overs_used={"Control": 2, "Attack": 3, "Expensive": 4},
        overs_to_plan=2,
        innings=1,
        incoming_batters=["Batter C"],
        beam_width=6,
        monte_carlo_simulations=40,
    )

    sequence = plan["primary"]["sequence"]

    assert sequence[0] == "Control"
    assert sequence[0] != sequence[1]
    assert "Expensive" not in sequence


def test_bowling_optimizer_excludes_previous_over_bowler_first():
    plan = optimize_bowling_plan(
        BowlerSensitiveOutcomeModel(),
        ["Control", "Attack"],
        "Batter A",
        "Batter B",
        balls_remaining=12,
        wickets_lost=4,
        current_score=120,
        current_run_rate=8.0,
        required_run_rate=12.0,
        previous_bowler="Control",
        overs_to_plan=1,
        innings=1,
        monte_carlo_simulations=40,
    )

    assert plan["primary"]["sequence"] == ["Attack"]


def test_bowling_optimizer_reports_stochastic_plan_validation():
    matchup_context = {
        ("Batter A", "Control"): {"TOTAL_BALLS": 30},
        ("Batter B", "Control"): {"TOTAL_BALLS": 10},
        ("Batter A", "Attack"): {"TOTAL_BALLS": 18},
        ("Batter B", "Attack"): {"TOTAL_BALLS": 6},
        ("Batter A", "Expensive"): {"TOTAL_BALLS": 4},
        ("Batter B", "Expensive"): {"TOTAL_BALLS": 2},
    }
    plan = optimize_bowling_plan(
        BowlerSensitiveOutcomeModel(),
        ["Control", "Attack", "Expensive"],
        "Batter A",
        "Batter B",
        balls_remaining=12,
        wickets_lost=4,
        current_score=120,
        current_run_rate=8.0,
        required_run_rate=12.0,
        overs_to_plan=2,
        innings=1,
        incoming_batters=["Batter C"],
        matchup_context=matchup_context,
        monte_carlo_simulations=80,
        seed=9,
    )

    simulation = plan["primary"]["monte_carlo"]

    assert simulation["runs_low"] <= simulation["runs_high"]
    assert simulation["expected_wickets"] >= 0
    assert simulation["incoming_batter_used_probability"] > 0
    assert 0 <= plan["validation"]["primary_better_probability"] <= 1
    assert plan["validation"]["simulations"] == 80
    first_step = plan["primary"]["steps"][0]
    expected_evidence = (
        matchup_context[("Batter A", first_step["bowler"])]["TOTAL_BALLS"]
        + matchup_context[("Batter B", first_step["bowler"])]["TOTAL_BALLS"]
    ) / 2
    assert first_step["evidence_balls"] == expected_evidence


def test_granular_outcomes_preserve_calibrated_event_totals():
    model = EventProbabilityBundle(
        FixedProbabilityModel(0.20),
        FixedProbabilityModel(0.10),
        run_model=FixedRunModel(),
    )
    features = build_event_feature_frame(
        "Batter A", "Bowler A", 24, 4, 8.0, 10.0
    )

    probabilities = model.predict_outcome_proba(features)[0]

    assert np.isclose(probabilities.sum(), 1.0)
    assert np.isclose(probabilities[4:6].sum(), 0.20)
    assert np.isclose(probabilities[-1], 0.10)
    assert np.isclose(probabilities[:4].sum(), 0.70)


def test_expected_run_evidence_blend_uses_observed_matchup_mean():
    no_evidence = blend_mean_with_matchup_evidence(1.2, 0, 0)
    strong_low_scoring_evidence = blend_mean_with_matchup_evidence(
        1.2, observed_total=30, ball_count=60
    )

    assert np.isclose(no_evidence, 1.2)
    assert strong_low_scoring_evidence < no_evidence


def test_historical_run_baseline_is_phase_aware_and_normalized():
    powerplay = build_event_feature_frame(
        "Batter A", "Bowler A", 100, 1, 9.0, 0.0, innings=1
    )
    death = build_event_feature_frame(
        "Batter A", "Bowler A", 12, 4, 9.0, 14.0
    )
    training = pd.concat(
        [powerplay] * 20 + [death] * 20,
        ignore_index=True,
    )
    target = np.array([0] * 20 + [6] * 20)
    baseline = HistoricalRunOutcomeBaseline(smoothing=1.0).fit(
        training, target
    )

    probabilities = baseline.predict_proba(
        pd.concat([powerplay, death], ignore_index=True)
    )

    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0, 0] > probabilities[1, 0]
    assert probabilities[1, -1] > probabilities[0, -1]


def test_outcome_drift_adapter_corrects_phase_rates_and_normalizes():
    death = build_event_feature_frame(
        "Batter A", "Bowler A", 12, 4, 9.0, 14.0
    )
    features = pd.concat([death] * 20, ignore_index=True)
    classes = np.array([0, 1, 2, 3, 4, 6, 7])
    base_probability = np.tile(
        [0.40, 0.30, 0.08, 0.02, 0.10, 0.05, 0.05],
        (20, 1),
    )
    target = np.array([6] * 10 + [0] * 10)
    adapter = OutcomeDriftAdapter(smoothing=1.0).fit(
        features, target, base_probability, classes
    )

    adjusted = adapter.transform(features.iloc[:1], base_probability[:1])

    assert np.isclose(adjusted.sum(), 1.0)
    assert adjusted[0, 5] > base_probability[0, 5]


def test_terminal_chase_states():
    assert estimate_chase_win_percentage(30, 3, 170, 160, 8.0) == 100.0
    assert estimate_chase_win_percentage(0, 3, 120, 160, 8.0) == 0.0
    assert estimate_chase_win_percentage(30, 10, 120, 160, 8.0) == 0.0


def test_probability_responds_to_match_pressure():
    comfortable = estimate_chase_win_percentage(30, 3, 140, 160, 8.0)
    difficult = estimate_chase_win_percentage(30, 7, 110, 160, 6.0)
    assert comfortable > difficult


def test_history_counts_each_match_once():
    repeated_loss = pd.DataFrame(
        [
            {
                "MATCH_ID": "loss",
                "BALLS_REMAINING": 30,
                "RUNS_NEEDED": 30,
                "CURRENT_WICKETS_LOST": 4,
                "REQUIRED_RUN_RATE": 6.0,
                "CHASE_WON": 0,
            }
        ]
        * 200
    )
    wins = pd.DataFrame(
        [
            {
                "MATCH_ID": f"win-{index}",
                "BALLS_REMAINING": 30,
                "RUNS_NEEDED": 30,
                "CURRENT_WICKETS_LOST": 4,
                "REQUIRED_RUN_RATE": 6.0,
                "CHASE_WON": 1,
            }
            for index in range(20)
        ]
    )

    probability = estimate_chase_win_percentage(
        30, 4, 130, 160, 8.0, pd.concat([repeated_loss, wins], ignore_index=True)
    )
    assert probability > 50.0


def test_event_feature_frame_preserves_matchup_identity():
    features = build_event_feature_frame(
        "AB de Villiers", "A Mishra", 24, 6, 8.0, 12.25
    )
    assert features.iloc[0]["striker_bowler"] == "AB de Villiers|A Mishra"
    assert features.iloc[0]["current_wickets_lost"] == 6


def test_event_feature_frame_supports_batting_first():
    features = build_event_feature_frame(
        "Batter A",
        "Bowler A",
        60,
        2,
        8.5,
        0.0,
        innings=1,
        current_team_score=85,
    )

    assert features.iloc[0]["innings_context"] == "batting_first"
    assert features.iloc[0]["current_team_score"] == 85
    assert features.iloc[0]["innings_phase"] == "middle"


def test_over_simulation_is_reproducible_and_bounded():
    forecast = simulate_over_outcomes(
        [0.70, 0.20, 0.10],
        safe_runs=[0, 1, 2],
        boundary_runs=[4, 6],
        simulations=1000,
        seed=7,
    )

    assert 0 <= forecast["low_runs"] <= forecast["high_runs"] <= 36
    assert 0 <= forecast["wicket_probability"] <= 1
    assert 0 <= forecast["boundary_probability"] <= 1


def test_dynamic_over_simulation_updates_state():
    class ConstantEventModel:
        classes_ = np.array([0, 1, 2])

        def predict_proba(self, features):
            return np.tile([0.70, 0.20, 0.10], (len(features), 1))

    forecast = simulate_dynamic_over(
        ConstantEventModel(),
        "Batter A",
        "Batter B",
        "Bowler A",
        balls_remaining=24,
        wickets_lost=4,
        current_score=120,
        target_score=160,
        current_run_rate=7.5,
        required_run_rate=10.0,
        safe_runs=[0, 1, 2],
        boundary_runs=[4, 6],
        simulations=200,
        seed=5,
    )

    assert forecast["expected_runs"] > 0
    assert 0 <= forecast["wicket_probability"] <= 1


def test_historical_rate_transformer_handles_unseen_matchup():
    training = pd.concat(
        [
            build_event_feature_frame(
                "Batter A", "Bowler A", 60, 2, 8.0, 9.0
            ),
            build_event_feature_frame(
                "Batter B", "Bowler B", 24, 5, 7.0, 12.0
            ),
            build_event_feature_frame(
                "Batter A", "Bowler B", 12, 6, 8.0, 15.0
            ),
        ],
        ignore_index=True,
    )
    transformer = HistoricalEventRateTransformer(smoothing=2.0)
    transformer.fit(training, np.array([1, 0, 2]))
    unseen = build_event_feature_frame(
        "New Batter", "New Bowler", 30, 4, 8.0, 10.0
    )
    transformed = transformer.transform(unseen)

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()


def test_matchup_evidence_updates_historical_model_probability():
    model_probability = 0.05
    no_evidence = blend_model_with_matchup_evidence(
        model_probability, event_count=0, ball_count=0
    )
    strong_evidence = blend_model_with_matchup_evidence(
        model_probability, event_count=6, ball_count=60
    )

    assert np.isclose(no_evidence, model_probability)
    assert strong_evidence > no_evidence
    assert matchup_confidence_label(3) == "Low"
    assert matchup_confidence_label(30) == "Medium"
    assert matchup_confidence_label(70) == "High"


def test_batter_history_changes_same_match_situation():
    rows = []
    for index in range(25):
        common = {
            "BALLS_REMAINING": 24,
            "RUNS_NEEDED": 49,
            "CURRENT_WICKETS_LOST": 6,
            "REQUIRED_RUN_RATE": 12.25,
        }
        rows.append(
            {
                **common,
                "MATCH_ID": f"finisher-{index}",
                "STRIKER": "Elite Finisher",
                "CHASE_WON": 1,
            }
        )
        rows.append(
            {
                **common,
                "MATCH_ID": f"other-{index}",
                "STRIKER": "Other Batter",
                "CHASE_WON": 0,
            }
        )
    history = pd.DataFrame(rows)

    elite_probability = estimate_chase_win_percentage(
        24, 6, 111, 160, 8.0, history, "Elite Finisher"
    )
    other_probability = estimate_chase_win_percentage(
        24, 6, 111, 160, 8.0, history, "Other Batter"
    )

    assert elite_probability > other_probability


def test_wicket_losses_have_material_monotonic_impact():
    seven_in_hand = estimate_chase_win_percentage(36, 3, 120, 160, 7.0)
    five_in_hand = estimate_chase_win_percentage(36, 5, 120, 160, 7.0)
    three_in_hand = estimate_chase_win_percentage(36, 7, 120, 160, 7.0)

    assert seven_in_hand > five_in_hand > three_in_hand
    assert seven_in_hand - three_in_hand > 15.0


def test_win_feature_frame_encodes_player_pressure_context():
    features = build_win_feature_frame(
        24, 111, 49, 6, 8.0, 12.25, "AB de Villiers", "Player B", "Venue A"
    )

    assert (
        features.iloc[0]["striker_pressure"]
        == "AB de Villiers|extreme|tail_exposed"
    )
    assert features.iloc[0]["current_wickets_lost"] == 6


def test_win_feature_frame_marks_beyond_sixes_equation():
    features = build_win_feature_frame(
        6, 122, 38, 4, 6.42, 38.0, "Batter A", "Batter B", "Wankhede Stadium"
    )

    assert features.iloc[0]["equation_band"] == "beyond_sixes"
    assert features.iloc[0]["runs_above_six_rate"] == 2.0
    assert features.iloc[0]["required_run_rate_squared"] == 1444.0


def test_high_pressure_estimate_uses_comparable_match_outcomes():
    history = pd.DataFrame(
        [
            {
                "MATCH_ID": f"match-{index}",
                "VENUE": "Wankhede Stadium",
                "STRIKER": "Batter A",
                "NON_STRIKER": "Batter B",
                "BALLS_REMAINING": 12,
                "RUNS_NEEDED": 38,
                "CURRENT_WICKETS_LOST": 3,
                "REQUIRED_RUN_RATE": 19.0,
                "CHASE_WON": 1 if index == 0 else 0,
            }
            for index in range(20)
        ]
    )

    probability = estimate_high_pressure_from_history(
        history, 12, 38, 3, "Batter A", "Batter B", "Wankhede Stadium"
    )
    assert probability == 5.0


def test_venue_city_suffixes_are_deduplicated():
    assert canonicalize_venue_name("Eden Gardens, Kolkata") == "Eden Gardens"
    assert canonicalize_venue_name("Eden Gardens") == "Eden Gardens"
    assert (
        canonicalize_venue_name("Dr DY Patil Sports Academy, Mumbai")
        == "Dr DY Patil Sports Academy"
    )

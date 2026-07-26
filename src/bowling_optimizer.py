"""Bowling-plan optimization and delivery simulations."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.feature_context import (
    build_event_feature_frame,
    build_win_feature_frame,
    canonicalize_venue_name,
)


def optimize_bowling_plan(
    event_model,
    bowlers,
    striker,
    non_striker,
    balls_remaining,
    wickets_lost,
    current_score,
    current_run_rate,
    required_run_rate,
    overs_used=None,
    previous_bowler=None,
    overs_to_plan=3,
    innings=2,
    target_score=0,
    venue="Unknown",
    incoming_batters=None,
    batting_hands=None,
    bowling_types=None,
    matchup_context=None,
    win_model=None,
    beam_width=10,
    monte_carlo_simulations=160,
    seed=42,
):
    overs_used = dict(overs_used or {})
    incoming_batters = list(incoming_batters or [])
    batting_hands = batting_hands or {}
    bowling_types = bowling_types or {}
    matchup_context = matchup_context or {}
    available_overs = sum(
        max(4 - int(overs_used.get(bowler, 0)), 0) for bowler in bowlers
    )
    plan_length = min(
        max(int(overs_to_plan), 0),
        int(np.ceil(max(float(balls_remaining), 0) / 6.0)),
        available_overs,
    )
    if plan_length == 0:
        return {"primary": None, "alternative": None, "candidates": []}

    initial_state = {
        "sequence": [],
        "steps": [],
        "score": float(current_score),
        "balls": float(balls_remaining),
        "wickets": float(wickets_lost),
        "striker": striker,
        "non_striker": non_striker,
        "incoming_index": 0,
        "overs_used": overs_used,
        "previous_bowler": previous_bowler,
        "expected_runs": 0.0,
        "expected_wickets": 0.0,
        "boundary_risk_sum": 0.0,
        "dot_rate_sum": 0.0,
        "objective": 0.0,
    }

    def context_for(batter, bowler):
        return matchup_context.get((batter, bowler), {})

    def outcome_probabilities(batters, bowler, state):
        batter_names = list(batters)
        completed_balls = max(120.0 - state["balls"], 0.0)
        projected_crr = (
            state["score"] * 6.0 / completed_balls
            if completed_balls > 0
            else float(current_run_rate)
        )
        projected_runs_needed = max(float(target_score) - state["score"], 0.0)
        projected_rrr = (
            projected_runs_needed * 6.0 / max(state["balls"], 1.0)
            if int(innings) == 2
            else 0.0
        )
        features = pd.concat(
            [
                build_event_feature_frame(
                    batter,
                    bowler,
                    state["balls"],
                    state["wickets"],
                    projected_crr,
                    projected_rrr,
                    innings=innings,
                    current_team_score=state["score"],
                    batting_hand=batting_hands.get(batter, "Unknown"),
                    bowling_type=bowling_types.get(bowler, "Unknown"),
                    striker_recent_boundary_rate=float(
                        context_for(batter, bowler).get(
                            "STRIKER_RECENT_BOUNDARY_RATE", 0
                        )
                        or 0
                    ),
                    bowler_recent_boundary_rate=float(
                        context_for(batter, bowler).get(
                            "BOWLER_RECENT_BOUNDARY_RATE", 0
                        )
                        or 0
                    ),
                    bowler_recent_dismissal_rate=float(
                        context_for(batter, bowler).get(
                            "BOWLER_RECENT_DISMISSAL_RATE", 0
                        ) or 0
                    ),
                    striker_recent_run_rate=float(
                        context_for(batter, bowler).get(
                            "STRIKER_RECENT_RUN_RATE", 0
                        ) or 0
                    ),
                    bowler_recent_run_rate=float(
                        context_for(batter, bowler).get(
                            "BOWLER_RECENT_RUN_RATE", 0
                        ) or 0
                    ),
                    venue=venue,
                    venue_recent_run_rate=float(
                        context_for(batter, bowler).get(
                            "VENUE_RECENT_RUN_RATE", 0
                        ) or 0
                    ),
                )
                for batter in batter_names
            ],
            ignore_index=True,
        )
        probabilities = event_model.predict_outcome_proba(features)
        return [
            {
                "expected_runs": float(
                    probability @ event_model.outcome_run_values_
                ),
                "dot": float(probability[0]),
                "boundary": float(probability[4] + probability[5]),
                "wicket": float(probability[-1]),
                "probability": probability.copy(),
            }
            for probability in probabilities
        ]

    def chase_probability(state):
        if win_model is None or int(innings) != 2:
            return None
        needed = max(float(target_score) - state["score"], 0.0)
        rrr = needed * 6.0 / max(state["balls"], 1.0)
        completed = max(120.0 - state["balls"], 1.0)
        features = build_win_feature_frame(
            state["balls"], state["score"], needed, state["wickets"],
            state["score"] * 6.0 / completed, rrr, state["striker"],
            state["non_striker"], venue,
        )
        classes = list(win_model.classes_)
        return float(win_model.predict_proba(features)[0][classes.index(1)])

    def simulate_sequence(plan, count, simulation_seed):
        rng = np.random.default_rng(simulation_seed)
        scores = np.full(count, float(current_score))
        wickets = np.full(count, int(wickets_lost))
        balls = np.full(count, int(balls_remaining))
        strikers = np.full(count, striker, dtype=object)
        non_strikers = np.full(count, non_striker, dtype=object)
        incoming_indices = np.zeros(count, dtype=int)
        for over_index, _ in enumerate(plan["sequence"]):
            profiles = plan["steps"][over_index]["outcome_profiles"]
            fallback = np.mean(
                [value["probability"] for value in profiles.values()], axis=0
            )
            for delivery in range(6):
                active = (balls > 0) & (wickets < 10)
                if int(innings) == 2:
                    active &= scores < float(target_score)
                indices = np.flatnonzero(active)
                if not len(indices):
                    break
                probability = np.vstack([
                    profiles.get(name, {"probability": fallback})["probability"]
                    for name in strikers[active]
                ])
                outcomes = (
                    rng.random(len(indices))[:, None]
                    > np.cumsum(probability, axis=1)
                ).sum(axis=1)
                outcomes = np.minimum(
                    outcomes, len(event_model.outcome_classes_) - 1
                )
                labels = event_model.outcome_classes_[outcomes]
                runs = event_model.outcome_run_values_[outcomes].astype(int)
                scores[indices] += runs
                balls[indices] -= 1
                wicket_indices = indices[labels == 7]
                wickets[wicket_indices] += 1
                for wicket_index in wicket_indices:
                    replacement_index = incoming_indices[wicket_index]
                    strikers[wicket_index] = (
                        incoming_batters[replacement_index]
                        if replacement_index < len(incoming_batters)
                        else "Replacement Batter"
                    )
                    incoming_indices[wicket_index] += 1
                odd = indices[(runs % 2) == 1]
                previous = strikers[odd].copy()
                strikers[odd] = non_strikers[odd]
                non_strikers[odd] = previous
                if delivery == 5:
                    previous = strikers[indices].copy()
                    strikers[indices] = non_strikers[indices]
                    non_strikers[indices] = previous
        runs_added = scores - float(current_score)
        comparison = scores.copy()
        projected_win = None
        if win_model is not None and int(innings) == 2:
            frames = []
            for index in range(count):
                needed = max(float(target_score) - scores[index], 0.0)
                rrr = needed * 6.0 / max(balls[index], 1)
                completed = max(120 - balls[index], 1)
                frames.append(build_win_feature_frame(
                    balls[index], scores[index], needed, wickets[index],
                    scores[index] * 6.0 / completed, rrr, strikers[index],
                    non_strikers[index], venue,
                ))
            classes = list(win_model.classes_)
            comparison = win_model.predict_proba(
                pd.concat(frames, ignore_index=True)
            )[:, classes.index(1)]
            projected_win = float(comparison.mean())
        return {
            "expected_runs": float(runs_added.mean()),
            "runs_low": float(np.percentile(runs_added, 10)),
            "runs_high": float(np.percentile(runs_added, 90)),
            "expected_wickets": float((wickets - int(wickets_lost)).mean()),
            "incoming_batter_used_probability": float(
                (incoming_indices > 0).mean()
            ),
            "wickets_high": float(np.percentile(
                wickets - int(wickets_lost), 90
            )),
            "projected_win_probability": projected_win,
            "projected_win_probability_low": (
                float(np.percentile(comparison, 10))
                if projected_win is not None
                else None
            ),
            "projected_win_probability_high": (
                float(np.percentile(comparison, 90))
                if projected_win is not None
                else None
            ),
            "comparison_values": comparison,
        }

    beam = [initial_state]
    for _ in range(plan_length):
        expanded = []
        for state in beam:
            eligible = [
                bowler for bowler in bowlers
                if state["overs_used"].get(bowler, 0) < 4
                and bowler != state["previous_bowler"]
            ]
            for bowler in eligible:
                batters = list(dict.fromkeys([
                    state["striker"], state["non_striker"], *incoming_batters
                ]))
                profiles = dict(zip(
                    batters, outcome_probabilities(batters, bowler, state)
                ))
                first = profiles[state["striker"]]
                second = profiles[state["non_striker"]]
                average = {
                    key: (first[key] + second[key]) / 2.0
                    for key in ("expected_runs", "dot", "boundary", "wicket")
                }
                over_balls = min(6.0, state["balls"])
                over_runs = average["expected_runs"] * over_balls
                over_wickets = average["wicket"] * over_balls
                boundary_risk = 1 - (1 - average["boundary"]) ** over_balls
                next_wickets = min(state["wickets"] + over_wickets, 10.0)
                next_striker = state["striker"]
                next_incoming_index = state["incoming_index"]
                wicket_crossings = max(
                    int(np.floor(next_wickets))
                    - int(np.floor(state["wickets"])),
                    0,
                )
                for _ in range(wicket_crossings):
                    if next_incoming_index < len(incoming_batters):
                        next_striker = incoming_batters[
                            next_incoming_index
                        ]
                        next_incoming_index += 1
                next_overs = dict(state["overs_used"])
                next_overs[bowler] = next_overs.get(bowler, 0) + 1
                evidence_balls = (
                    float(
                        context_for(state["striker"], bowler).get(
                            "TOTAL_BALLS", 0
                        )
                        or 0
                    )
                    + float(
                        context_for(state["non_striker"], bowler).get(
                            "TOTAL_BALLS", 0
                        )
                        or 0
                    )
                ) / 2.0
                next_state = {
                    **state,
                    "sequence": state["sequence"] + [bowler],
                    "steps": state["steps"] + [{
                        "bowler": bowler,
                        "expected_runs": over_runs,
                        "expected_wickets": over_wickets,
                        "dot_probability": average["dot"],
                        "boundary_in_over": boundary_risk,
                        "evidence_balls": evidence_balls,
                        "outcome_profiles": profiles,
                    }],
                    "score": state["score"] + over_runs,
                    "balls": max(state["balls"] - over_balls, 0.0),
                    "wickets": next_wickets,
                    "striker": next_striker,
                    "incoming_index": next_incoming_index,
                    "overs_used": next_overs,
                    "previous_bowler": bowler,
                    "expected_runs": state["expected_runs"] + over_runs,
                    "expected_wickets": (
                        state["expected_wickets"] + over_wickets
                    ),
                    "boundary_risk_sum": (
                        state["boundary_risk_sum"] + boundary_risk
                    ),
                    "dot_rate_sum": (
                        state["dot_rate_sum"] + average["dot"]
                    ),
                }
                win_probability = chase_probability(next_state)
                if win_probability is not None:
                    next_state["projected_batting_win_probability"] = (
                        win_probability
                    )
                    next_state["objective"] = (
                        win_probability
                        + 0.02
                        * next_state["boundary_risk_sum"]
                        / len(next_state["sequence"])
                    )
                else:
                    next_state["projected_batting_win_probability"] = None
                    next_state["objective"] = (
                        next_state["expected_runs"]
                        - 3.5 * next_state["expected_wickets"]
                        + 1.5
                        * next_state["boundary_risk_sum"]
                    )
                expanded.append(next_state)
        if not expanded:
            break
        expanded.sort(key=lambda state: state["objective"])
        beam = expanded[: max(int(beam_width), 1)]

    candidates = sorted(beam, key=lambda state: state["objective"])
    if not candidates:
        return {"primary": None, "alternative": None, "candidates": []}
    primary = candidates[0]
    alternative_pool = [
        candidate
        for candidate in candidates[1:]
        if candidate["sequence"][0] != primary["sequence"][0]
    ]
    if not alternative_pool:
        alternative_pool = candidates[1:]
    alternative = (
        min(
            alternative_pool,
            key=lambda state: (
                state["boundary_risk_sum"] / len(state["sequence"]),
                state["objective"],
            ),
        )
        if alternative_pool
        else None
    )
    validation = None
    if int(monte_carlo_simulations) > 0:
        if alternative is not None:
            with ThreadPoolExecutor(max_workers=2) as executor:
                primary_future = executor.submit(
                    simulate_sequence,
                    primary,
                    int(monte_carlo_simulations),
                    seed,
                )
                alternative_future = executor.submit(
                    simulate_sequence,
                    alternative,
                    int(monte_carlo_simulations),
                    seed,
                )
                primary_simulation = primary_future.result()
                alternative_simulation = alternative_future.result()
        else:
            primary_simulation = simulate_sequence(
                primary,
                int(monte_carlo_simulations),
                seed,
            )
            alternative_simulation = None
        primary_values = primary_simulation.pop("comparison_values")
        primary["monte_carlo"] = primary_simulation
        if alternative is not None:
            alternative_values = alternative_simulation.pop(
                "comparison_values"
            )
            alternative["monte_carlo"] = alternative_simulation
            primary_better_probability = float(
                (
                    (primary_values < alternative_values).astype(float)
                    + 0.5
                    * (primary_values == alternative_values).astype(float)
                ).mean()
            )
            stable = primary_better_probability >= 0.60
            alternative_materially_safer = (
                alternative_simulation["runs_high"] + 2.0
                < primary_simulation["runs_high"]
            )
            preferred_plan = (
                "B"
                if not stable and alternative_materially_safer
                else "A"
            )
            validation = {
                "simulations": int(monte_carlo_simulations),
                "primary_better_probability": primary_better_probability,
                "stable": stable,
                "preferred_plan": preferred_plan,
                "alternative_materially_safer": (
                    alternative_materially_safer
                ),
            }
        else:
            validation = {
                "simulations": int(monte_carlo_simulations),
                "primary_better_probability": 1.0,
                "stable": True,
                "preferred_plan": "A",
                "alternative_materially_safer": False,
            }
    return {
        "primary": primary,
        "alternative": alternative,
        "candidates": candidates,
        "validation": validation,
    }

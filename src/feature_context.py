"""Feature-frame construction and venue normalization."""

import numpy as np
import pandas as pd


VENUE_ALIASES = {
    "M Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
    "MA Chidambaram Stadium": "M. A. Chidambaram Stadium",
    "MA Chidambaram Stadium Chepauk": "M. A. Chidambaram Stadium",
    "Wankhede Stadium Mumbai": "Wankhede Stadium",
    "Arun Jaitley Stadium Delhi": "Arun Jaitley Stadium",
}

def canonicalize_venue_name(venue):
    if venue is None or pd.isna(venue):
        return "Unknown"

    normalized = " ".join(str(venue).strip().split())
    base_name = normalized.split(",", maxsplit=1)[0].strip()
    return VENUE_ALIASES.get(base_name, base_name)


def build_win_feature_frame(
    balls_remaining,
    current_team_score,
    runs_needed,
    wickets_lost,
    current_run_rate,
    required_run_rate,
    striker,
    non_striker,
    venue,
):
    safe_balls_remaining = max(float(balls_remaining), 1.0)
    runs_per_ball_needed = float(runs_needed) / safe_balls_remaining
    if runs_per_ball_needed <= 1.5:
        equation_band = "manageable"
    elif runs_per_ball_needed <= 2.5:
        equation_band = "severe"
    elif runs_per_ball_needed <= 4:
        equation_band = "desperate"
    elif runs_per_ball_needed <= 6:
        equation_band = "near_impossible"
    else:
        equation_band = "beyond_sixes"

    if required_run_rate <= 7:
        pressure_band = "low"
    elif required_run_rate <= 9:
        pressure_band = "medium"
    elif required_run_rate <= 12:
        pressure_band = "high"
    else:
        pressure_band = "extreme"

    if wickets_lost <= 2:
        wicket_band = "healthy"
    elif wickets_lost <= 5:
        wicket_band = "under_pressure"
    elif wickets_lost <= 7:
        wicket_band = "tail_exposed"
    else:
        wicket_band = "last_pair"

    return pd.DataFrame(
        [
            {
                "venue": canonicalize_venue_name(venue),
                "striker": striker,
                "non_striker": non_striker,
                "striker_pressure": f"{striker}|{pressure_band}|{wicket_band}",
                "non_striker_pressure": (
                    f"{non_striker}|{pressure_band}|{wicket_band}"
                ),
                "equation_band": equation_band,
                "balls_remaining": balls_remaining,
                "current_team_score": current_team_score,
                "runs_needed": runs_needed,
                "current_wickets_lost": wickets_lost,
                "current_run_rate": current_run_rate,
                "required_run_rate": required_run_rate,
                "runs_per_ball_needed": runs_per_ball_needed,
                "required_run_rate_squared": required_run_rate ** 2,
                "runs_above_six_rate": max(
                    float(runs_needed) - 6.0 * float(balls_remaining), 0.0
                ),
            }
        ]
    )


def build_event_feature_frame(
    striker,
    bowler,
    balls_remaining,
    wickets_lost,
    current_run_rate,
    required_run_rate,
    innings=2,
    current_team_score=0,
    batting_hand="Unknown",
    bowling_type="Unknown",
    season_context="current",
    striker_recent_boundary_rate=0.0,
    bowler_recent_boundary_rate=0.0,
    bowler_recent_dismissal_rate=0.0,
    striker_recent_run_rate=0.0,
    bowler_recent_run_rate=0.0,
    venue="Unknown",
    venue_recent_run_rate=0.0,
):
    if balls_remaining > 84:
        innings_phase = "powerplay"
    elif balls_remaining > 30:
        innings_phase = "middle"
    else:
        innings_phase = "death"

    return pd.DataFrame(
        [
            {
                "striker": striker,
                "bowler": bowler,
                "venue": canonicalize_venue_name(venue),
                "striker_bowler": f"{striker}|{bowler}",
                "innings_context": (
                    "batting_first" if int(innings) == 1 else "chasing"
                ),
                "innings_phase": innings_phase,
                "batting_hand": batting_hand or "Unknown",
                "bowling_type": bowling_type or "Unknown",
                "season_context": season_context,
                "balls_remaining": balls_remaining,
                "current_team_score": current_team_score,
                "current_wickets_lost": wickets_lost,
                "current_run_rate": current_run_rate,
                "required_run_rate": required_run_rate,
                "striker_recent_boundary_rate": striker_recent_boundary_rate,
                "bowler_recent_boundary_rate": bowler_recent_boundary_rate,
                "bowler_recent_dismissal_rate": bowler_recent_dismissal_rate,
                "striker_recent_run_rate": striker_recent_run_rate,
                "bowler_recent_run_rate": bowler_recent_run_rate,
                "venue_recent_run_rate": venue_recent_run_rate,
            }
        ]
    )

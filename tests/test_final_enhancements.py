from src.model_monitoring import assess_model_drift
from src.plan_cache import load_cached_plan, store_cached_plan
from src.plan_export import build_plan_csv, build_plan_pdf


def test_persistent_plan_cache_round_trip(tmp_path):
    cache_path = tmp_path / "plans.pkl"
    plan = {"primary": {"sequence": ["Bowler A"]}}
    assert store_cached_plan(cache_path, ("state", 1), plan)
    assert load_cached_plan(cache_path, ("state", 1)) == plan


def test_drift_monitor_flags_latest_error_growth():
    status = assess_model_drift(
        {
            "outcome_log_loss": 0.8,
            "latest_season_outcome_log_loss": 1.0,
            "expected_runs_mae": 1.0,
            "latest_season_expected_runs_mae": 1.2,
            "latest_season": 2025,
        },
        {"latest_season": "2026"},
    )
    assert status["retrain_recommended"]
    assert len(status["reasons"]) == 3


def test_plan_exports_are_downloadable():
    rows = [
        {
            "Planned Over": 1,
            "Bowler": "Bowler A",
            "Expected Runs": 7.2,
            "Expected Wickets": 0.3,
            "Dot Ball %": 42.0,
            "Boundary in Over %": 31.0,
            "Direct Evidence": "24 avg balls (Medium)",
        }
    ]
    plan = {
        "sequence": ["Bowler A"],
        "expected_runs": 7.2,
        "expected_wickets": 0.3,
    }
    assert b"Bowler A" in build_plan_csv(rows)
    assert build_plan_pdf(plan, {"simulations": 80}, rows, "Test").startswith(
        b"%PDF-1.4"
    )

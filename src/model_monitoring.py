def assess_model_drift(metrics, feature_freshness=None):
    feature_freshness = feature_freshness or {}
    reasons = []

    heldout_loss = metrics.get("outcome_log_loss")
    latest_loss = metrics.get("latest_season_outcome_log_loss")
    if heldout_loss and latest_loss and latest_loss > heldout_loss * 1.12:
        reasons.append(
            f"Latest outcome log loss is {(latest_loss / heldout_loss - 1) * 100:.1f}% "
            "above the held-out baseline."
        )

    heldout_mae = metrics.get("expected_runs_mae")
    latest_mae = metrics.get("latest_season_expected_runs_mae")
    if heldout_mae and latest_mae and latest_mae > heldout_mae * 1.15:
        reasons.append(
            f"Latest expected-runs error is {(latest_mae / heldout_mae - 1) * 100:.1f}% "
            "above the held-out baseline."
        )

    model_season = str(metrics.get("latest_season", ""))
    warehouse_season = str(feature_freshness.get("latest_season", ""))
    if (
        model_season
        and warehouse_season
        and model_season != "Unknown"
        and warehouse_season != "Unknown"
        and model_season != warehouse_season
    ):
        reasons.append(
            f"Warehouse season {warehouse_season} is newer than model season {model_season}."
        )

    return {
        "status": "retrain" if reasons else "healthy",
        "retrain_recommended": bool(reasons),
        "reasons": reasons,
    }

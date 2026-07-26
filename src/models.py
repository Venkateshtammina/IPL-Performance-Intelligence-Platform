import os
import pickle
import sys
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import snowflake.connector
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, log_loss, brier_score_loss
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_context import canonicalize_venue_name
from src.model_components import (
    EventProbabilityBundle,
    HistoricalEventRateTransformer,
    HistoricalRunOutcomeBaseline,
    OutcomeDriftAdapter,
)

# Initialize environment variables
load_dotenv()

def train_matchup_model_from_snowflake():
    print("🚀 Initializing Cloud Data Model Training Pipeline...")
    
    # --- STEP 1: ESTABLISH SECURE OPERATIONAL HANDSHAKE ---
    try:
        ctx = snowflake.connector.connect(
            user=os.getenv("SF_USER"),
            password=os.getenv("SF_PASSWORD"),
            account=os.getenv("SF_ACCOUNT"),
            warehouse=os.getenv("SF_WAREHOUSE"),
            database=os.getenv("SF_DATABASE"),
            schema=os.getenv("SF_SCHEMA")
        )
        print("✅ Secure connection established with Snowflake.")
    except Exception as e:
        print(f"❌ Connection to Snowflake failed: {e}")
        return

    # --- STEP 2: EXTRACT ENGINEERED CLOUD FEATURE MATRIX ---
    print("📥 Downloading materialized features table from the cloud...")
    query = """
        SELECT 
            MATCH_ID, SEASON, START_DATE, INNINGS, VENUE, STRIKER, NON_STRIKER, BOWLER, BALLS_REMAINING,
            CURRENT_TEAM_SCORE, RUNS_NEEDED, CURRENT_WICKETS_LOST,
            CURRENT_RUN_RATE, REQUIRED_RUN_RATE,
            STRIKER_RECENT_BOUNDARY_RATE, BOWLER_RECENT_BOUNDARY_RATE,
            BOWLER_RECENT_DISMISSAL_RATE, STRIKER_RECENT_RUN_RATE,
            BOWLER_RECENT_RUN_RATE, VENUE_RECENT_RUN_RATE,
            RUNS_OFF_BAT, EVENT_TYPE, CHASE_WON
        FROM ANALYTICAL_MATCHUP_FEATURES
    """
    df = pd.read_sql(query, ctx)
    ctx.close()
    print(f"📦 Successfully ingested {len(df):,} engineered training vectors into memory.")

    # Standardize column naming conventions
    df.columns = df.columns.str.lower()
    raw_df = df.copy()

    # --- STEP 3: ENCODE CATEGORICAL STRINGS ---
    print("🏷️ Initializing dynamic label encoding matrices...")
    event_df = raw_df.copy()
    event_df['venue'] = event_df['venue'].map(canonicalize_venue_name)
    metadata_path = os.path.join("data", "player_metadata.csv")
    if os.path.exists(metadata_path):
        player_metadata = pd.read_csv(metadata_path).fillna("Unknown")
        batting_hand_map = dict(
            zip(player_metadata['PLAYER'], player_metadata['BATTING_HAND'])
        )
        bowling_type_map = dict(
            zip(player_metadata['PLAYER'], player_metadata['BOWLING_TYPE'])
        )
    else:
        batting_hand_map = {}
        bowling_type_map = {}
    event_df['striker_bowler'] = (
        event_df['striker'].astype(str) + '|' + event_df['bowler'].astype(str)
    )
    event_df['innings_context'] = np.where(
        event_df['innings'] == 1, 'batting_first', 'chasing'
    )
    event_df['innings_phase'] = pd.cut(
        event_df['balls_remaining'],
        bins=[-np.inf, 30, 84, np.inf],
        labels=['death', 'middle', 'powerplay'],
    ).astype(str)
    event_df['batting_hand'] = (
        event_df['striker'].map(batting_hand_map).fillna('Unknown')
    )
    event_df['bowling_type'] = (
        event_df['bowler'].map(bowling_type_map).fillna('Unknown')
    )
    latest_event_season = pd.to_numeric(
        event_df['season'], errors='coerce'
    ).max()
    event_df['season_context'] = np.where(
        pd.to_numeric(event_df['season'], errors='coerce')
        == latest_event_season,
        'current',
        'historical',
    )
    event_categorical = [
        'venue',
        'striker',
        'bowler',
        'striker_bowler',
        'innings_context',
        'innings_phase',
        'batting_hand',
        'bowling_type',
        'season_context',
    ]
    event_numeric = [
        'balls_remaining',
        'current_team_score',
        'current_wickets_lost',
        'current_run_rate',
        'required_run_rate',
        'striker_recent_boundary_rate',
        'bowler_recent_boundary_rate',
        'bowler_recent_dismissal_rate',
        'striker_recent_run_rate',
        'bowler_recent_run_rate',
        'venue_recent_run_rate',
    ]
    X = event_df[event_categorical + event_numeric].replace(
        [np.inf, -np.inf], np.nan
    )
    X[event_categorical] = X[event_categorical].fillna('Unknown')
    X[event_numeric] = X[event_numeric].fillna(0)
    y = event_df['event_type'].astype(int)
    run_target = (
        pd.to_numeric(event_df['runs_off_bat'], errors='coerce')
        .fillna(0)
        .clip(lower=0, upper=6)
        .astype(int)
        .replace(5, 6)
    )
    outcome_target = run_target.where(y != 2, 7)
    groups = event_df['match_id']

    # Keep complete matches together so neighboring deliveries cannot leak
    # between training and evaluation data.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    numeric_seasons = pd.to_numeric(
        event_df['season'], errors='coerce'
    ).fillna(0)
    latest_season = numeric_seasons.max()
    recency_weights = np.power(
        0.5, (latest_season - numeric_seasons).clip(lower=0) / 4.0
    )
    recency_rng = np.random.default_rng(42)
    retained_training_rows = (
        recency_rng.random(len(train_index))
        <= recency_weights.iloc[train_index].to_numpy()
    )
    recency_train_index = train_index[retained_training_rows]
    X_train = X.iloc[recency_train_index]
    y_train = y.iloc[recency_train_index]

    # --- STEP 4: ENSEMBLE MODEL TRAINING ---
    print("🧠 Training Random Forest Classifier across distributed compute layers...")
    def build_base_event_model():
        return Pipeline(
            [
                (
                    'historical_rates',
                    HistoricalEventRateTransformer(smoothing=30.0),
                ),
                (
                    'classifier',
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=180,
                        max_leaf_nodes=24,
                        min_samples_leaf=40,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        )

    def build_linear_event_model(class_weight=None):
        linear_preprocessor = ColumnTransformer(
            [
                (
                    'categorical',
                    OneHotEncoder(handle_unknown='ignore', min_frequency=5),
                    event_categorical,
                ),
                ('numeric', StandardScaler(), event_numeric),
            ]
        )
        return Pipeline(
            [
                ('preprocessor', linear_preprocessor),
                (
                    'classifier',
                    LogisticRegression(
                        max_iter=1000,
                        C=0.5,
                        solver='lbfgs',
                        class_weight=class_weight,
                    ),
                ),
            ]
        )

    base_event_model = build_base_event_model()
    training_groups = groups.iloc[recency_train_index].reset_index(drop=True)
    calibration_folds = list(
        GroupKFold(n_splits=5).split(
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            groups=training_groups,
        )
    )
    model = CalibratedClassifierCV(
        estimator=base_event_model,
        method='sigmoid',
        cv=calibration_folds,
    )
    model.fit(X_train, y_train)
    linear_candidate = CalibratedClassifierCV(
        estimator=build_linear_event_model(),
        method='sigmoid',
        cv=calibration_folds,
    )
    linear_candidate.fit(X_train, y_train)
    boosted_validation_probability = model.predict_proba(X_test)
    linear_validation_probability = linear_candidate.predict_proba(X_test)
    boosted_validation_loss = log_loss(
        y_test, boosted_validation_probability, labels=model.classes_
    )
    linear_validation_loss = log_loss(
        y_test,
        linear_validation_probability,
        labels=linear_candidate.classes_,
    )
    if linear_validation_loss < boosted_validation_loss:
        model = linear_candidate
        selected_event_model = "calibrated_linear"
        selected_base_factory = build_linear_event_model
    else:
        selected_event_model = "historical_rate_gradient_boosting"
        selected_base_factory = build_base_event_model
    print(
        f"Selected event model: {selected_event_model} "
        f"(linear {linear_validation_loss:.4f}, boosted "
        f"{boosted_validation_loss:.4f})"
    )
    boundary_model = CalibratedClassifierCV(
        estimator=build_linear_event_model(),
        method='sigmoid',
        cv=calibration_folds,
    )
    dismissal_model = CalibratedClassifierCV(
        estimator=build_linear_event_model(),
        method='sigmoid',
        cv=calibration_folds,
    )
    boundary_model.fit(X_train, (y_train == 1).astype(int))
    dismissal_model.fit(X_train, (y_train == 2).astype(int))
    run_training_index = recency_train_index[
        y.iloc[recency_train_index].to_numpy() != 2
    ]
    run_training_groups = groups.iloc[run_training_index].reset_index(drop=True)
    run_calibration_folds = list(
        GroupKFold(n_splits=5).split(
            X.iloc[run_training_index].reset_index(drop=True),
            run_target.iloc[run_training_index].reset_index(drop=True),
            groups=run_training_groups,
        )
    )
    run_selection_splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=73
    )
    selection_train_position, selection_validation_position = next(
        run_selection_splitter.split(
            X.iloc[run_training_index],
            run_target.iloc[run_training_index],
            groups=groups.iloc[run_training_index],
        )
    )
    selection_train_index = run_training_index[selection_train_position]
    selection_validation_index = run_training_index[
        selection_validation_position
    ]
    selection_groups = groups.iloc[
        selection_train_index
    ].reset_index(drop=True)
    selection_folds = list(
        GroupKFold(n_splits=5).split(
            X.iloc[selection_train_index].reset_index(drop=True),
            run_target.iloc[selection_train_index].reset_index(drop=True),
            groups=selection_groups,
        )
    )
    run_candidates = {}
    for candidate_name, class_weight in [
        ("calibrated_linear", None),
        ("balanced_calibrated_linear", "balanced"),
    ]:
        candidate = CalibratedClassifierCV(
            estimator=build_linear_event_model(class_weight=class_weight),
            method='sigmoid',
            cv=selection_folds,
        )
        candidate.fit(
            X.iloc[selection_train_index],
            run_target.iloc[selection_train_index],
        )
        run_candidates[candidate_name] = candidate
    historical_run_baseline = HistoricalRunOutcomeBaseline(
        smoothing=120.0
    ).fit(
        X.iloc[selection_train_index],
        run_target.iloc[selection_train_index],
    )
    run_candidates["historical_phase_fallback"] = historical_run_baseline

    run_candidate_losses = {}
    for candidate_name, candidate in run_candidates.items():
        candidate_probability = candidate.predict_proba(
            X.iloc[selection_validation_index]
        )
        run_candidate_losses[candidate_name] = float(
            log_loss(
                run_target.iloc[selection_validation_index],
                candidate_probability,
                labels=HistoricalRunOutcomeBaseline.classes_,
            )
        )
    selected_run_model = min(run_candidate_losses, key=run_candidate_losses.get)
    if selected_run_model == "historical_phase_fallback":
        run_model = HistoricalRunOutcomeBaseline(smoothing=120.0).fit(
            X.iloc[run_training_index],
            run_target.iloc[run_training_index],
        )
    else:
        selected_class_weight = (
            "balanced"
            if selected_run_model == "balanced_calibrated_linear"
            else None
        )
        run_model = CalibratedClassifierCV(
            estimator=build_linear_event_model(
                class_weight=selected_class_weight
            ),
            method='sigmoid',
            cv=run_calibration_folds,
        )
        run_model.fit(
            X.iloc[run_training_index],
            run_target.iloc[run_training_index],
        )
    model = EventProbabilityBundle(
        boundary_model, dismissal_model, run_model=run_model
    )
    selected_event_model = "separate_calibrated_binary"
    print(
        "Selected run model: "
        f"{selected_run_model} "
        + ", ".join(
            f"{name}={loss:.4f}"
            for name, loss in run_candidate_losses.items()
        )
    )
    encoders = {}
    print("🎯 Model training loop finalized.")

    # --- STEP 5: PERFORMANCE EVALUATION ---
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)
    print("\n📊 Model Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['0-3 Runs (0)', 'Boundary (1)', 'Striker Dismissal (2)']))
    print(f"Multiclass log loss: {log_loss(y_test, y_probability, labels=model.classes_):.4f}")
    for event_class, event_name in [(1, 'Boundary'), (2, 'Bowler dismissal')]:
        class_index = list(model.classes_).index(event_class)
        binary_target = (y_test.to_numpy() == event_class).astype(int)
        print(
            f"{event_name} Brier score: "
            f"{brier_score_loss(binary_target, y_probability[:, class_index]):.4f}"
        )
    outcome_test_target = outcome_target.iloc[test_index]
    outcome_probability = model.predict_outcome_proba(X_test)
    outcome_log_loss = log_loss(
        outcome_test_target,
        outcome_probability,
        labels=model.outcome_classes_,
    )
    expected_runs = outcome_probability @ model.outcome_run_values_
    outcome_runs = run_target.iloc[test_index].where(
        y.iloc[test_index] != 2, 0
    )
    expected_runs_mae = float(
        np.mean(np.abs(expected_runs - outcome_runs.to_numpy()))
    )
    training_outcome = outcome_target.iloc[recency_train_index]
    naive_counts = np.array(
        [
            (training_outcome.to_numpy() == label).sum() + 1
            for label in model.outcome_classes_
        ],
        dtype=float,
    )
    naive_probabilities = naive_counts / naive_counts.sum()
    naive_probability = np.tile(
        naive_probabilities, (len(outcome_test_target), 1)
    )
    naive_outcome_log_loss = float(
        log_loss(
            outcome_test_target,
            naive_probability,
            labels=model.outcome_classes_,
        )
    )
    naive_expected_runs = float(
        naive_probabilities @ model.outcome_run_values_
    )
    naive_expected_runs_mae = float(
        np.mean(
            np.abs(naive_expected_runs - outcome_runs.to_numpy())
        )
    )
    outcome_calibration = {}
    for outcome_index, outcome_class in enumerate(model.outcome_classes_):
        binary_target = (
            outcome_test_target.to_numpy() == outcome_class
        ).astype(int)
        predicted_probability = outcome_probability[:, outcome_index]
        outcome_calibration[str(outcome_class)] = {
            "observed_rate": float(binary_target.mean()),
            "predicted_rate": float(predicted_probability.mean()),
            "calibration_gap": float(
                abs(binary_target.mean() - predicted_probability.mean())
            ),
            "brier": float(
                brier_score_loss(binary_target, predicted_probability)
            ),
        }
    absolute_run_error = np.abs(
        expected_runs - outcome_runs.to_numpy()
    )
    phase_expected_runs_mae = {
        str(phase): float(absolute_run_error[phase_indices].mean())
        for phase, phase_indices in event_df.iloc[test_index]
        .reset_index(drop=True)
        .groupby("innings_phase")
        .indices.items()
    }
    print(f"Granular outcome log loss: {outcome_log_loss:.4f}")
    print(f"Expected runs per ball MAE: {expected_runs_mae:.4f}")
    print(
        f"Naive historical baseline: log loss "
        f"{naive_outcome_log_loss:.4f}, expected-runs MAE "
        f"{naive_expected_runs_mae:.4f}"
    )
    print(f"Expected-runs MAE by phase: {phase_expected_runs_mae}")

    latest_test_index = np.flatnonzero(
        numeric_seasons.to_numpy() == latest_season
    )
    latest_train_candidates = np.flatnonzero(
        numeric_seasons.to_numpy() < latest_season
    )
    latest_retained = (
        recency_rng.random(len(latest_train_candidates))
        <= recency_weights.iloc[latest_train_candidates].to_numpy()
    )
    latest_train_index = latest_train_candidates[latest_retained]
    latest_groups = groups.iloc[latest_train_index].reset_index(drop=True)
    latest_folds = list(
        GroupKFold(n_splits=3).split(
            X.iloc[latest_train_index].reset_index(drop=True),
            y.iloc[latest_train_index].reset_index(drop=True),
            groups=latest_groups,
        )
    )
    latest_boundary_model = CalibratedClassifierCV(
        estimator=build_linear_event_model(),
        method='sigmoid',
        cv=latest_folds,
    )
    latest_dismissal_model = CalibratedClassifierCV(
        estimator=build_linear_event_model(),
        method='sigmoid',
        cv=latest_folds,
    )
    latest_boundary_model.fit(
        X.iloc[latest_train_index],
        (y.iloc[latest_train_index] == 1).astype(int),
    )
    latest_dismissal_model.fit(
        X.iloc[latest_train_index],
        (y.iloc[latest_train_index] == 2).astype(int),
    )
    latest_run_train_index = latest_train_index[
        y.iloc[latest_train_index].to_numpy() != 2
    ]
    latest_run_groups = groups.iloc[
        latest_run_train_index
    ].reset_index(drop=True)
    latest_run_folds = list(
        GroupKFold(n_splits=3).split(
            X.iloc[latest_run_train_index].reset_index(drop=True),
            run_target.iloc[latest_run_train_index].reset_index(drop=True),
            groups=latest_run_groups,
        )
    )
    if selected_run_model == "historical_phase_fallback":
        latest_run_model = HistoricalRunOutcomeBaseline(
            smoothing=120.0
        ).fit(
            X.iloc[latest_run_train_index],
            run_target.iloc[latest_run_train_index],
        )
    else:
        latest_class_weight = (
            "balanced"
            if selected_run_model == "balanced_calibrated_linear"
            else None
        )
        latest_run_model = CalibratedClassifierCV(
            estimator=build_linear_event_model(
                class_weight=latest_class_weight
            ),
            method='sigmoid',
            cv=latest_run_folds,
        )
        latest_run_model.fit(
            X.iloc[latest_run_train_index],
            run_target.iloc[latest_run_train_index],
        )
    latest_season_model = EventProbabilityBundle(
        latest_boundary_model,
        latest_dismissal_model,
        run_model=latest_run_model,
    )
    latest_probability = latest_season_model.predict_proba(
        X.iloc[latest_test_index]
    )
    latest_target = y.iloc[latest_test_index]
    latest_event_log_loss = log_loss(
        latest_target,
        latest_probability,
        labels=latest_season_model.classes_,
    )
    print(
        f"Latest-season-only event log loss ({int(latest_season)}): "
        f"{latest_event_log_loss:.4f}"
    )
    latest_outcome_probability = (
        latest_season_model.predict_outcome_proba(
            X.iloc[latest_test_index]
        )
    )
    latest_outcome_target = outcome_target.iloc[latest_test_index]
    latest_outcome_log_loss = float(
        log_loss(
            latest_outcome_target,
            latest_outcome_probability,
            labels=latest_season_model.outcome_classes_,
        )
    )
    latest_expected_runs = (
        latest_outcome_probability
        @ latest_season_model.outcome_run_values_
    )
    latest_observed_runs = run_target.iloc[latest_test_index].where(
        y.iloc[latest_test_index] != 2, 0
    )
    latest_expected_runs_mae = float(
        np.mean(
            np.abs(
                latest_expected_runs - latest_observed_runs.to_numpy()
            )
        )
    )
    print(
        f"Latest-season granular log loss: "
        f"{latest_outcome_log_loss:.4f}; expected-runs MAE "
        f"{latest_expected_runs_mae:.4f}"
    )
    latest_dates = pd.to_datetime(
        event_df.iloc[latest_test_index]['start_date'], errors='coerce'
    )
    chronological_order = np.argsort(latest_dates.fillna(pd.Timestamp.min))
    ordered_outcome_probability = latest_outcome_probability[
        chronological_order
    ].copy()
    ordered_outcome_target = latest_outcome_target.iloc[
        chronological_order
    ].to_numpy()
    ordered_outcome_features = X.iloc[latest_test_index].iloc[
        chronological_order
    ].reset_index(drop=True)
    rolling_outcome_probability = ordered_outcome_probability.copy()
    outcome_block_edges = np.linspace(
        0, len(ordered_outcome_target), num=5, dtype=int
    )
    for block_number in range(1, 4):
        history_end = outcome_block_edges[block_number]
        block_start = outcome_block_edges[block_number]
        block_end = outcome_block_edges[block_number + 1]
        rolling_adapter = OutcomeDriftAdapter(smoothing=120.0).fit(
            ordered_outcome_features.iloc[:history_end],
            ordered_outcome_target[:history_end],
            ordered_outcome_probability[:history_end],
            latest_season_model.outcome_classes_,
        )
        rolling_outcome_probability[block_start:block_end] = (
            rolling_adapter.transform(
                ordered_outcome_features.iloc[block_start:block_end],
                ordered_outcome_probability[block_start:block_end],
            )
        )
    rolling_outcome_log_loss = float(
        log_loss(
            ordered_outcome_target,
            rolling_outcome_probability,
            labels=latest_season_model.outcome_classes_,
        )
    )
    rolling_expected_runs = (
        rolling_outcome_probability
        @ latest_season_model.outcome_run_values_
    )
    ordered_observed_runs = latest_observed_runs.iloc[
        chronological_order
    ].to_numpy()
    rolling_expected_runs_mae = float(
        np.mean(np.abs(rolling_expected_runs - ordered_observed_runs))
    )
    drift_adapter_deployed = (
        rolling_outcome_log_loss < latest_outcome_log_loss
    )
    deployment_adapter = OutcomeDriftAdapter(smoothing=120.0).fit(
        X.iloc[latest_test_index].reset_index(drop=True),
        latest_outcome_target.to_numpy(),
        latest_outcome_probability,
        latest_season_model.outcome_classes_,
    )
    if drift_adapter_deployed:
        model.outcome_adapter = deployment_adapter
    drift_multipliers = {
        phase: {
            str(outcome_class): float(multiplier)
            for outcome_class, multiplier in zip(
                latest_season_model.outcome_classes_, multipliers
            )
        }
        for phase, multipliers in deployment_adapter.phase_multipliers_.items()
    }
    print(
        f"Rolling granular recalibration: log loss "
        f"{rolling_outcome_log_loss:.4f}, expected-runs MAE "
        f"{rolling_expected_runs_mae:.4f}; deployed "
        f"{drift_adapter_deployed}"
    )
    ordered_probability = latest_probability[chronological_order].copy()
    ordered_target = latest_target.iloc[chronological_order].to_numpy()
    rolling_probability = ordered_probability.copy()
    block_edges = np.linspace(
        0, len(ordered_target), num=5, dtype=int
    )
    for block_number in range(1, 4):
        history_end = block_edges[block_number]
        block_start = block_edges[block_number]
        block_end = block_edges[block_number + 1]
        for event_class in (1, 2):
            observed_rate = np.mean(
                ordered_target[:history_end] == event_class
            )
            predicted_rate = ordered_probability[
                :history_end, event_class
            ].mean()
            observed_odds = observed_rate / max(1.0 - observed_rate, 1e-6)
            predicted_odds = predicted_rate / max(
                1.0 - predicted_rate, 1e-6
            )
            odds_multiplier = observed_odds / max(predicted_odds, 1e-6)
            block_probability = rolling_probability[
                block_start:block_end, event_class
            ]
            block_odds = block_probability / np.maximum(
                1.0 - block_probability, 1e-6
            )
            rolling_probability[
                block_start:block_end, event_class
            ] = (block_odds * odds_multiplier) / (
                1.0 + block_odds * odds_multiplier
            )
        event_sum = rolling_probability[
            block_start:block_end, 1:
        ].sum(axis=1)
        overflow = event_sum > 0.98
        if overflow.any():
            block_events = rolling_probability[
                block_start:block_end, 1:
            ].copy()
            block_events[overflow] *= (
                0.98 / event_sum[overflow]
            )[:, None]
            rolling_probability[
                block_start:block_end, 1:
            ] = block_events
        rolling_probability[
            block_start:block_end, 0
        ] = 1.0 - rolling_probability[
            block_start:block_end, 1:
        ].sum(axis=1)
    rolling_latest_log_loss = log_loss(
        ordered_target,
        rolling_probability,
        labels=[0, 1, 2],
    )
    print(
        f"Rolling latest-season calibrated log loss: "
        f"{rolling_latest_log_loss:.4f}"
    )

    # Train a separate, interpretable chase model. Player-pressure interaction
    # features let the data learn that the same batter may perform differently
    # in high required-rate and low-wicket situations.
    win_df = raw_df[raw_df['innings'] == 2].copy()
    win_df['venue'] = win_df['venue'].map(canonicalize_venue_name)
    win_df['pressure_band'] = pd.cut(
        win_df['required_run_rate'],
        bins=[-np.inf, 7, 9, 12, np.inf],
        labels=['low', 'medium', 'high', 'extreme'],
    ).astype(str)
    win_df['wicket_band'] = pd.cut(
        win_df['current_wickets_lost'],
        bins=[-np.inf, 2, 5, 7, np.inf],
        labels=['healthy', 'under_pressure', 'tail_exposed', 'last_pair'],
    ).astype(str)
    win_df['striker_pressure'] = (
        win_df['striker'].astype(str)
        + '|'
        + win_df['pressure_band']
        + '|'
        + win_df['wicket_band']
    )
    win_df['non_striker_pressure'] = (
        win_df['non_striker'].astype(str)
        + '|'
        + win_df['pressure_band']
        + '|'
        + win_df['wicket_band']
    )
    safe_balls = win_df['balls_remaining'].clip(lower=1)
    win_df['runs_per_ball_needed'] = win_df['runs_needed'] / safe_balls
    win_df['equation_band'] = pd.cut(
        win_df['runs_per_ball_needed'],
        bins=[-np.inf, 1.5, 2.5, 4, 6, np.inf],
        labels=[
            'manageable',
            'severe',
            'desperate',
            'near_impossible',
            'beyond_sixes',
        ],
    ).astype(str)
    win_df['required_run_rate_squared'] = win_df['required_run_rate'] ** 2
    win_df['runs_above_six_rate'] = (
        win_df['runs_needed'] - 6 * win_df['balls_remaining']
    ).clip(lower=0)

    win_categorical = [
        'venue',
        'striker',
        'non_striker',
        'striker_pressure',
        'non_striker_pressure',
        'equation_band',
    ]
    win_numeric = [
        'balls_remaining',
        'current_team_score',
        'runs_needed',
        'current_wickets_lost',
        'current_run_rate',
        'required_run_rate',
        'runs_per_ball_needed',
        'required_run_rate_squared',
        'runs_above_six_rate',
    ]
    win_features = win_df[win_categorical + win_numeric].replace(
        [np.inf, -np.inf], np.nan
    )
    win_features[win_categorical] = win_features[win_categorical].fillna('Unknown')
    win_features[win_numeric] = win_features[win_numeric].fillna(0)
    win_target = win_df['chase_won'].astype(int)
    win_groups = win_df['match_id']

    win_train_index, win_test_index = next(
        splitter.split(win_features, win_target, groups=win_groups)
    )
    win_preprocessor = ColumnTransformer(
        [
            (
                'categorical',
                OneHotEncoder(handle_unknown='ignore', min_frequency=5),
                win_categorical,
            ),
            ('numeric', StandardScaler(), win_numeric),
        ]
    )
    win_model = Pipeline(
        [
            ('preprocessor', win_preprocessor),
            (
                'classifier',
                LogisticRegression(max_iter=1000, C=0.5, solver='liblinear'),
            ),
        ]
    )
    match_row_counts = win_df.groupby('match_id')['match_id'].transform('count')
    win_seasons = pd.to_numeric(win_df['season'], errors='coerce').fillna(0)
    win_latest_season = win_seasons.max()
    win_recency_weights = np.power(
        0.5, (win_latest_season - win_seasons).clip(lower=0) / 4.0
    )
    sample_weights = win_recency_weights / match_row_counts
    win_model.fit(
        win_features.iloc[win_train_index],
        win_target.iloc[win_train_index],
        classifier__sample_weight=sample_weights.iloc[win_train_index],
    )
    win_test_probability = win_model.predict_proba(
        win_features.iloc[win_test_index]
    )[:, 1]
    print(
        f"Chase model log loss: "
        f"{log_loss(win_target.iloc[win_test_index], win_test_probability):.4f}"
    )
    print(
        f"Chase model Brier score: "
        f"{brier_score_loss(win_target.iloc[win_test_index], win_test_probability):.4f}"
    )

    # --- STEP 6: SERIALIZE MACHINE LEARNING OBJECTS ---
    processed_dir = os.path.join("data", "2_processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    model_path = os.path.join(processed_dir, "matchup_model.pkl")
    encoder_path = os.path.join(processed_dir, "encoders.pkl")
    win_model_path = os.path.join(processed_dir, "win_probability_model.pkl")
    metrics_path = os.path.join(processed_dir, "model_metrics.pkl")

    print("💾 Serializing model artifacts locally...")
    with open(model_path, "wb") as m_f:
        pickle.dump(model, m_f)
    with open(encoder_path, "wb") as e_f:
        pickle.dump(encoders, e_f)
    with open(win_model_path, "wb") as w_f:
        pickle.dump(win_model, w_f)
    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(raw_df)),
        "training_latest_match_date": str(
            pd.to_datetime(
                raw_df["start_date"], errors="coerce"
            ).max().date()
        ),
        "event_log_loss": float(
            log_loss(y_test, y_probability, labels=model.classes_)
        ),
        "boundary_brier": float(
            brier_score_loss(
                (y_test.to_numpy() == 1).astype(int),
                y_probability[:, list(model.classes_).index(1)],
            )
        ),
        "dismissal_brier": float(
            brier_score_loss(
                (y_test.to_numpy() == 2).astype(int),
                y_probability[:, list(model.classes_).index(2)],
            )
        ),
        "outcome_log_loss": float(outcome_log_loss),
        "expected_runs_mae": expected_runs_mae,
        "naive_outcome_log_loss": naive_outcome_log_loss,
        "naive_expected_runs_mae": naive_expected_runs_mae,
        "selected_run_model": selected_run_model,
        "run_candidate_log_losses": run_candidate_losses,
        "outcome_calibration": outcome_calibration,
        "phase_expected_runs_mae": phase_expected_runs_mae,
        "win_log_loss": float(
            log_loss(win_target.iloc[win_test_index], win_test_probability)
        ),
        "win_brier": float(
            brier_score_loss(
                win_target.iloc[win_test_index], win_test_probability
            )
        ),
        "latest_season": int(latest_season),
        "selected_event_model": selected_event_model,
        "linear_candidate_log_loss": float(linear_validation_loss),
        "boosted_candidate_log_loss": float(boosted_validation_loss),
        "latest_season_event_log_loss": float(latest_event_log_loss),
        "latest_season_outcome_log_loss": latest_outcome_log_loss,
        "latest_season_expected_runs_mae": latest_expected_runs_mae,
        "rolling_outcome_log_loss": rolling_outcome_log_loss,
        "rolling_expected_runs_mae": rolling_expected_runs_mae,
        "drift_adapter_deployed": drift_adapter_deployed,
        "drift_multipliers": drift_multipliers,
        "rolling_latest_season_log_loss": float(
            rolling_latest_log_loss
        ),
        "latest_season_test_deliveries": int(len(latest_target)),
        "event_test_deliveries": int(len(y_test)),
    }
    with open(metrics_path, "wb") as metrics_file:
        pickle.dump(metrics, metrics_file)
        
    print(f"🎉 Success! Machine learning model assets saved safely:\n 🔹 {model_path}\n 🔹 {encoder_path}")
    print("🏆 Step 3 Production Model Training Pipeline Complete!")

if __name__ == "__main__":
    train_matchup_model_from_snowflake();

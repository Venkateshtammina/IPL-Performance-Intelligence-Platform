"""Serializable estimators and probability adapters."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HistoricalEventRateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=30.0):
        self.smoothing = smoothing

    def fit(self, features, target):
        frame = pd.DataFrame(features).reset_index(drop=True).copy()
        target_values = np.asarray(target)
        frame["_boundary_target"] = (target_values == 1).astype(float)
        frame["_dismissal_target"] = (target_values == 2).astype(float)
        self.global_boundary_rate_ = float(frame["_boundary_target"].mean())
        self.global_dismissal_rate_ = float(frame["_dismissal_target"].mean())
        self.group_columns_ = [
            "striker",
            "bowler",
            "striker_bowler",
            "innings_context",
            "innings_phase",
            "batting_hand",
            "bowling_type",
        ]
        self.rate_maps_ = {}
        for column in self.group_columns_:
            grouped = frame.groupby(column, dropna=False).agg(
                count=("_boundary_target", "size"),
                boundary_events=("_boundary_target", "sum"),
                dismissal_events=("_dismissal_target", "sum"),
            )
            grouped["boundary_rate"] = (
                grouped["boundary_events"]
                + self.smoothing * self.global_boundary_rate_
            ) / (grouped["count"] + self.smoothing)
            grouped["dismissal_rate"] = (
                grouped["dismissal_events"]
                + self.smoothing * self.global_dismissal_rate_
            ) / (grouped["count"] + self.smoothing)
            self.rate_maps_[column] = grouped[
                ["boundary_rate", "dismissal_rate"]
            ]
        return self

    def transform(self, features):
        frame = pd.DataFrame(features).reset_index(drop=True).copy()
        numeric_columns = [
            "balls_remaining",
            "current_team_score",
            "current_wickets_lost",
            "current_run_rate",
            "required_run_rate",
        ]
        transformed = frame[numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0)
        for column in self.group_columns_:
            rates = self.rate_maps_[column]
            transformed[f"{column}_boundary_rate"] = (
                frame[column]
                .map(rates["boundary_rate"])
                .fillna(self.global_boundary_rate_)
            )
            transformed[f"{column}_dismissal_rate"] = (
                frame[column]
                .map(rates["dismissal_rate"])
                .fillna(self.global_dismissal_rate_)
            )
        return transformed.to_numpy(dtype=float)


class HistoricalRunOutcomeBaseline(BaseEstimator):
    classes_ = np.array([0, 1, 2, 3, 4, 6])

    def __init__(self, smoothing=120.0):
        self.smoothing = smoothing

    def fit(self, features, target):
        frame = pd.DataFrame(features).reset_index(drop=True)
        target_values = pd.Series(target).reset_index(drop=True).astype(int)
        global_counts = target_values.value_counts()
        self.global_probabilities_ = np.array(
            [global_counts.get(label, 0) + 1 for label in self.classes_],
            dtype=float,
        )
        self.global_probabilities_ /= self.global_probabilities_.sum()
        self.phase_probabilities_ = {}
        grouped_indices = frame.groupby(
            ["innings_context", "innings_phase"], dropna=False
        ).indices
        for key, indices in grouped_indices.items():
            group_counts = target_values.iloc[indices].value_counts()
            probabilities = np.array(
                [group_counts.get(label, 0) for label in self.classes_],
                dtype=float,
            )
            probabilities += self.smoothing * self.global_probabilities_
            probabilities /= probabilities.sum()
            self.phase_probabilities_[key] = probabilities
        return self

    def predict_proba(self, features):
        frame = pd.DataFrame(features).reset_index(drop=True)
        return np.vstack(
            [
                self.phase_probabilities_.get(
                    (row["innings_context"], row["innings_phase"]),
                    self.global_probabilities_,
                )
                for _, row in frame.iterrows()
            ]
        )

    def predict(self, features):
        probabilities = self.predict_proba(features)
        return self.classes_[np.argmax(probabilities, axis=1)]


class OutcomeDriftAdapter:
    def __init__(self, smoothing=120.0, min_multiplier=0.5, max_multiplier=2.0):
        self.smoothing = smoothing
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

    def fit(self, features, target, predicted_probabilities, classes):
        frame = pd.DataFrame(features).reset_index(drop=True)
        target_values = np.asarray(target)
        probabilities = np.asarray(predicted_probabilities, dtype=float)
        self.classes_ = np.asarray(classes)
        self.phase_multipliers_ = {}
        for phase, indices in frame.groupby("innings_phase").indices.items():
            phase_probability = probabilities[indices]
            phase_target = target_values[indices]
            multipliers = []
            for class_index, outcome_class in enumerate(self.classes_):
                predicted_rate = float(
                    phase_probability[:, class_index].mean()
                )
                observed_events = float(
                    (phase_target == outcome_class).sum()
                )
                adjusted_observed_rate = (
                    observed_events + self.smoothing * predicted_rate
                ) / (len(indices) + self.smoothing)
                predicted_odds = predicted_rate / max(
                    1.0 - predicted_rate, 1e-6
                )
                observed_odds = adjusted_observed_rate / max(
                    1.0 - adjusted_observed_rate, 1e-6
                )
                multiplier = observed_odds / max(predicted_odds, 1e-6)
                multipliers.append(
                    float(
                        np.clip(
                            multiplier,
                            self.min_multiplier,
                            self.max_multiplier,
                        )
                    )
                )
            self.phase_multipliers_[str(phase)] = np.asarray(multipliers)
        return self

    def transform(self, features, predicted_probabilities):
        frame = pd.DataFrame(features).reset_index(drop=True)
        adjusted = np.asarray(predicted_probabilities, dtype=float).copy()
        for phase, indices in frame.groupby("innings_phase").indices.items():
            multipliers = self.phase_multipliers_.get(str(phase))
            if multipliers is not None:
                adjusted[indices] *= multipliers
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        return adjusted


class EventProbabilityBundle:
    classes_ = np.array([0, 1, 2])
    outcome_classes_ = np.array([0, 1, 2, 3, 4, 6, 7])
    outcome_run_values_ = np.array([0, 1, 2, 3, 4, 6, 0], dtype=float)

    def __init__(
        self,
        boundary_model,
        dismissal_model,
        run_model=None,
        outcome_adapter=None,
    ):
        self.boundary_model = boundary_model
        self.dismissal_model = dismissal_model
        self.run_model = run_model
        self.outcome_adapter = outcome_adapter

    def predict_proba(self, features):
        boundary_classes = list(self.boundary_model.classes_)
        dismissal_classes = list(self.dismissal_model.classes_)
        boundary_probability = self.boundary_model.predict_proba(features)[
            :, boundary_classes.index(1)
        ].copy()
        dismissal_probability = self.dismissal_model.predict_proba(features)[
            :, dismissal_classes.index(1)
        ].copy()
        combined_events = boundary_probability + dismissal_probability
        overflow = combined_events > 0.98
        if overflow.any():
            scale = 0.98 / combined_events[overflow]
            boundary_probability[overflow] *= scale
            dismissal_probability[overflow] *= scale
        safe_probability = 1.0 - boundary_probability - dismissal_probability
        return np.column_stack(
            [safe_probability, boundary_probability, dismissal_probability]
        )

    def predict_outcome_proba(self, features):
        if self.run_model is None:
            raise AttributeError(
                "This artifact predates granular delivery-outcome training."
            )

        event_probabilities = self.predict_proba(features)
        safe_probability = event_probabilities[:, 0]
        boundary_probability = event_probabilities[:, 1]
        dismissal_probability = event_probabilities[:, 2]
        raw_run_probability = self.run_model.predict_proba(features)
        run_class_index = {
            int(label): index
            for index, label in enumerate(self.run_model.classes_)
        }
        run_probability = np.column_stack(
            [
                raw_run_probability[:, run_class_index[label]]
                if label in run_class_index
                else np.zeros(len(features))
                for label in (0, 1, 2, 3, 4, 6)
            ]
        )

        safe_weights = run_probability[:, :4]
        safe_weight_sum = safe_weights.sum(axis=1, keepdims=True)
        safe_fallback = np.array([0.48, 0.38, 0.11, 0.03])
        safe_weights = np.divide(
            safe_weights,
            safe_weight_sum,
            out=np.tile(safe_fallback, (len(features), 1)),
            where=safe_weight_sum > 0,
        )

        boundary_weights = run_probability[:, 4:]
        boundary_weight_sum = boundary_weights.sum(axis=1, keepdims=True)
        boundary_fallback = np.array([0.82, 0.18])
        boundary_weights = np.divide(
            boundary_weights,
            boundary_weight_sum,
            out=np.tile(boundary_fallback, (len(features), 1)),
            where=boundary_weight_sum > 0,
        )

        probabilities = np.column_stack(
            [
                safe_weights * safe_probability[:, None],
                boundary_weights * boundary_probability[:, None],
                dismissal_probability,
            ]
        )
        outcome_adapter = getattr(self, "outcome_adapter", None)
        if outcome_adapter is not None:
            probabilities = outcome_adapter.transform(features, probabilities)
        return probabilities

    def predict(self, features):
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

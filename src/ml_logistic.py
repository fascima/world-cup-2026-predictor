"""Multinomial logistic regression baseline for match outcomes."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss

from src.probability_adjustment import (
    WorldCupAdjustmentParams,
    apply_world_cup_adjustment,
    tune_world_cup_adjustment,
    validation_folds,
)


TARGET_COLUMN = "target"
TRAIN_TEST_SPLIT_DATE = pd.Timestamp("2022-01-01")
VALIDATION_SPLIT_DATE = pd.Timestamp("2020-01-01")
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = {
    0: "team_a_win",
    1: "draw",
    2: "team_b_win",
}
METADATA_COLUMNS = {
    "date",
    "team_a",
    "team_b",
    "tournament",
    "team_a_score",
    "team_b_score",
    TARGET_COLUMN,
}
CATEGORICAL_FEATURES = ["tournament_type"]
DISABLED_MODEL_FEATURES = {
    "has_market_values",
    "team_a_market_value_eur",
    "team_b_market_value_eur",
    "market_value_baseline_eur",
    "market_value_log_ratio",
    "market_value_adjustment_diff",
    "team_a_effective_market_value_eur",
    "team_b_effective_market_value_eur",
    "effective_market_value_log_ratio",
    "effective_market_value_adjustment_diff",
    "effective_market_value_loss_diff",
    "effective_market_value_loss_share_diff",
}
C_VALUES = [0.05, 0.1, 0.25]
CLASS_WEIGHT_OPTIONS: list[dict[int, float] | str | None] = [
    None,
    {0: 1.0, 1: 1.25, 2: 1.0},
]
DRAW_PROBABILITY_MULTIPLIERS = [0.85, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
DRAW_DECISION_THRESHOLDS = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32]
DRAW_DECISION_MAX_WIN_PROB_GAPS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
MIN_DRAW_DECISION_VALIDATION_ACCURACY_GAIN = 0.002
USE_FLIPPED_TRAINING_ROWS = True
FLIPPED_DECISIVE_TARGET_AS_DRAW = False

SWAP_COLUMN_PAIRS = [
    ("team_a", "team_b"),
    ("team_a_pre_elo", "team_b_pre_elo"),
    ("team_a_prediction_elo", "team_b_prediction_elo"),
    ("team_a_recent_win_rate_5", "team_b_recent_win_rate_5"),
    ("team_a_recent_win_rate_10", "team_b_recent_win_rate_10"),
    ("team_a_avg_goals_for_last_5", "team_b_avg_goals_for_last_5"),
    ("team_a_avg_goals_for_last_10", "team_b_avg_goals_for_last_10"),
    ("team_a_avg_goals_against_last_5", "team_b_avg_goals_against_last_5"),
    ("team_a_avg_goals_against_last_10", "team_b_avg_goals_against_last_10"),
    ("team_a_attack_vs_team_b_defense_last_5", "team_b_attack_vs_team_a_defense_last_5"),
    ("team_a_attack_vs_team_b_defense_last_10", "team_b_attack_vs_team_a_defense_last_10"),
    ("team_a_avg_points_last_5", "team_b_avg_points_last_5"),
    ("team_a_avg_points_last_10", "team_b_avg_points_last_10"),
    ("team_a_group_matches_played", "team_b_group_matches_played"),
    ("team_a_group_points_before", "team_b_group_points_before"),
    ("team_a_group_goal_diff_before", "team_b_group_goal_diff_before"),
    ("team_a_likely_qualified", "team_b_likely_qualified"),
    ("team_a_must_win", "team_b_must_win"),
    ("team_a_rotation_risk", "team_b_rotation_risk"),
    ("team_a_opponent_rotation_opportunity", "team_b_opponent_rotation_opportunity"),
    ("team_a_qualified_vs_team_b_must_win", "team_b_qualified_vs_team_a_must_win"),
    ("team_a_statsbomb_matches_before", "team_b_statsbomb_matches_before"),
    ("team_a_has_statsbomb_features", "team_b_has_statsbomb_features"),
    ("team_a_statsbomb_xg_for_last_3", "team_b_statsbomb_xg_for_last_3"),
    ("team_a_statsbomb_xg_against_last_3", "team_b_statsbomb_xg_against_last_3"),
    ("team_a_statsbomb_xg_diff_last_3", "team_b_statsbomb_xg_diff_last_3"),
    ("team_a_statsbomb_xg_for_last_5", "team_b_statsbomb_xg_for_last_5"),
    ("team_a_statsbomb_xg_against_last_5", "team_b_statsbomb_xg_against_last_5"),
    ("team_a_statsbomb_xg_diff_last_5", "team_b_statsbomb_xg_diff_last_5"),
    ("team_a_market_value_eur", "team_b_market_value_eur"),
    ("team_a_effective_market_value_eur", "team_b_effective_market_value_eur"),
    ("elo_team_a_win_prob", "elo_team_b_win_prob"),
    ("poisson_team_a_win_prob", "poisson_team_b_win_prob"),
    ("dixon_coles_team_a_win_prob", "dixon_coles_team_b_win_prob"),
    ("team_a_home_advantage", "team_b_home_advantage"),
    ("team_a_injured_players_count", "team_b_injured_players_count"),
    ("team_a_injured_market_value_eur", "team_b_injured_market_value_eur"),
    ("team_a_injured_market_value_share", "team_b_injured_market_value_share"),
    ("team_a_key_injured_players_count", "team_b_key_injured_players_count"),
    ("team_a_max_injured_player_market_value_eur", "team_b_max_injured_player_market_value_eur"),
    ("team_a_injured_gk_count", "team_b_injured_gk_count"),
    ("team_a_injured_df_count", "team_b_injured_df_count"),
    ("team_a_injured_mf_count", "team_b_injured_mf_count"),
    ("team_a_injured_fw_count", "team_b_injured_fw_count"),
    ("team_a_injured_gk_market_value_eur", "team_b_injured_gk_market_value_eur"),
    ("team_a_injured_df_market_value_eur", "team_b_injured_df_market_value_eur"),
    ("team_a_injured_mf_market_value_eur", "team_b_injured_mf_market_value_eur"),
    ("team_a_injured_fw_market_value_eur", "team_b_injured_fw_market_value_eur"),
    ("team_a_wc_prior_matches", "team_b_wc_prior_matches"),
    ("team_a_wc_prior_weight", "team_b_wc_prior_weight"),
    ("team_a_wc_prior_points_per_match", "team_b_wc_prior_points_per_match"),
    ("team_a_wc_prior_goal_diff_per_match", "team_b_wc_prior_goal_diff_per_match"),
    ("team_a_wc_prior_win_rate", "team_b_wc_prior_win_rate"),
    ("team_a_wc_prior_draw_rate", "team_b_wc_prior_draw_rate"),
    ("team_a_wc_prior_knockout_matches", "team_b_wc_prior_knockout_matches"),
]
SIGNED_DIFFERENCE_COLUMNS = [
    "elo_diff",
    "prediction_elo_diff",
    "adjusted_elo_diff",
    "recent_win_rate_diff_5",
    "recent_win_rate_diff_10",
    "avg_goals_for_diff_5",
    "avg_goals_for_diff_10",
    "avg_goals_against_diff_5",
    "avg_goals_against_diff_10",
    "attack_defense_pressure_diff_5",
    "attack_defense_pressure_diff_10",
    "avg_points_diff_5",
    "avg_points_diff_10",
    "group_points_diff_before",
    "group_goal_diff_diff_before",
    "motivation_diff",
    "statsbomb_xg_diff_delta_last_3",
    "statsbomb_non_penalty_xg_diff_delta_last_3",
    "statsbomb_shot_diff_delta_last_3",
    "statsbomb_box_entry_diff_delta_last_3",
    "statsbomb_pressure_diff_delta_last_3",
    "statsbomb_xg_diff_delta_last_5",
    "statsbomb_non_penalty_xg_diff_delta_last_5",
    "statsbomb_shot_diff_delta_last_5",
    "statsbomb_box_entry_diff_delta_last_5",
    "statsbomb_pressure_diff_delta_last_5",
    "market_value_log_ratio",
    "market_value_adjustment_diff",
    "effective_market_value_log_ratio",
    "effective_market_value_adjustment_diff",
    "effective_market_value_loss_diff",
    "effective_market_value_loss_share_diff",
    "elo_diff_neutral_interaction",
    "elo_diff_home_interaction",
    "injured_players_count_diff",
    "injured_market_value_diff",
    "injured_market_value_share_diff",
    "key_injured_players_count_diff",
    "max_injured_player_market_value_diff",
    "injured_gk_count_diff",
    "injured_df_count_diff",
    "injured_mf_count_diff",
    "injured_fw_count_diff",
    "injured_gk_market_value_diff",
    "injured_df_market_value_diff",
    "injured_mf_market_value_diff",
    "injured_fw_market_value_diff",
    "wc_prior_matches_diff",
    "wc_prior_weight_diff",
    "wc_prior_points_per_match_diff",
    "wc_prior_goal_diff_per_match_diff",
    "wc_prior_win_rate_diff",
    "wc_prior_draw_rate_diff",
    "wc_prior_knockout_matches_diff",
]


def _available_feature_columns(features_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature columns available for training."""
    categorical_columns = [
        column for column in CATEGORICAL_FEATURES if column in features_df.columns
    ]
    excluded_columns = METADATA_COLUMNS | set(categorical_columns) | DISABLED_MODEL_FEATURES
    candidate_columns = [
        column
        for column in features_df.columns
        if column not in excluded_columns and "statsbomb" not in column
    ]
    numeric_columns = [
        column
        for column in candidate_columns
        if pd.api.types.is_numeric_dtype(features_df[column])
        or pd.api.types.is_bool_dtype(features_df[column])
    ]
    return numeric_columns, categorical_columns


def _class_weight_label(class_weight: dict[int, float] | str | None) -> str:
    """Return a readable label for a class-weight option."""
    if class_weight is None:
        return "none"
    if isinstance(class_weight, str):
        return class_weight
    return ";".join(f"{key}:{value:g}" for key, value in sorted(class_weight.items()))


def _augment_with_flipped_rows(features_df: pd.DataFrame) -> pd.DataFrame:
    """Return features with added team-swapped rows for model fitting only."""
    if not USE_FLIPPED_TRAINING_ROWS or features_df.empty:
        return features_df

    flipped = features_df.copy()
    for left_column, right_column in SWAP_COLUMN_PAIRS:
        if left_column in flipped.columns and right_column in flipped.columns:
            flipped[left_column], flipped[right_column] = (
                flipped[right_column].copy(),
                flipped[left_column].copy(),
            )

    for column in SIGNED_DIFFERENCE_COLUMNS:
        if column in flipped.columns:
            flipped[column] = -pd.to_numeric(flipped[column], errors="coerce")

    if "abs_adjusted_elo_diff" in flipped.columns and "adjusted_elo_diff" in flipped.columns:
        flipped["abs_adjusted_elo_diff"] = pd.to_numeric(
            flipped["adjusted_elo_diff"],
            errors="coerce",
        ).abs()
    if "close_elo_gap_50" in flipped.columns and "abs_adjusted_elo_diff" in flipped.columns:
        flipped["close_elo_gap_50"] = (flipped["abs_adjusted_elo_diff"] < 50.0).astype(float)
    if "close_elo_gap_100" in flipped.columns and "abs_adjusted_elo_diff" in flipped.columns:
        flipped["close_elo_gap_100"] = (flipped["abs_adjusted_elo_diff"] < 100.0).astype(float)
    if "close_elo_gap_150" in flipped.columns and "abs_adjusted_elo_diff" in flipped.columns:
        flipped["close_elo_gap_150"] = (flipped["abs_adjusted_elo_diff"] < 150.0).astype(float)

    if TARGET_COLUMN in flipped.columns:
        target = flipped[TARGET_COLUMN].astype(int)
        if FLIPPED_DECISIVE_TARGET_AS_DRAW:
            flipped[TARGET_COLUMN] = target.map({0: 1, 1: 1, 2: 1}).astype(int)
        else:
            flipped[TARGET_COLUMN] = target.map({0: 2, 1: 1, 2: 0}).astype(int)

    augmented = pd.concat([features_df, flipped], ignore_index=True)
    return augmented.sort_values("date").reset_index(drop=True)


def _build_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    c_value: float = 1.0,
    class_weight: dict[int, float] | str | None = "balanced",
) -> Pipeline:
    """Create the preprocessing and multinomial logistic regression pipeline."""
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("No usable ML feature columns found.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier_kwargs: dict[str, Any] = {
        "C": c_value,
        "max_iter": 2000,
        "class_weight": class_weight,
    }
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        classifier_kwargs["multi_class"] = "multinomial"
    classifier = LogisticRegression(**classifier_kwargs)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def _tune_logistic_hyperparameters(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[float, dict[int, float] | str | None, float, float, float, pd.DataFrame, pd.DataFrame]:
    """Tune logistic C and class weights on a chronological validation split."""
    tune_train_df = train_df[train_df["date"] < VALIDATION_SPLIT_DATE].copy()
    validation_df = train_df[train_df["date"] >= VALIDATION_SPLIT_DATE].copy()
    if tune_train_df.empty or validation_df.empty or tune_train_df[TARGET_COLUMN].nunique() < 2:
        return 1.0, "balanced", 1.0, 0.30, 0.02, pd.DataFrame(
            [
                {
                    "C": 1.0,
                    "class_weight": "balanced",
                    "draw_probability_multiplier": 1.0,
                    "validation_accuracy": np.nan,
                    "validation_log_loss": np.nan,
                    "selected": True,
                }
            ]
        ), pd.DataFrame(
            [
                {
                    "draw_decision_threshold": 0.30,
                    "draw_decision_max_win_prob_gap": 0.02,
                    "validation_decision_accuracy": np.nan,
                    "predicted_draws": np.nan,
                    "selected": True,
                }
            ]
        )

    rows: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_c = 1.0
    best_class_weight: dict[int, float] | str | None = "balanced"
    best_draw_multiplier = 1.0

    for c_value in C_VALUES:
        for class_weight in CLASS_WEIGHT_OPTIONS:
            model = _build_pipeline(
                numeric_columns,
                categorical_columns,
                c_value=c_value,
                class_weight=class_weight,
            )
            fit_df = _augment_with_flipped_rows(tune_train_df)
            model.fit(fit_df[feature_columns], fit_df[TARGET_COLUMN].astype(int))
            base_proba = _probabilities_by_class(model, validation_df[feature_columns])
            for draw_multiplier in DRAW_PROBABILITY_MULTIPLIERS:
                y_proba = _apply_draw_probability_multiplier(base_proba, draw_multiplier)
                y_pred = _top_probability_predictions(y_proba)
                validation_log_loss = float(
                    log_loss(validation_df[TARGET_COLUMN].astype(int), y_proba, labels=CLASS_LABELS)
                )
                validation_accuracy = float(
                    accuracy_score(validation_df[TARGET_COLUMN].astype(int), y_pred)
                )
                key = (validation_log_loss, -validation_accuracy)
                selected = best_key is None or key < best_key
                if selected:
                    best_key = key
                    best_c = c_value
                    best_class_weight = class_weight
                    best_draw_multiplier = draw_multiplier
                rows.append(
                    {
                        "C": c_value,
                        "class_weight": _class_weight_label(class_weight),
                        "draw_probability_multiplier": draw_multiplier,
                        "validation_accuracy": validation_accuracy,
                        "validation_log_loss": validation_log_loss,
                        "selected": False,
                    }
                )

    tuning_results = pd.DataFrame(rows)
    selected_mask = (
        (tuning_results["C"] == best_c)
        & (tuning_results["class_weight"] == _class_weight_label(best_class_weight))
        & (tuning_results["draw_probability_multiplier"] == best_draw_multiplier)
    )
    tuning_results.loc[selected_mask, "selected"] = True

    best_model = _build_pipeline(
        numeric_columns,
        categorical_columns,
        c_value=best_c,
        class_weight=best_class_weight,
    )
    fit_df = _augment_with_flipped_rows(tune_train_df)
    best_model.fit(fit_df[feature_columns], fit_df[TARGET_COLUMN].astype(int))
    validation_proba = _apply_draw_probability_multiplier(
        _probabilities_by_class(best_model, validation_df[feature_columns]),
        best_draw_multiplier,
    )
    best_threshold, best_gap, decision_tuning_results = _tune_draw_decision_rule(
        validation_df[TARGET_COLUMN].astype(int).to_numpy(),
        validation_proba,
    )

    return (
        best_c,
        best_class_weight,
        best_draw_multiplier,
        best_threshold,
        best_gap,
        tuning_results,
        decision_tuning_results,
    )


def _probabilities_by_class(model: Any, feature_rows: pd.DataFrame) -> np.ndarray:
    """Return predict_proba output mapped into columns [0, 1, 2]."""
    raw_proba = model.predict_proba(feature_rows)
    classes = list(getattr(model, "classes_", getattr(model.named_steps["classifier"], "classes_", [])))
    class_to_index = {int(class_label): index for index, class_label in enumerate(classes)}
    mapped = np.zeros((raw_proba.shape[0], len(CLASS_LABELS)), dtype=float)

    for output_index, class_label in enumerate(CLASS_LABELS):
        if class_label in class_to_index:
            mapped[:, output_index] = raw_proba[:, class_to_index[class_label]]

    row_sums = mapped.sum(axis=1)
    missing_rows = row_sums <= 0
    if np.any(missing_rows):
        mapped[missing_rows, :] = 1.0 / len(CLASS_LABELS)
        row_sums = mapped.sum(axis=1)
    return mapped / row_sums[:, None]


def _apply_draw_probability_multiplier(y_proba: np.ndarray, draw_multiplier: float) -> np.ndarray:
    """Increase or decrease draw probability and renormalize each row."""
    adjusted = np.asarray(y_proba, dtype=float).copy()
    adjusted[:, 1] *= float(draw_multiplier)
    row_sums = adjusted.sum(axis=1)
    missing_rows = row_sums <= 0
    if np.any(missing_rows):
        adjusted[missing_rows, :] = 1.0 / len(CLASS_LABELS)
        row_sums = adjusted.sum(axis=1)
    return adjusted / row_sums[:, None]


def _top_probability_predictions(y_proba: np.ndarray) -> np.ndarray:
    """Return class labels from the highest calibrated probability."""
    return np.array(CLASS_LABELS, dtype=int)[np.argmax(y_proba, axis=1)]


def _draw_decision_predictions(
    y_proba: np.ndarray,
    draw_decision_threshold: float,
    draw_decision_max_win_prob_gap: float,
) -> np.ndarray:
    """Return class labels using a calibrated draw decision rule."""
    predictions = _top_probability_predictions(y_proba)
    draw_mask = (
        (y_proba[:, 1] >= float(draw_decision_threshold))
        & (np.abs(y_proba[:, 0] - y_proba[:, 2]) <= float(draw_decision_max_win_prob_gap))
    )
    predictions[draw_mask] = 1
    return predictions


def _tune_draw_decision_rule(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float, pd.DataFrame]:
    """Tune draw hard-classification thresholds on validation accuracy."""
    baseline_pred = _top_probability_predictions(y_proba)
    baseline_accuracy = float(accuracy_score(y_true, baseline_pred))
    baseline_predicted_draws = int((baseline_pred == 1).sum())
    baseline_true_predicted_draws = int(((baseline_pred == 1) & (y_true == 1)).sum())
    baseline_draw_recall = (
        float(baseline_true_predicted_draws / (y_true == 1).sum())
        if (y_true == 1).sum() > 0
        else 0.0
    )
    baseline_draw_precision = (
        float(baseline_true_predicted_draws / baseline_predicted_draws)
        if baseline_predicted_draws > 0
        else 0.0
    )
    rows: list[dict[str, Any]] = [
        {
            "draw_decision_threshold": 1.0,
            "draw_decision_max_win_prob_gap": 0.0,
            "validation_decision_accuracy": baseline_accuracy,
            "predicted_draws": baseline_predicted_draws,
            "draw_recall": baseline_draw_recall,
            "draw_precision": baseline_draw_precision,
            "selected": False,
        }
    ]
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.30
    best_gap = 0.02
    best_accuracy = baseline_accuracy

    for threshold in DRAW_DECISION_THRESHOLDS:
        for gap in DRAW_DECISION_MAX_WIN_PROB_GAPS:
            y_pred = _draw_decision_predictions(y_proba, threshold, gap)
            validation_accuracy = float(accuracy_score(y_true, y_pred))
            predicted_draws = int((y_pred == 1).sum())
            true_draw_predictions = int(((y_pred == 1) & (y_true == 1)).sum())
            draw_recall = (
                float(true_draw_predictions / (y_true == 1).sum())
                if (y_true == 1).sum() > 0
                else 0.0
            )
            draw_precision = (
                float(true_draw_predictions / predicted_draws) if predicted_draws > 0 else 0.0
            )
            key = (validation_accuracy, draw_recall, -float(predicted_draws))
            selected = best_key is None or key > best_key
            if selected:
                best_key = key
                best_threshold = threshold
                best_gap = gap
                best_accuracy = validation_accuracy
            rows.append(
                {
                    "draw_decision_threshold": threshold,
                    "draw_decision_max_win_prob_gap": gap,
                    "validation_decision_accuracy": validation_accuracy,
                    "predicted_draws": predicted_draws,
                    "draw_recall": draw_recall,
                    "draw_precision": draw_precision,
                    "selected": False,
                }
            )

    results = pd.DataFrame(rows)
    if best_accuracy < baseline_accuracy + MIN_DRAW_DECISION_VALIDATION_ACCURACY_GAIN:
        best_threshold = 1.0
        best_gap = 0.0
    selected_mask = (
        (results["draw_decision_threshold"] == best_threshold)
        & (results["draw_decision_max_win_prob_gap"] == best_gap)
    )
    results.loc[selected_mask, "selected"] = True
    return best_threshold, best_gap, results


def _tune_world_cup_adjustment_for_logistic(
    working: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    c_value: float,
    class_weight: dict[int, float] | str | None,
    draw_multiplier: float,
) -> WorldCupAdjustmentParams:
    """Tune World Cup post-processing from rolling logistic validation predictions."""
    fold_predictions: list[np.ndarray] = []
    fold_targets: list[np.ndarray] = []
    fold_features: list[pd.DataFrame] = []

    for _, fold_train_source, fold_validation_source in validation_folds(working):
        fold_train_df = fold_train_source.copy()
        fold_validation_df = fold_validation_source.copy()
        fold_train_df[numeric_columns] = fold_train_df[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        fold_validation_df[numeric_columns] = fold_validation_df[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        for column in categorical_columns:
            fold_train_df[column] = fold_train_df[column].astype("string")
            fold_validation_df[column] = fold_validation_df[column].astype("string")

        model = _build_pipeline(
            numeric_columns,
            categorical_columns,
            c_value=c_value,
            class_weight=class_weight,
        )
        fit_df = _augment_with_flipped_rows(fold_train_df)
        model.fit(fit_df[feature_columns], fit_df[TARGET_COLUMN].astype(int))
        y_proba = _apply_draw_probability_multiplier(
            _probabilities_by_class(model, fold_validation_df[feature_columns]),
            draw_multiplier,
        )
        fold_predictions.append(y_proba)
        fold_targets.append(fold_validation_df[TARGET_COLUMN].astype(int).to_numpy())
        fold_features.append(fold_validation_df)

    if not fold_predictions:
        return WorldCupAdjustmentParams()

    params, _ = tune_world_cup_adjustment(
        np.concatenate(fold_targets),
        np.vstack(fold_predictions),
        pd.concat(fold_features, ignore_index=True),
        output_prefix="logistic",
    )
    return params


def train_logistic_regression_model(
    features_df: pd.DataFrame,
    model_path: str = "models/logistic_match_outcome.joblib",
) -> tuple[object, pd.DataFrame]:
    """Train a chronological multinomial logistic regression classifier.

    The default holdout uses matches on or after 2022-01-01 as the test set.
    The trained sklearn pipeline is saved to ``model_path``.
    """
    if TARGET_COLUMN not in features_df.columns:
        raise ValueError("features_df must include a target column.")

    working = features_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date", TARGET_COLUMN]).sort_values("date").reset_index(drop=True)
    if working.empty:
        raise ValueError("No rows available for logistic regression training.")

    numeric_columns, categorical_columns = _available_feature_columns(working)
    feature_columns = numeric_columns + categorical_columns
    working[numeric_columns] = working[numeric_columns].apply(pd.to_numeric, errors="coerce")
    for column in categorical_columns:
        working[column] = working[column].astype("string")

    train_df = working[working["date"] < TRAIN_TEST_SPLIT_DATE].copy()
    test_df = working[working["date"] >= TRAIN_TEST_SPLIT_DATE].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            "Chronological split requires training rows before 2022-01-01 "
            "and test rows from 2022-01-01 onward."
        )
    if train_df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training data must contain at least two outcome classes.")

    (
        best_c,
        best_class_weight,
        best_draw_multiplier,
        best_draw_decision_threshold,
        best_draw_decision_max_win_prob_gap,
        tuning_results,
        decision_tuning_results,
    ) = _tune_logistic_hyperparameters(
        train_df,
        feature_columns,
        numeric_columns,
        categorical_columns,
    )
    tuning_output = Path("results/logistic_tuning_results.csv")
    tuning_output.parent.mkdir(parents=True, exist_ok=True)
    tuning_results.to_csv(tuning_output, index=False)
    decision_tuning_results.to_csv("results/logistic_draw_decision_tuning_results.csv", index=False)

    model = _build_pipeline(
        numeric_columns,
        categorical_columns,
        c_value=best_c,
        class_weight=best_class_weight,
    )
    fit_train_df = _augment_with_flipped_rows(train_df)
    model.fit(fit_train_df[feature_columns], fit_train_df[TARGET_COLUMN].astype(int))

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    y_proba = _apply_draw_probability_multiplier(
        _probabilities_by_class(model, test_df[feature_columns]),
        best_draw_multiplier,
    )
    world_cup_adjustment_params = _tune_world_cup_adjustment_for_logistic(
        working,
        feature_columns,
        numeric_columns,
        categorical_columns,
        best_c,
        best_class_weight,
        best_draw_multiplier,
    )
    y_proba = apply_world_cup_adjustment(y_proba, test_df, world_cup_adjustment_params)
    top_probability_pred = _top_probability_predictions(y_proba)
    y_pred = _draw_decision_predictions(
        y_proba,
        best_draw_decision_threshold,
        best_draw_decision_max_win_prob_gap,
    )
    metadata_columns = [
        column
        for column in ["date", "team_a", "team_b", "tournament", "neutral", TARGET_COLUMN]
        if column in test_df.columns
    ]
    prediction_rows = test_df[metadata_columns].copy()
    prediction_rows["top_probability_target"] = top_probability_pred
    prediction_rows["predicted_target"] = y_pred
    prediction_rows["team_a_win_prob"] = y_proba[:, 0]
    prediction_rows["draw_prob"] = y_proba[:, 1]
    prediction_rows["team_b_win_prob"] = y_proba[:, 2]
    prediction_rows.attrs["feature_columns"] = feature_columns
    prediction_rows.attrs["numeric_columns"] = numeric_columns
    prediction_rows.attrs["categorical_columns"] = categorical_columns
    prediction_rows.attrs["best_c"] = best_c
    prediction_rows.attrs["best_class_weight"] = _class_weight_label(best_class_weight)
    prediction_rows.attrs["draw_probability_multiplier"] = best_draw_multiplier
    prediction_rows.attrs["world_cup_group_shrinkage"] = world_cup_adjustment_params.group_shrinkage
    prediction_rows.attrs["world_cup_knockout_shrinkage"] = world_cup_adjustment_params.knockout_shrinkage
    prediction_rows.attrs["world_cup_rotation_strength"] = world_cup_adjustment_params.rotation_strength
    prediction_rows.attrs["world_cup_rotation_draw_share"] = world_cup_adjustment_params.rotation_draw_share
    prediction_rows.attrs["draw_decision_threshold"] = best_draw_decision_threshold
    prediction_rows.attrs["draw_decision_max_win_prob_gap"] = best_draw_decision_max_win_prob_gap
    prediction_rows.attrs["use_flipped_training_rows"] = USE_FLIPPED_TRAINING_ROWS
    prediction_rows.attrs["flipped_decisive_target_as_draw"] = FLIPPED_DECISIVE_TARGET_AS_DRAW
    return model, prediction_rows


def load_logistic_model(
    model_path: str = "models/logistic_match_outcome.joblib",
):
    """Load a saved logistic regression pipeline."""
    return joblib.load(model_path)


def predict_match_logistic(
    team_a: str,
    team_b: str,
    model,
    feature_row: pd.DataFrame,
) -> dict[str, float | str]:
    """Predict W/D/L probabilities for one match feature row."""
    if feature_row.empty:
        raise ValueError("feature_row must contain one row of pre-match features.")

    probabilities = _probabilities_by_class(model, feature_row.iloc[[0]])[0]
    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_win_prob": float(probabilities[0]),
        "draw_prob": float(probabilities[1]),
        "team_b_win_prob": float(probabilities[2]),
    }

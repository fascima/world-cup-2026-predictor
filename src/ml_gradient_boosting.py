"""Histogram gradient boosting model for match outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.ml_logistic import (
    CATEGORICAL_FEATURES,
    CLASS_LABELS,
    DISABLED_MODEL_FEATURES,
    DRAW_DECISION_MAX_WIN_PROB_GAPS,
    DRAW_DECISION_THRESHOLDS,
    DRAW_PROBABILITY_MULTIPLIERS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    TRAIN_TEST_SPLIT_DATE,
    VALIDATION_SPLIT_DATE,
    _apply_draw_probability_multiplier,
    _augment_with_flipped_rows,
    _draw_decision_predictions,
    _probabilities_by_class,
    _top_probability_predictions,
)
from src.probability_adjustment import (
    WorldCupAdjustmentParams,
    apply_world_cup_adjustment,
    tune_world_cup_adjustment,
)


RAW_MARKET_VALUE_COLUMNS = {
    "team_a_market_value_eur",
    "team_b_market_value_eur",
    "market_value_baseline_eur",
    "team_a_effective_market_value_eur",
    "team_b_effective_market_value_eur",
}
MARKET_VALUE_COLUMNS = {
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
ORIGINAL_MARKET_DERIVED_COLUMNS = {
    "market_value_log_ratio",
    "market_value_adjustment_diff",
}
WORLD_CUP_PHASE_COLUMNS = {
    "is_world_cup",
    "is_world_cup_knockout",
    "world_cup_knockout_abs_adjusted_elo_diff",
    "world_cup_knockout_close_elo_gap_100",
    "world_cup_group_abs_adjusted_elo_diff",
}
WORLD_CUP_PRIOR_COLUMNS = {
    "team_a_wc_prior_matches",
    "team_a_wc_prior_weight",
    "team_a_wc_prior_points_per_match",
    "team_a_wc_prior_goal_diff_per_match",
    "team_a_wc_prior_win_rate",
    "team_a_wc_prior_draw_rate",
    "team_a_wc_prior_knockout_matches",
    "team_b_wc_prior_matches",
    "team_b_wc_prior_weight",
    "team_b_wc_prior_points_per_match",
    "team_b_wc_prior_goal_diff_per_match",
    "team_b_wc_prior_win_rate",
    "team_b_wc_prior_draw_rate",
    "team_b_wc_prior_knockout_matches",
    "wc_prior_matches_diff",
    "wc_prior_weight_diff",
    "wc_prior_points_per_match_diff",
    "wc_prior_goal_diff_per_match_diff",
    "wc_prior_win_rate_diff",
    "wc_prior_draw_rate_diff",
    "wc_prior_knockout_matches_diff",
}
INJURY_COLUMNS = {
    "has_injury_data",
    "team_a_injured_players_count",
    "team_a_injured_market_value_eur",
    "team_a_injured_market_value_share",
    "team_a_key_injured_players_count",
    "team_a_max_injured_player_market_value_eur",
    "team_a_injured_gk_count",
    "team_a_injured_gk_market_value_eur",
    "team_a_injured_df_count",
    "team_a_injured_df_market_value_eur",
    "team_a_injured_mf_count",
    "team_a_injured_mf_market_value_eur",
    "team_a_injured_fw_count",
    "team_a_injured_fw_market_value_eur",
    "team_b_injured_players_count",
    "team_b_injured_market_value_eur",
    "team_b_injured_market_value_share",
    "team_b_key_injured_players_count",
    "team_b_max_injured_player_market_value_eur",
    "team_b_injured_gk_count",
    "team_b_injured_gk_market_value_eur",
    "team_b_injured_df_count",
    "team_b_injured_df_market_value_eur",
    "team_b_injured_mf_count",
    "team_b_injured_mf_market_value_eur",
    "team_b_injured_fw_count",
    "team_b_injured_fw_market_value_eur",
    "injured_players_count_diff",
    "injured_market_value_diff",
    "injured_market_value_share_diff",
    "key_injured_players_count_diff",
    "max_injured_player_market_value_diff",
    "injured_gk_count_diff",
    "injured_gk_market_value_diff",
    "injured_df_count_diff",
    "injured_df_market_value_diff",
    "injured_mf_count_diff",
    "injured_mf_market_value_diff",
    "injured_fw_count_diff",
    "injured_fw_market_value_diff",
}
EXPERIMENTAL_CONTEXT_COLUMNS = WORLD_CUP_PHASE_COLUMNS | WORLD_CUP_PRIOR_COLUMNS | INJURY_COLUMNS
FEATURE_PROFILES = {
    "baseline_no_market": DISABLED_MODEL_FEATURES | EXPERIMENTAL_CONTEXT_COLUMNS,
    "context_no_market": MARKET_VALUE_COLUMNS,
    "effective_market_derived": RAW_MARKET_VALUE_COLUMNS | ORIGINAL_MARKET_DERIVED_COLUMNS,
    "market_full": set(),
}
MIN_GRADIENT_BOOSTING_DRAW_DECISION_VALIDATION_ACCURACY_GAIN = 0.005
CALIBRATION_TEMPERATURES = [0.85, 1.0, 1.15, 1.3, 1.5]
PARAMETER_GRID = [
    {
        "max_iter": 80,
        "learning_rate": 0.04,
        "max_leaf_nodes": 7,
        "l2_regularization": 1.0,
        "class_weight": None,
    },
    {
        "max_iter": 120,
        "learning_rate": 0.03,
        "max_leaf_nodes": 7,
        "l2_regularization": 1.0,
        "class_weight": None,
    },
    {
        "max_iter": 120,
        "learning_rate": 0.04,
        "max_leaf_nodes": 15,
        "l2_regularization": 3.0,
        "class_weight": None,
    },
    {
        "max_iter": 80,
        "learning_rate": 0.03,
        "max_leaf_nodes": 15,
        "l2_regularization": 5.0,
        "class_weight": {0: 1.0, 1: 1.05, 2: 1.0},
    },
]


def _available_feature_columns(
    features_df: pd.DataFrame,
    disabled_features: set[str],
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical gradient-boosting feature columns."""
    categorical_columns = [
        column for column in CATEGORICAL_FEATURES if column in features_df.columns
    ]
    excluded_columns = METADATA_COLUMNS | set(categorical_columns) | disabled_features
    candidate_columns = [column for column in features_df.columns if column not in excluded_columns]
    numeric_columns = [
        column
        for column in candidate_columns
        if pd.api.types.is_numeric_dtype(features_df[column])
        or pd.api.types.is_bool_dtype(features_df[column])
    ]
    return numeric_columns, categorical_columns


def _class_weight_label(class_weight: dict[int, float] | str | None) -> str:
    """Return a readable label for a class-weight setting."""
    if class_weight is None:
        return "none"
    if isinstance(class_weight, str):
        return class_weight
    return ";".join(f"{key}:{value:g}" for key, value in sorted(class_weight.items()))


def _build_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, Any],
) -> Pipeline:
    """Create a preprocessing and histogram gradient boosting pipeline."""
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median"),
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
        raise ValueError("No usable gradient boosting feature columns found.")

    classifier = HistGradientBoostingClassifier(
        max_iter=int(params["max_iter"]),
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        l2_regularization=float(params["l2_regularization"]),
        random_state=42,
        early_stopping=False,
    )
    return Pipeline(
        steps=[
            ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
            ("classifier", classifier),
        ]
    )


def _fit_model(
    model: Pipeline,
    fit_df: pd.DataFrame,
    feature_columns: list[str],
    class_weight: dict[int, float] | str | None,
) -> None:
    """Fit a model with optional per-row class weights."""
    y = fit_df[TARGET_COLUMN].astype(int)
    if class_weight is None:
        model.fit(fit_df[feature_columns], y)
        return

    sample_weight = compute_sample_weight(class_weight=class_weight, y=y)
    model.fit(fit_df[feature_columns], y, classifier__sample_weight=sample_weight)


def _apply_probability_calibration(
    y_proba: np.ndarray,
    draw_multiplier: float,
    temperature: float,
) -> np.ndarray:
    """Apply draw-prior and temperature calibration to class probabilities."""
    adjusted = _apply_draw_probability_multiplier(y_proba, draw_multiplier)
    if temperature == 1.0:
        return adjusted

    clipped = np.clip(adjusted, 1e-12, 1.0)
    powered = clipped ** (1.0 / float(temperature))
    row_sums = powered.sum(axis=1)
    return powered / row_sums[:, None]


def _validation_folds(train_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Return chronological folds for gradient boosting tuning."""
    folds: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    world_cup_rows = train_df[train_df.get("tournament_type", "").eq("world_cup")].copy()
    for year in sorted(world_cup_rows["date"].dt.year.unique()):
        validation_df = world_cup_rows[world_cup_rows["date"].dt.year == year].copy()
        if len(validation_df) < 20 or validation_df[TARGET_COLUMN].nunique() < 2:
            continue
        fold_train_df = train_df[train_df["date"] < validation_df["date"].min()].copy()
        if fold_train_df.empty or fold_train_df[TARGET_COLUMN].nunique() < 2:
            continue
        folds.append((f"world_cup_{int(year)}", fold_train_df, validation_df))

    recent_train_df = train_df[train_df["date"] < VALIDATION_SPLIT_DATE].copy()
    recent_validation_df = train_df[train_df["date"] >= VALIDATION_SPLIT_DATE].copy()
    if (
        not recent_train_df.empty
        and not recent_validation_df.empty
        and recent_train_df[TARGET_COLUMN].nunique() >= 2
        and recent_validation_df[TARGET_COLUMN].nunique() >= 2
    ):
        folds.append(("recent_2020_2021", recent_train_df, recent_validation_df))

    return folds


def _prepare_columns(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coerce feature columns consistently for one fold."""
    fold_train_df = train_df.copy()
    fold_validation_df = validation_df.copy()
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
    return fold_train_df, fold_validation_df


def _tune_draw_decision_rule(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float, pd.DataFrame]:
    """Tune draw hard-classification thresholds on validation accuracy."""
    baseline_pred = _top_probability_predictions(y_proba)
    baseline_accuracy = float(accuracy_score(y_true, baseline_pred))
    baseline_predicted_draws = int((baseline_pred == 1).sum())
    baseline_true_predicted_draws = int(((baseline_pred == 1) & (y_true == 1)).sum())
    actual_draws = int((y_true == 1).sum())
    rows: list[dict[str, Any]] = [
        {
            "draw_decision_threshold": 1.0,
            "draw_decision_max_win_prob_gap": 0.0,
            "validation_decision_accuracy": baseline_accuracy,
            "predicted_draws": baseline_predicted_draws,
            "draw_recall": (
                float(baseline_true_predicted_draws / actual_draws) if actual_draws else 0.0
            ),
            "draw_precision": (
                float(baseline_true_predicted_draws / baseline_predicted_draws)
                if baseline_predicted_draws
                else 0.0
            ),
            "selected": False,
        }
    ]
    best_key: tuple[float, float, float] | None = None
    best_threshold = 1.0
    best_gap = 0.0
    best_accuracy = baseline_accuracy

    for threshold in DRAW_DECISION_THRESHOLDS:
        for gap in DRAW_DECISION_MAX_WIN_PROB_GAPS:
            y_pred = _draw_decision_predictions(y_proba, threshold, gap)
            validation_accuracy = float(accuracy_score(y_true, y_pred))
            predicted_draws = int((y_pred == 1).sum())
            true_predicted_draws = int(((y_pred == 1) & (y_true == 1)).sum())
            draw_recall = float(true_predicted_draws / actual_draws) if actual_draws else 0.0
            draw_precision = (
                float(true_predicted_draws / predicted_draws) if predicted_draws else 0.0
            )
            key = (validation_accuracy, draw_recall, -float(predicted_draws))
            if best_key is None or key > best_key:
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

    if best_accuracy < (
        baseline_accuracy + MIN_GRADIENT_BOOSTING_DRAW_DECISION_VALIDATION_ACCURACY_GAIN
    ):
        best_threshold = 1.0
        best_gap = 0.0

    results = pd.DataFrame(rows)
    selected_mask = (
        (results["draw_decision_threshold"] == best_threshold)
        & (results["draw_decision_max_win_prob_gap"] == best_gap)
    )
    results.loc[selected_mask, "selected"] = True
    return best_threshold, best_gap, results


def _tune_gradient_boosting(
    train_df: pd.DataFrame,
) -> tuple[str, dict[str, Any], float, float, float, float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune profile, tree parameters, calibration, and draw decision rule."""
    folds = _validation_folds(train_df)
    if not folds:
        fallback_params = PARAMETER_GRID[0]
        return (
            "baseline_no_market",
            fallback_params,
            1.0,
            1.0,
            1.0,
            0.0,
            pd.DataFrame([{**fallback_params, "feature_profile": "baseline_no_market", "selected": True}]),
            pd.DataFrame(),
            pd.DataFrame(
                [
                    {
                        "draw_decision_threshold": 1.0,
                        "draw_decision_max_win_prob_gap": 0.0,
                        "selected": True,
                    }
                ]
            ),
        )

    rows: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_profile = "baseline_no_market"
    best_params = PARAMETER_GRID[0]
    best_draw_multiplier = 1.0
    best_temperature = 1.0
    best_validation_y_true = np.array([], dtype=int)
    best_validation_proba = np.empty((0, len(CLASS_LABELS)), dtype=float)
    fold_result_rows: list[dict[str, Any]] = []

    for profile_name, disabled_features in FEATURE_PROFILES.items():
        numeric_columns, categorical_columns = _available_feature_columns(
            train_df,
            disabled_features=disabled_features,
        )
        feature_columns = numeric_columns + categorical_columns

        for params in PARAMETER_GRID:
            fold_predictions: list[tuple[str, np.ndarray, np.ndarray]] = []
            for fold_name, fold_train_source, fold_validation_source in folds:
                fold_train_df, fold_validation_df = _prepare_columns(
                    fold_train_source,
                    fold_validation_source,
                    numeric_columns,
                    categorical_columns,
                )
                fit_df = _augment_with_flipped_rows(fold_train_df)
                model = _build_pipeline(numeric_columns, categorical_columns, params)
                _fit_model(model, fit_df, feature_columns, params["class_weight"])
                base_proba = _probabilities_by_class(model, fold_validation_df[feature_columns])
                fold_predictions.append(
                    (
                        fold_name,
                        fold_validation_df[TARGET_COLUMN].astype(int).to_numpy(),
                        base_proba,
                    )
                )

            for draw_multiplier in DRAW_PROBABILITY_MULTIPLIERS:
                for temperature in CALIBRATION_TEMPERATURES:
                    calibrated_folds = [
                        (
                            fold_name,
                            y_true,
                            _apply_probability_calibration(
                                base_proba,
                                draw_multiplier,
                                temperature,
                            ),
                        )
                        for fold_name, y_true, base_proba in fold_predictions
                    ]
                    fold_log_losses = [
                        float(log_loss(y_true, y_proba, labels=CLASS_LABELS))
                        for _, y_true, y_proba in calibrated_folds
                    ]
                    fold_accuracies = [
                        float(accuracy_score(y_true, _top_probability_predictions(y_proba)))
                        for _, y_true, y_proba in calibrated_folds
                    ]
                    validation_log_loss = float(np.mean(fold_log_losses))
                    validation_accuracy = float(np.mean(fold_accuracies))
                    pooled_y_true = np.concatenate([y_true for _, y_true, _ in calibrated_folds])
                    pooled_y_proba = np.vstack([y_proba for _, _, y_proba in calibrated_folds])
                    pooled_accuracy = float(
                        accuracy_score(pooled_y_true, _top_probability_predictions(pooled_y_proba))
                    )
                    key = (validation_log_loss, -validation_accuracy)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_profile = profile_name
                        best_params = dict(params)
                        best_draw_multiplier = draw_multiplier
                        best_temperature = temperature
                        best_validation_y_true = pooled_y_true
                        best_validation_proba = pooled_y_proba
                    rows.append(
                        {
                            "feature_profile": profile_name,
                            "max_iter": params["max_iter"],
                            "learning_rate": params["learning_rate"],
                            "max_leaf_nodes": params["max_leaf_nodes"],
                            "l2_regularization": params["l2_regularization"],
                            "class_weight": _class_weight_label(params["class_weight"]),
                            "draw_probability_multiplier": draw_multiplier,
                            "calibration_temperature": temperature,
                            "validation_accuracy": validation_accuracy,
                            "pooled_validation_accuracy": pooled_accuracy,
                            "validation_log_loss": validation_log_loss,
                            "selected": False,
                        }
                    )
                    for (fold_name, y_true, y_proba), fold_log_loss, fold_accuracy in zip(
                        calibrated_folds,
                        fold_log_losses,
                        fold_accuracies,
                        strict=False,
                    ):
                        fold_result_rows.append(
                            {
                                "feature_profile": profile_name,
                                "max_iter": params["max_iter"],
                                "learning_rate": params["learning_rate"],
                                "max_leaf_nodes": params["max_leaf_nodes"],
                                "l2_regularization": params["l2_regularization"],
                                "class_weight": _class_weight_label(params["class_weight"]),
                                "draw_probability_multiplier": draw_multiplier,
                                "calibration_temperature": temperature,
                                "fold": fold_name,
                                "fold_matches": len(y_true),
                                "fold_accuracy": fold_accuracy,
                                "fold_log_loss": fold_log_loss,
                            }
                        )

    tuning_results = pd.DataFrame(rows)
    selected_mask = (
        (tuning_results["feature_profile"] == best_profile)
        & (tuning_results["max_iter"] == best_params["max_iter"])
        & (tuning_results["learning_rate"] == best_params["learning_rate"])
        & (tuning_results["max_leaf_nodes"] == best_params["max_leaf_nodes"])
        & (tuning_results["l2_regularization"] == best_params["l2_regularization"])
        & (tuning_results["class_weight"] == _class_weight_label(best_params["class_weight"]))
        & (tuning_results["draw_probability_multiplier"] == best_draw_multiplier)
        & (tuning_results["calibration_temperature"] == best_temperature)
    )
    tuning_results.loc[selected_mask, "selected"] = True
    best_threshold, best_gap, decision_tuning_results = _tune_draw_decision_rule(
        best_validation_y_true,
        best_validation_proba,
    )

    return (
        best_profile,
        best_params,
        best_draw_multiplier,
        best_temperature,
        best_threshold,
        best_gap,
        tuning_results,
        pd.DataFrame(fold_result_rows),
        decision_tuning_results,
    )


def _tune_world_cup_adjustment_for_gradient_boosting(
    working: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, Any],
    draw_multiplier: float,
    temperature: float,
) -> WorldCupAdjustmentParams:
    """Tune World Cup post-processing from rolling gradient-boosting validation predictions."""
    fold_predictions: list[np.ndarray] = []
    fold_targets: list[np.ndarray] = []
    fold_features: list[pd.DataFrame] = []

    for _, fold_train_source, fold_validation_source in _validation_folds(working):
        fold_train_df, fold_validation_df = _prepare_columns(
            fold_train_source,
            fold_validation_source,
            numeric_columns,
            categorical_columns,
        )
        fit_df = _augment_with_flipped_rows(fold_train_df)
        model = _build_pipeline(numeric_columns, categorical_columns, params)
        _fit_model(model, fit_df, feature_columns, params["class_weight"])
        y_proba = _apply_probability_calibration(
            _probabilities_by_class(model, fold_validation_df[feature_columns]),
            draw_multiplier,
            temperature,
        )
        fold_predictions.append(y_proba)
        fold_targets.append(fold_validation_df[TARGET_COLUMN].astype(int).to_numpy())
        fold_features.append(fold_validation_df)

    if not fold_predictions:
        return WorldCupAdjustmentParams()

    tuned_params, _ = tune_world_cup_adjustment(
        np.concatenate(fold_targets),
        np.vstack(fold_predictions),
        pd.concat(fold_features, ignore_index=True),
        output_prefix="gradient_boosting",
    )
    return tuned_params


def train_gradient_boosting_model(
    features_df: pd.DataFrame,
    model_path: str = "models/gradient_boosting_match_outcome.joblib",
) -> tuple[object, pd.DataFrame]:
    """Train a chronological histogram gradient boosting classifier."""
    if TARGET_COLUMN not in features_df.columns:
        raise ValueError("features_df must include a target column.")

    working = features_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date", TARGET_COLUMN]).sort_values("date").reset_index(drop=True)
    if working.empty:
        raise ValueError("No rows available for gradient boosting training.")

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
        best_profile,
        best_params,
        best_draw_multiplier,
        best_temperature,
        best_draw_decision_threshold,
        best_draw_decision_max_win_prob_gap,
        tuning_results,
        fold_results,
        decision_tuning_results,
    ) = _tune_gradient_boosting(train_df)
    Path("results").mkdir(parents=True, exist_ok=True)
    tuning_results.to_csv("results/gradient_boosting_tuning_results.csv", index=False)
    fold_results.to_csv("results/gradient_boosting_rolling_validation_results.csv", index=False)
    decision_tuning_results.to_csv(
        "results/gradient_boosting_draw_decision_tuning_results.csv",
        index=False,
    )

    numeric_columns, categorical_columns = _available_feature_columns(
        working,
        disabled_features=FEATURE_PROFILES[best_profile],
    )
    feature_columns = numeric_columns + categorical_columns
    working[numeric_columns] = working[numeric_columns].apply(pd.to_numeric, errors="coerce")
    for column in categorical_columns:
        working[column] = working[column].astype("string")
    train_df = working[working["date"] < TRAIN_TEST_SPLIT_DATE].copy()
    test_df = working[working["date"] >= TRAIN_TEST_SPLIT_DATE].copy()

    model = _build_pipeline(numeric_columns, categorical_columns, best_params)
    fit_train_df = _augment_with_flipped_rows(train_df)
    _fit_model(model, fit_train_df, feature_columns, best_params["class_weight"])

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    y_proba = _apply_probability_calibration(
        _probabilities_by_class(model, test_df[feature_columns]),
        best_draw_multiplier,
        best_temperature,
    )
    world_cup_adjustment_params = _tune_world_cup_adjustment_for_gradient_boosting(
        working,
        feature_columns,
        numeric_columns,
        categorical_columns,
        best_params,
        best_draw_multiplier,
        best_temperature,
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
    prediction_rows.attrs["feature_profile"] = best_profile
    prediction_rows.attrs["feature_columns"] = feature_columns
    prediction_rows.attrs["numeric_columns"] = numeric_columns
    prediction_rows.attrs["categorical_columns"] = categorical_columns
    prediction_rows.attrs["max_iter"] = best_params["max_iter"]
    prediction_rows.attrs["learning_rate"] = best_params["learning_rate"]
    prediction_rows.attrs["max_leaf_nodes"] = best_params["max_leaf_nodes"]
    prediction_rows.attrs["l2_regularization"] = best_params["l2_regularization"]
    prediction_rows.attrs["class_weight"] = _class_weight_label(best_params["class_weight"])
    prediction_rows.attrs["draw_probability_multiplier"] = best_draw_multiplier
    prediction_rows.attrs["calibration_temperature"] = best_temperature
    prediction_rows.attrs["world_cup_group_shrinkage"] = world_cup_adjustment_params.group_shrinkage
    prediction_rows.attrs["world_cup_knockout_shrinkage"] = world_cup_adjustment_params.knockout_shrinkage
    prediction_rows.attrs["world_cup_rotation_strength"] = world_cup_adjustment_params.rotation_strength
    prediction_rows.attrs["world_cup_rotation_draw_share"] = world_cup_adjustment_params.rotation_draw_share
    prediction_rows.attrs["draw_decision_threshold"] = best_draw_decision_threshold
    prediction_rows.attrs["draw_decision_max_win_prob_gap"] = best_draw_decision_max_win_prob_gap
    return model, prediction_rows

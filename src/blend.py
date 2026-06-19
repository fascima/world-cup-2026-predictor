"""Blend tuning for supervised match outcome models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.ml_gradient_boosting import (
    FEATURE_PROFILES,
    _apply_probability_calibration,
    _available_feature_columns as _available_gradient_boosting_feature_columns,
    _build_pipeline as _build_gradient_boosting_pipeline,
    _fit_model as _fit_gradient_boosting_model,
    _prepare_columns as _prepare_gradient_boosting_columns,
)
from src.ml_logistic import (
    CLASS_LABELS,
    TARGET_COLUMN,
    _apply_draw_probability_multiplier,
    _augment_with_flipped_rows,
    _available_feature_columns as _available_logistic_feature_columns,
    _build_pipeline as _build_logistic_pipeline,
    _probabilities_by_class,
    _top_probability_predictions,
)
from src.probability_adjustment import (
    WorldCupAdjustmentParams,
    apply_world_cup_adjustment,
    validation_folds,
)


BLEND_WEIGHT_GRID = [i / 10.0 for i in range(11)]


def _parse_class_weight(label: str | float | None) -> dict[int, float] | str | None:
    if label is None or pd.isna(label) or str(label) == "none":
        return None
    text = str(label)
    if text in {"balanced", "None"}:
        return text if text == "balanced" else None
    values: dict[int, float] = {}
    for item in text.split(";"):
        key, value = item.split(":", 1)
        values[int(key)] = float(value)
    return values


def _selected_row(path: str) -> pd.Series | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    rows = pd.read_csv(csv_path)
    selected = rows[rows["selected"].astype(bool)]
    if selected.empty:
        return None
    return selected.iloc[0]


def _selected_world_cup_params(path: str) -> WorldCupAdjustmentParams:
    row = _selected_row(path)
    if row is None:
        return WorldCupAdjustmentParams()
    return WorldCupAdjustmentParams(
        group_shrinkage=float(row["group_shrinkage"]),
        knockout_shrinkage=float(row["knockout_shrinkage"]),
        rotation_strength=float(row["rotation_strength"]),
        rotation_draw_share=float(row["rotation_draw_share"]),
    )


def _selected_logistic_settings() -> dict[str, Any]:
    row = _selected_row("results/logistic_tuning_results.csv")
    if row is None:
        return {"C": 1.0, "class_weight": None, "draw_probability_multiplier": 1.0}
    return {
        "C": float(row["C"]),
        "class_weight": _parse_class_weight(row["class_weight"]),
        "draw_probability_multiplier": float(row["draw_probability_multiplier"]),
    }


def _selected_gradient_boosting_settings() -> dict[str, Any]:
    row = _selected_row("results/gradient_boosting_tuning_results.csv")
    if row is None:
        return {
            "feature_profile": "no_market",
            "max_iter": 120,
            "learning_rate": 0.03,
            "max_leaf_nodes": 7,
            "l2_regularization": 1.0,
            "class_weight": None,
            "draw_probability_multiplier": 1.0,
            "calibration_temperature": 1.0,
        }
    return {
        "feature_profile": str(row["feature_profile"]),
        "max_iter": int(row["max_iter"]),
        "learning_rate": float(row["learning_rate"]),
        "max_leaf_nodes": int(row["max_leaf_nodes"]),
        "l2_regularization": float(row["l2_regularization"]),
        "class_weight": _parse_class_weight(row["class_weight"]),
        "draw_probability_multiplier": float(row["draw_probability_multiplier"]),
        "calibration_temperature": float(row["calibration_temperature"]),
    }


def tune_blend_weight(
    features_df: pd.DataFrame,
    output_path: str = "results/blend_weight_tuning_results.csv",
) -> tuple[float, pd.DataFrame]:
    """Tune logistic/gradient-boosting blend weight on rolling validation folds."""
    working = features_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date", TARGET_COLUMN]).sort_values("date").reset_index(drop=True)
    folds = validation_folds(working)
    if not folds:
        fallback = pd.DataFrame(
            [
                {
                    "logistic_weight": 0.5,
                    "gradient_boosting_weight": 0.5,
                    "validation_log_loss": np.nan,
                    "validation_accuracy": np.nan,
                    "selected": True,
                }
            ]
        )
        fallback.to_csv(output_path, index=False)
        return 0.5, fallback

    logistic_settings = _selected_logistic_settings()
    gradient_boosting_settings = _selected_gradient_boosting_settings()
    logistic_wc_params = _selected_world_cup_params("results/logistic_world_cup_adjustment_tuning.csv")
    gradient_boosting_wc_params = _selected_world_cup_params(
        "results/gradient_boosting_world_cup_adjustment_tuning.csv"
    )

    logistic_predictions: list[np.ndarray] = []
    gradient_boosting_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    logistic_numeric, logistic_categorical = _available_logistic_feature_columns(working)
    logistic_features = logistic_numeric + logistic_categorical

    gb_disabled = FEATURE_PROFILES[gradient_boosting_settings["feature_profile"]]
    gb_numeric, gb_categorical = _available_gradient_boosting_feature_columns(
        working,
        disabled_features=gb_disabled,
    )
    gb_features = gb_numeric + gb_categorical
    gb_params = {
        "max_iter": gradient_boosting_settings["max_iter"],
        "learning_rate": gradient_boosting_settings["learning_rate"],
        "max_leaf_nodes": gradient_boosting_settings["max_leaf_nodes"],
        "l2_regularization": gradient_boosting_settings["l2_regularization"],
        "class_weight": gradient_boosting_settings["class_weight"],
    }

    for _, fold_train_source, fold_validation_source in folds:
        logistic_train = fold_train_source.copy()
        logistic_validation = fold_validation_source.copy()
        logistic_train[logistic_numeric] = logistic_train[logistic_numeric].apply(
            pd.to_numeric,
            errors="coerce",
        )
        logistic_validation[logistic_numeric] = logistic_validation[logistic_numeric].apply(
            pd.to_numeric,
            errors="coerce",
        )
        for column in logistic_categorical:
            logistic_train[column] = logistic_train[column].astype("string")
            logistic_validation[column] = logistic_validation[column].astype("string")

        logistic_model = _build_logistic_pipeline(
            logistic_numeric,
            logistic_categorical,
            c_value=logistic_settings["C"],
            class_weight=logistic_settings["class_weight"],
        )
        logistic_fit = _augment_with_flipped_rows(logistic_train)
        logistic_model.fit(
            logistic_fit[logistic_features],
            logistic_fit[TARGET_COLUMN].astype(int),
        )
        logistic_proba = _apply_draw_probability_multiplier(
            _probabilities_by_class(logistic_model, logistic_validation[logistic_features]),
            logistic_settings["draw_probability_multiplier"],
        )
        logistic_proba = apply_world_cup_adjustment(
            logistic_proba,
            logistic_validation,
            logistic_wc_params,
        )

        gb_train, gb_validation = _prepare_gradient_boosting_columns(
            fold_train_source,
            fold_validation_source,
            gb_numeric,
            gb_categorical,
        )
        gb_model = _build_gradient_boosting_pipeline(gb_numeric, gb_categorical, gb_params)
        gb_fit = _augment_with_flipped_rows(gb_train)
        _fit_gradient_boosting_model(gb_model, gb_fit, gb_features, gb_params["class_weight"])
        gb_proba = _apply_probability_calibration(
            _probabilities_by_class(gb_model, gb_validation[gb_features]),
            gradient_boosting_settings["draw_probability_multiplier"],
            gradient_boosting_settings["calibration_temperature"],
        )
        gb_proba = apply_world_cup_adjustment(
            gb_proba,
            gb_validation,
            gradient_boosting_wc_params,
        )

        logistic_predictions.append(logistic_proba)
        gradient_boosting_predictions.append(gb_proba)
        targets.append(fold_validation_source[TARGET_COLUMN].astype(int).to_numpy())

    y_true = np.concatenate(targets)
    logistic_proba = np.vstack(logistic_predictions)
    gradient_boosting_proba = np.vstack(gradient_boosting_predictions)

    rows: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_weight = 0.5
    for logistic_weight in BLEND_WEIGHT_GRID:
        gradient_boosting_weight = 1.0 - logistic_weight
        y_proba = (
            logistic_weight * logistic_proba
            + gradient_boosting_weight * gradient_boosting_proba
        )
        y_pred = _top_probability_predictions(y_proba)
        validation_log_loss = float(log_loss(y_true, y_proba, labels=CLASS_LABELS))
        validation_accuracy = float(accuracy_score(y_true, y_pred))
        key = (validation_log_loss, -validation_accuracy)
        if best_key is None or key < best_key:
            best_key = key
            best_weight = logistic_weight
        rows.append(
            {
                "logistic_weight": logistic_weight,
                "gradient_boosting_weight": gradient_boosting_weight,
                "validation_log_loss": validation_log_loss,
                "validation_accuracy": validation_accuracy,
                "selected": False,
            }
        )

    results = pd.DataFrame(rows)
    results.loc[results["logistic_weight"].eq(best_weight), "selected"] = True
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return best_weight, results

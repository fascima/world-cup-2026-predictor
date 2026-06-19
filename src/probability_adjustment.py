"""Probability post-processing for tournament-specific match context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


CLASS_LABELS = [0, 1, 2]
TARGET_COLUMN = "target"


@dataclass(frozen=True)
class WorldCupAdjustmentParams:
    """Tuned World Cup probability adjustment settings."""

    group_shrinkage: float = 0.0
    knockout_shrinkage: float = 0.0
    rotation_strength: float = 0.0
    rotation_draw_share: float = 0.5


ADJUSTMENT_GRID = [
    WorldCupAdjustmentParams(group, knockout, rotation, draw_share)
    for group in [0.0, 0.05, 0.10, 0.15, 0.20]
    for knockout in [0.0, 0.05, 0.10, 0.15]
    for rotation in [0.0, 0.05, 0.10, 0.15, 0.20]
    for draw_share in [0.25, 0.50]
]


def _column(rows: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    """Return a numeric column as an array, with a default when absent."""
    if column not in rows.columns:
        return np.full(len(rows), default, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce").fillna(default).to_numpy(dtype=float)


def _is_world_cup(rows: pd.DataFrame) -> np.ndarray:
    """Return a mask for FIFA World Cup rows."""
    if "tournament_type" in rows.columns:
        return rows["tournament_type"].astype(str).eq("world_cup").to_numpy()
    if "tournament" in rows.columns:
        return rows["tournament"].astype(str).eq("FIFA World Cup").to_numpy()
    return np.zeros(len(rows), dtype=bool)


def _normalize(probs: np.ndarray) -> np.ndarray:
    """Normalize rows to valid probability vectors."""
    clipped = np.clip(probs, 1e-12, None)
    return clipped / clipped.sum(axis=1, keepdims=True)


def apply_world_cup_adjustment(
    y_proba: np.ndarray,
    feature_rows: pd.DataFrame,
    params: WorldCupAdjustmentParams,
) -> np.ndarray:
    """Apply World Cup calibration and group-state motivation adjustments."""
    adjusted = np.asarray(y_proba, dtype=float).copy()
    if adjusted.size == 0:
        return adjusted

    world_cup_mask = _is_world_cup(feature_rows)
    group_mask = _column(feature_rows, "is_world_cup_group_stage") >= 0.5
    knockout_mask = world_cup_mask & ~group_mask
    group_mask = world_cup_mask & group_mask

    for mask, shrinkage in [
        (group_mask, params.group_shrinkage),
        (knockout_mask, params.knockout_shrinkage),
    ]:
        if shrinkage > 0 and np.any(mask):
            adjusted[mask] = (1.0 - shrinkage) * adjusted[mask] + shrinkage / len(CLASS_LABELS)

    rotation_strength = float(params.rotation_strength)
    if rotation_strength > 0:
        draw_share = float(params.rotation_draw_share)
        team_a_opportunity = (
            _column(feature_rows, "team_a_opponent_rotation_opportunity") >= 0.5
        )
        team_b_opportunity = (
            _column(feature_rows, "team_b_opponent_rotation_opportunity") >= 0.5
        )

        if np.any(team_a_opportunity):
            weights = np.ones((int(team_a_opportunity.sum()), len(CLASS_LABELS)), dtype=float)
            weights[:, 0] *= 1.0 + rotation_strength
            weights[:, 1] *= 1.0 + rotation_strength * draw_share
            weights[:, 2] *= max(0.05, 1.0 - rotation_strength)
            adjusted[team_a_opportunity] *= weights

        if np.any(team_b_opportunity):
            weights = np.ones((int(team_b_opportunity.sum()), len(CLASS_LABELS)), dtype=float)
            weights[:, 0] *= max(0.05, 1.0 - rotation_strength)
            weights[:, 1] *= 1.0 + rotation_strength * draw_share
            weights[:, 2] *= 1.0 + rotation_strength
            adjusted[team_b_opportunity] *= weights

    return _normalize(adjusted)


def tune_world_cup_adjustment(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    feature_rows: pd.DataFrame,
    output_prefix: str,
) -> tuple[WorldCupAdjustmentParams, pd.DataFrame]:
    """Tune World Cup adjustment settings by validation log loss."""
    rows: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_params = WorldCupAdjustmentParams()

    for params in ADJUSTMENT_GRID:
        adjusted = apply_world_cup_adjustment(y_proba, feature_rows, params)
        y_pred = np.argmax(adjusted, axis=1)
        validation_log_loss = float(log_loss(y_true, adjusted, labels=CLASS_LABELS))
        validation_accuracy = float(accuracy_score(y_true, y_pred))
        key = (validation_log_loss, -validation_accuracy)
        selected = best_key is None or key < best_key
        if selected:
            best_key = key
            best_params = params
        rows.append(
            {
                "group_shrinkage": params.group_shrinkage,
                "knockout_shrinkage": params.knockout_shrinkage,
                "rotation_strength": params.rotation_strength,
                "rotation_draw_share": params.rotation_draw_share,
                "validation_log_loss": validation_log_loss,
                "validation_accuracy": validation_accuracy,
                "selected": False,
            }
        )

    results = pd.DataFrame(rows)
    selected_mask = (
        results["group_shrinkage"].eq(best_params.group_shrinkage)
        & results["knockout_shrinkage"].eq(best_params.knockout_shrinkage)
        & results["rotation_strength"].eq(best_params.rotation_strength)
        & results["rotation_draw_share"].eq(best_params.rotation_draw_share)
    )
    results.loc[selected_mask, "selected"] = True
    results.to_csv(f"results/{output_prefix}_world_cup_adjustment_tuning.csv", index=False)
    return best_params, results


def validation_folds(features_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Return rolling World Cup folds plus a recent chronological fold."""
    folds: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    working = features_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    if "tournament_type" in working.columns:
        world_cup_rows = working[working["tournament_type"].astype(str).eq("world_cup")].copy()
    else:
        world_cup_rows = working[working["tournament"].astype(str).eq("FIFA World Cup")].copy()

    for year in sorted(world_cup_rows["date"].dt.year.unique()):
        validation_df = world_cup_rows[world_cup_rows["date"].dt.year == year].copy()
        if len(validation_df) < 20 or validation_df[TARGET_COLUMN].nunique() < 2:
            continue
        fold_train_df = working[working["date"] < validation_df["date"].min()].copy()
        if fold_train_df.empty or fold_train_df[TARGET_COLUMN].nunique() < 2:
            continue
        folds.append((f"world_cup_{int(year)}", fold_train_df, validation_df))

    recent_split = pd.Timestamp("2020-01-01")
    recent_train_df = working[working["date"] < recent_split].copy()
    recent_validation_df = working[
        (working["date"] >= recent_split) & (working["date"] < pd.Timestamp("2022-01-01"))
    ].copy()
    if (
        not recent_train_df.empty
        and not recent_validation_df.empty
        and recent_train_df[TARGET_COLUMN].nunique() >= 2
        and recent_validation_df[TARGET_COLUMN].nunique() >= 2
    ):
        folds.append(("recent_2020_2021", recent_train_df, recent_validation_df))

    return folds

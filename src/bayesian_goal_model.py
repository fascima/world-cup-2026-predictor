"""Regularized attack/defense goal model with W/D/L probability output."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml_logistic import CLASS_LABELS, TARGET_COLUMN, TRAIN_TEST_SPLIT_DATE


CATEGORICAL_COLUMNS = ["scoring_team", "opponent_team", "tournament_type"]
NUMERIC_COLUMNS = [
    "is_home",
    "is_neutral",
    "is_world_cup_group_stage",
    "is_world_cup_knockout",
    "scorer_recent_goals_for_5",
    "opponent_recent_goals_against_5",
    "scorer_recent_goals_for_10",
    "opponent_recent_goals_against_10",
    "scorer_points_last_5",
    "opponent_points_last_5",
    "effective_market_log_ratio",
    "injury_market_value_share_diff",
    "scorer_rotation_risk",
    "opponent_rotation_opportunity",
    "scorer_must_win",
    "opponent_must_win",
    "wc_prior_points_per_match_diff",
    "wc_prior_goal_diff_per_match_diff",
]
ALPHA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
DRAW_MULTIPLIER_GRID = [0.85, 0.95, 1.0, 1.1, 1.2, 1.35, 1.55]
DRAW_DECISION_THRESHOLDS = [0.24, 0.25, 0.26, 0.27, 0.28]
DRAW_DECISION_MAX_WIN_GAPS = [0.01, 0.02, 0.03, 0.04, 0.05]
DOMAIN_DRAW_DECISION_THRESHOLD = 0.26
DOMAIN_DRAW_DECISION_MAX_WIN_GAP = 0.02
DOMAIN_DRAW_DECISION_VALIDATION_TOLERANCE = 0.002
MAX_GOALS = 10
MARKET_GUARDRAIL_LOG_GAP = 0.75
MARKET_GUARDRAIL_TRANSFER = 0.30
WEAK_HOST_EXTRA_TRANSFER = 0.25
MARKET_GUARDRAIL_DRAW_SHARE = 0.15


def _normalize(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _poisson_pmf(lam: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    lam = max(float(lam), 1e-8)
    values = np.empty(max_goals + 1, dtype=float)
    values[0] = math.exp(-lam)
    for goals in range(1, max_goals + 1):
        values[goals] = values[goals - 1] * lam / goals
    return values


def _outcome_probs_from_lambdas(
    team_a_lambda: np.ndarray,
    team_b_lambda: np.ndarray,
    draw_multiplier: float = 1.0,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    rows = []
    for lam_a, lam_b in zip(team_a_lambda, team_b_lambda, strict=False):
        a_pmf = _poisson_pmf(float(lam_a), max_goals=max_goals)
        b_pmf = _poisson_pmf(float(lam_b), max_goals=max_goals)
        matrix = np.outer(a_pmf, b_pmf)
        matrix = matrix / matrix.sum()
        team_a_win = float(np.tril(matrix, k=-1).sum())
        draw = float(np.trace(matrix))
        team_b_win = float(np.triu(matrix, k=1).sum())
        rows.append([team_a_win, draw * float(draw_multiplier), team_b_win])
    return _normalize(np.asarray(rows, dtype=float))


def _apply_market_sanity_correction(matches: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
    """Prevent huge market mismatches from being overridden by model context alone."""
    adjusted = np.asarray(probs, dtype=float).copy()
    if "effective_market_value_log_ratio" not in matches.columns and "market_value_log_ratio" not in matches.columns:
        return adjusted

    for row_position, (_, match) in enumerate(matches.iterrows()):
        top = int(adjusted[row_position].argmax())
        if top == 1:
            continue

        log_ratio = _numeric(match, "effective_market_value_log_ratio", default=np.nan)
        if pd.isna(log_ratio):
            log_ratio = _numeric(match, "market_value_log_ratio", default=np.nan)
        if pd.isna(log_ratio):
            continue

        if log_ratio >= MARKET_GUARDRAIL_LOG_GAP:
            market_favorite = 0
        elif log_ratio <= -MARKET_GUARDRAIL_LOG_GAP:
            market_favorite = 2
        else:
            continue

        if market_favorite == top:
            continue

        transfer_fraction = MARKET_GUARDRAIL_TRANSFER
        weak_home_favorite = (
            top == 0
            and market_favorite == 2
            and bool(match.get("tournament", "") == "FIFA World Cup")
            and not bool(match.get("neutral", True))
        )
        if weak_home_favorite:
            transfer_fraction += WEAK_HOST_EXTRA_TRANSFER

        transfer = adjusted[row_position, top] * min(transfer_fraction, 0.75)
        adjusted[row_position, top] -= transfer
        adjusted[row_position, market_favorite] += transfer * (1.0 - MARKET_GUARDRAIL_DRAW_SHARE)
        adjusted[row_position, 1] += transfer * MARKET_GUARDRAIL_DRAW_SHARE

    return _normalize(adjusted)


def _draw_decision_predictions(
    probs: np.ndarray,
    threshold: float,
    max_win_gap: float,
) -> np.ndarray:
    pred = probs.argmax(axis=1)
    draw_mask = (probs[:, 1] >= threshold) & (np.abs(probs[:, 0] - probs[:, 2]) <= max_win_gap)
    pred[draw_mask] = 1
    return pred


def _numeric(row: pd.Series, column: str, default: float = 0.0) -> float:
    if column not in row or pd.isna(row[column]):
        return default
    return float(pd.to_numeric(row[column], errors="coerce"))


def _scoring_row(match: pd.Series, side: str) -> dict[str, Any]:
    """Create one scorer-oriented row for Team A or Team B."""
    if side not in {"team_a", "team_b"}:
        raise ValueError("side must be team_a or team_b")
    scorer = side
    opponent = "team_b" if side == "team_a" else "team_a"
    sign = 1.0 if side == "team_a" else -1.0

    return {
        "match_index": int(match.name),
        "date": match["date"],
        "scoring_team": str(match[scorer]),
        "opponent_team": str(match[opponent]),
        "tournament_type": str(match.get("tournament_type", "other")),
        "is_home": float(side == "team_a" and not bool(match.get("neutral", False))),
        "is_neutral": float(bool(match.get("neutral", False))),
        "is_world_cup_group_stage": _numeric(match, "is_world_cup_group_stage"),
        "is_world_cup_knockout": _numeric(match, "is_world_cup_knockout"),
        "scorer_recent_goals_for_5": _numeric(match, f"{scorer}_avg_goals_for_last_5", 1.0),
        "opponent_recent_goals_against_5": _numeric(match, f"{opponent}_avg_goals_against_last_5", 1.0),
        "scorer_recent_goals_for_10": _numeric(match, f"{scorer}_avg_goals_for_last_10", 1.0),
        "opponent_recent_goals_against_10": _numeric(match, f"{opponent}_avg_goals_against_last_10", 1.0),
        "scorer_points_last_5": _numeric(match, f"{scorer}_avg_points_last_5", 1.0),
        "opponent_points_last_5": _numeric(match, f"{opponent}_avg_points_last_5", 1.0),
        "effective_market_log_ratio": sign * _numeric(match, "effective_market_value_log_ratio"),
        "injury_market_value_share_diff": sign * _numeric(match, "injured_market_value_share_diff"),
        "scorer_rotation_risk": _numeric(match, f"{scorer}_rotation_risk"),
        "opponent_rotation_opportunity": _numeric(match, f"{scorer}_opponent_rotation_opportunity"),
        "scorer_must_win": _numeric(match, f"{scorer}_must_win"),
        "opponent_must_win": _numeric(match, f"{opponent}_must_win"),
        "wc_prior_points_per_match_diff": sign * _numeric(match, "wc_prior_points_per_match_diff"),
        "wc_prior_goal_diff_per_match_diff": sign * _numeric(match, "wc_prior_goal_diff_per_match_diff"),
        "goals": _numeric(match, f"{scorer}_score"),
    }


def _build_scoring_frame(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, match in matches.iterrows():
        match = match.copy()
        match.name = index
        rows.append(_scoring_row(match, "team_a"))
        rows.append(_scoring_row(match, "team_b"))
    return pd.DataFrame(rows)


def _build_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            NUMERIC_COLUMNS,
                        ),
                        (
                            "categorical",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            CATEGORICAL_COLUMNS,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            (
                "model",
                PoissonRegressor(
                    alpha=float(alpha),
                    max_iter=1000,
                ),
            ),
        ]
    )


def _fit_goal_model(matches: pd.DataFrame, alpha: float) -> Pipeline:
    scoring = _build_scoring_frame(matches)
    model = _build_pipeline(alpha)
    model.fit(scoring[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS], scoring["goals"])
    return model


def _predict_goal_rates(model: Pipeline, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scoring = _build_scoring_frame(matches)
    scoring["lambda"] = np.clip(
        model.predict(scoring[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]),
        0.05,
        6.0,
    )
    pivot = scoring.pivot(index="match_index", columns="scoring_team", values="lambda")
    team_a_lambdas = []
    team_b_lambdas = []
    for index, match in matches.iterrows():
        team_a_lambdas.append(float(pivot.loc[index, match["team_a"]]))
        team_b_lambdas.append(float(pivot.loc[index, match["team_b"]]))
    return np.asarray(team_a_lambdas, dtype=float), np.asarray(team_b_lambdas, dtype=float)


def _validation_folds(train_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    folds: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    world_cup_rows = train_df[train_df["tournament_type"].astype(str).eq("world_cup")].copy()
    for year in sorted(world_cup_rows["date"].dt.year.unique()):
        validation_df = world_cup_rows[world_cup_rows["date"].dt.year == year].copy()
        if len(validation_df) < 20 or validation_df[TARGET_COLUMN].nunique() < 2:
            continue
        fold_train_df = train_df[train_df["date"] < validation_df["date"].min()].copy()
        if fold_train_df.empty:
            continue
        folds.append((f"world_cup_{int(year)}", fold_train_df, validation_df))

    recent_train_df = train_df[train_df["date"] < pd.Timestamp("2020-01-01")].copy()
    recent_validation_df = train_df[
        (train_df["date"] >= pd.Timestamp("2020-01-01"))
        & (train_df["date"] < TRAIN_TEST_SPLIT_DATE)
    ].copy()
    if not recent_train_df.empty and not recent_validation_df.empty:
        folds.append(("recent_2020_2021", recent_train_df, recent_validation_df))
    return folds


def _tune_draw_decision_rule(
    validation_targets: np.ndarray,
    validation_probs: np.ndarray,
) -> tuple[float, float, pd.DataFrame]:
    """Tune a hard draw decision rule on validation accuracy."""
    rows = []
    baseline_pred = validation_probs.argmax(axis=1)
    baseline_accuracy = float(accuracy_score(validation_targets, baseline_pred))
    best_key = (baseline_accuracy, 0.0, 0.0)
    best_threshold = 1.0
    best_gap = 0.0
    rows.append(
        {
            "draw_decision_threshold": best_threshold,
            "draw_decision_max_win_gap": best_gap,
            "validation_accuracy": baseline_accuracy,
            "predicted_draws": int((baseline_pred == 1).sum()),
            "selected": False,
        }
    )

    for threshold in DRAW_DECISION_THRESHOLDS:
        for gap in DRAW_DECISION_MAX_WIN_GAPS:
            pred = _draw_decision_predictions(validation_probs, threshold, gap)
            accuracy = float(accuracy_score(validation_targets, pred))
            predicted_draws = int((pred == 1).sum())
            true_draws = int(((pred == 1) & (validation_targets == 1)).sum())
            key = (accuracy, float(true_draws), -float(predicted_draws))
            if key > best_key:
                best_key = key
                best_threshold = threshold
                best_gap = gap
            rows.append(
                {
                    "draw_decision_threshold": threshold,
                    "draw_decision_max_win_gap": gap,
                    "validation_accuracy": accuracy,
                    "predicted_draws": predicted_draws,
                    "selected": False,
                }
            )

    best_accuracy = max(row["validation_accuracy"] for row in rows)
    domain_rows = [
        row
        for row in rows
        if row["draw_decision_threshold"] == DOMAIN_DRAW_DECISION_THRESHOLD
        and row["draw_decision_max_win_gap"] == DOMAIN_DRAW_DECISION_MAX_WIN_GAP
    ]
    if domain_rows and domain_rows[0]["validation_accuracy"] >= (
        best_accuracy - DOMAIN_DRAW_DECISION_VALIDATION_TOLERANCE
    ):
        best_threshold = DOMAIN_DRAW_DECISION_THRESHOLD
        best_gap = DOMAIN_DRAW_DECISION_MAX_WIN_GAP

    results = pd.DataFrame(rows)
    results.loc[
        results["draw_decision_threshold"].eq(best_threshold)
        & results["draw_decision_max_win_gap"].eq(best_gap),
        "selected",
    ] = True
    return best_threshold, best_gap, results


def _tune_model(train_df: pd.DataFrame) -> tuple[float, float, float, float, pd.DataFrame]:
    folds = _validation_folds(train_df)
    if not folds:
        return 1.0, 1.0, 0.26, 0.02, pd.DataFrame()

    rows = []
    best_key: tuple[float, float] | None = None
    best_alpha = ALPHA_GRID[0]
    best_draw_multiplier = 1.0
    best_validation_targets = np.array([], dtype=int)
    best_validation_probs = np.empty((0, 3), dtype=float)

    for alpha in ALPHA_GRID:
        fold_predictions = []
        for fold_name, fold_train, fold_validation in folds:
            model = _fit_goal_model(fold_train, alpha)
            team_a_lambda, team_b_lambda = _predict_goal_rates(model, fold_validation)
            fold_predictions.append((fold_name, fold_validation, team_a_lambda, team_b_lambda))

        for draw_multiplier in DRAW_MULTIPLIER_GRID:
            fold_log_losses = []
            fold_accuracies = []
            pooled_targets = []
            pooled_probs = []
            for _, fold_validation, team_a_lambda, team_b_lambda in fold_predictions:
                probs = _outcome_probs_from_lambdas(
                    team_a_lambda,
                    team_b_lambda,
                    draw_multiplier=draw_multiplier,
                )
                probs = _apply_market_sanity_correction(fold_validation, probs)
                y_true = fold_validation[TARGET_COLUMN].astype(int)
                pooled_targets.append(y_true.to_numpy())
                pooled_probs.append(probs)
                fold_log_losses.append(float(log_loss(y_true, probs, labels=CLASS_LABELS)))
                fold_accuracies.append(float(accuracy_score(y_true, probs.argmax(axis=1))))
            validation_log_loss = float(np.mean(fold_log_losses))
            validation_accuracy = float(np.mean(fold_accuracies))
            key = (validation_log_loss, -validation_accuracy)
            if best_key is None or key < best_key:
                best_key = key
                best_alpha = alpha
                best_draw_multiplier = draw_multiplier
                best_validation_targets = np.concatenate(pooled_targets)
                best_validation_probs = np.vstack(pooled_probs)
            rows.append(
                {
                    "alpha": alpha,
                    "draw_multiplier": draw_multiplier,
                    "validation_log_loss": validation_log_loss,
                    "validation_accuracy": validation_accuracy,
                    "selected": False,
                }
            )

    tuning = pd.DataFrame(rows)
    tuning.loc[
        tuning["alpha"].eq(best_alpha) & tuning["draw_multiplier"].eq(best_draw_multiplier),
        "selected",
    ] = True
    decision_threshold, decision_gap, decision_tuning = _tune_draw_decision_rule(
        best_validation_targets,
        best_validation_probs,
    )
    if not decision_tuning.empty:
        decision_tuning["alpha"] = best_alpha
        decision_tuning["draw_multiplier"] = best_draw_multiplier
        decision_tuning = decision_tuning.rename(columns={"validation_accuracy": "decision_validation_accuracy"})
        tuning = pd.concat(
            [
                tuning,
                decision_tuning,
            ],
            ignore_index=True,
            sort=False,
        )
    return best_alpha, best_draw_multiplier, decision_threshold, decision_gap, tuning


def _prediction_rows(
    matches: pd.DataFrame,
    team_a_lambda: np.ndarray,
    team_b_lambda: np.ndarray,
    probs: np.ndarray,
    predicted_target: np.ndarray,
) -> pd.DataFrame:
    output = matches[["date", "team_a", "team_b", "tournament", "neutral", TARGET_COLUMN]].copy()
    output["team_a_expected_goals"] = team_a_lambda
    output["team_b_expected_goals"] = team_b_lambda
    output["team_a_win_prob"] = probs[:, 0]
    output["draw_prob"] = probs[:, 1]
    output["team_b_win_prob"] = probs[:, 2]
    output["top_probability_target"] = probs.argmax(axis=1)
    output["predicted_target"] = predicted_target
    return output


def train_bayesian_goal_model(
    features_df: pd.DataFrame,
    model_path: str = "models/bayesian_goal_model.joblib",
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """Train and backtest the regularized attack/defense goal model."""
    required = {"date", "team_a", "team_b", "team_a_score", "team_b_score", TARGET_COLUMN}
    missing = required - set(features_df.columns)
    if missing:
        raise ValueError(f"features_df missing required columns: {', '.join(sorted(missing))}")

    working = features_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date", TARGET_COLUMN, "team_a_score", "team_b_score"])
    working = working.sort_values("date").reset_index(drop=True)
    train_df = working[working["date"] < TRAIN_TEST_SPLIT_DATE].copy()
    test_df = working[working["date"] >= TRAIN_TEST_SPLIT_DATE].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Need train rows before 2022-01-01 and test rows from 2022 onward.")

    alpha, draw_multiplier, decision_threshold, decision_gap, tuning = _tune_model(train_df)
    model = _fit_goal_model(train_df, alpha)
    team_a_lambda, team_b_lambda = _predict_goal_rates(model, test_df)
    probs = _outcome_probs_from_lambdas(
        team_a_lambda,
        team_b_lambda,
        draw_multiplier=draw_multiplier,
    )
    probs = _apply_market_sanity_correction(test_df, probs)
    predicted_target = _draw_decision_predictions(probs, decision_threshold, decision_gap)
    predictions = _prediction_rows(test_df, team_a_lambda, team_b_lambda, probs, predicted_target)
    predictions.attrs["alpha"] = alpha
    predictions.attrs["draw_multiplier"] = draw_multiplier
    predictions.attrs["draw_decision_threshold"] = decision_threshold
    predictions.attrs["draw_decision_max_win_gap"] = decision_gap

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return model, predictions, tuning


def score_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """Return W/D/L metrics for Bayesian goal predictions."""
    probs = predictions[["team_a_win_prob", "draw_prob", "team_b_win_prob"]].to_numpy()
    y_true = predictions[TARGET_COLUMN].astype(int)
    y_pred = predictions["predicted_target"].astype(int)
    actual_draws = int(y_true.eq(1).sum())
    predicted_draws = int(y_pred.eq(1).sum())
    true_draws = int((y_true.eq(1) & y_pred.eq(1)).sum())
    return {
        "matches": float(len(predictions)),
        "log_loss": float(log_loss(y_true, probs, labels=CLASS_LABELS)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "predicted_draws": float(predicted_draws),
        "actual_draws": float(actual_draws),
        "draw_recall": float(true_draws / actual_draws) if actual_draws else 0.0,
        "draw_precision": float(true_draws / predicted_draws) if predicted_draws else 0.0,
    }

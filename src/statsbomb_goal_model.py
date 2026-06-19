"""StatsBomb-informed goal model experiment."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import score_prediction_rows
from src.config import POISSON_MAX_EXPECTED_GOALS, POISSON_MAX_GOALS, POISSON_MIN_EXPECTED_GOALS
from src.poisson_backtest import (
    _actual_outcome,
    _build_elo_calibration_and_scored,
    _decision_outcome,
    _top_probability_outcome,
    outcome_diagnostics,
)
from src.poisson_model import (
    apply_dixon_coles_adjustment,
    expected_goals_from_elo_expected,
    fit_average_total_goals,
    fit_goal_profile_state,
    goal_profile_multipliers,
    independent_poisson_score_matrix,
    outcome_probabilities_from_score_matrix,
    total_goals_for_elo_gap,
    update_goal_profile_state,
)
from src.statsbomb_features import (
    build_statsbomb_pair_features,
    statsbomb_metric_defaults,
    update_statsbomb_histories_until,
)
from src.utils import normalize_probabilities


@dataclass(frozen=True)
class StatsBombGoalParams:
    """Settings for xG-prior lambda adjustment."""

    xg_strength: float = 0.0
    total_goal_strength: float = 0.0
    dixon_coles_rho: float = -0.08
    statsbomb_window: int = 5
    goal_profile_weight: float = 0.35


PARAMETER_GRID = [
    StatsBombGoalParams(xg_strength, total_strength, rho, window, goal_weight)
    for xg_strength in [0.0, 0.25, 0.5, 0.75, 1.0]
    for total_strength in [0.0, 0.25, 0.5]
    for rho in [-0.12, -0.08, -0.04, 0.0]
    for window in [3, 5]
    for goal_weight in [0.2, 0.35]
]


def _safe_log_ratio(value: float, baseline: float) -> float:
    value = max(0.05, float(value))
    baseline = max(0.05, float(baseline))
    return math.log(value / baseline)


def _clamp_lambda(value: float) -> float:
    return max(POISSON_MIN_EXPECTED_GOALS, min(POISSON_MAX_EXPECTED_GOALS, float(value)))


def _adjust_lambdas_with_statsbomb(
    home_lambda: float,
    away_lambda: float,
    home_team: str,
    away_team: str,
    statsbomb_histories: dict[str, deque[dict[str, float]]],
    statsbomb_defaults: dict[str, float],
    average_team_goals: float,
    params: StatsBombGoalParams,
) -> tuple[float, float, dict[str, float]]:
    """Adjust expected goals from rolling xG priors known before the match."""
    if params.xg_strength == 0.0 and params.total_goal_strength == 0.0:
        return float(home_lambda), float(away_lambda), {
            "statsbomb_reliability": 0.0,
            "statsbomb_home_log_adjustment": 0.0,
            "statsbomb_away_log_adjustment": 0.0,
            "statsbomb_total_log_adjustment": 0.0,
        }

    features = build_statsbomb_pair_features(
        home_team,
        away_team,
        statsbomb_histories,
        statsbomb_defaults,
    )
    home_matches = float(features["team_a_statsbomb_matches_before"])
    away_matches = float(features["team_b_statsbomb_matches_before"])
    reliability = min(1.0, (home_matches + away_matches) / 6.0)
    if reliability <= 0.0:
        return float(home_lambda), float(away_lambda), {
            "statsbomb_reliability": 0.0,
            "statsbomb_home_log_adjustment": 0.0,
            "statsbomb_away_log_adjustment": 0.0,
            "statsbomb_total_log_adjustment": 0.0,
        }

    window = int(params.statsbomb_window)
    avg = max(0.1, float(average_team_goals))
    home_xg_for = float(features[f"team_a_statsbomb_xg_for_last_{window}"])
    home_xg_against = float(features[f"team_a_statsbomb_xg_against_last_{window}"])
    away_xg_for = float(features[f"team_b_statsbomb_xg_for_last_{window}"])
    away_xg_against = float(features[f"team_b_statsbomb_xg_against_last_{window}"])

    home_attack_log = _safe_log_ratio(home_xg_for, avg)
    home_target_defense_log = _safe_log_ratio(away_xg_against, avg)
    away_attack_log = _safe_log_ratio(away_xg_for, avg)
    away_target_defense_log = _safe_log_ratio(home_xg_against, avg)
    raw_home_log = 0.5 * home_attack_log + 0.5 * home_target_defense_log
    raw_away_log = 0.5 * away_attack_log + 0.5 * away_target_defense_log

    center_log = 0.5 * (raw_home_log + raw_away_log)
    home_relative_log = raw_home_log - center_log
    away_relative_log = raw_away_log - center_log
    total_signal = (home_xg_for + home_xg_against + away_xg_for + away_xg_against) / (4.0 * avg)
    total_log = _safe_log_ratio(total_signal, 1.0)

    max_adjustment = 0.45
    home_adjustment = max(
        -max_adjustment,
        min(max_adjustment, reliability * params.xg_strength * home_relative_log),
    )
    away_adjustment = max(
        -max_adjustment,
        min(max_adjustment, reliability * params.xg_strength * away_relative_log),
    )
    total_adjustment = max(
        -max_adjustment,
        min(max_adjustment, reliability * params.total_goal_strength * total_log),
    )

    adjusted_home = _clamp_lambda(float(home_lambda) * math.exp(home_adjustment + total_adjustment))
    adjusted_away = _clamp_lambda(float(away_lambda) * math.exp(away_adjustment + total_adjustment))
    return adjusted_home, adjusted_away, {
        "statsbomb_reliability": reliability,
        "statsbomb_home_log_adjustment": home_adjustment,
        "statsbomb_away_log_adjustment": away_adjustment,
        "statsbomb_total_log_adjustment": total_adjustment,
        "statsbomb_home_xg_for": home_xg_for,
        "statsbomb_home_xg_against": home_xg_against,
        "statsbomb_away_xg_for": away_xg_for,
        "statsbomb_away_xg_against": away_xg_against,
    }


def _prepare_statsbomb_rows(statsbomb_team_match_features: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, float]]:
    if statsbomb_team_match_features is None or statsbomb_team_match_features.empty:
        return pd.DataFrame(), {}
    rows = statsbomb_team_match_features.copy()
    rows["match_date"] = pd.to_datetime(rows["match_date"], errors="coerce")
    rows = rows.dropna(subset=["match_date", "team"]).sort_values(
        ["match_date", "match_id", "team"],
    ).reset_index(drop=True)
    return rows, statsbomb_metric_defaults(rows)


def _initial_statsbomb_state(
    statsbomb_rows: pd.DataFrame,
    start_date: pd.Timestamp,
) -> tuple[dict[str, deque[dict[str, float]]], int]:
    histories: dict[str, deque[dict[str, float]]] = defaultdict(deque)
    if statsbomb_rows.empty:
        return histories, 0
    next_index = update_statsbomb_histories_until(statsbomb_rows, histories, 0, start_date)
    return histories, next_index


def _predict_scored_matches(
    scored: pd.DataFrame,
    training_matches: pd.DataFrame,
    average_total_goals: float,
    statsbomb_rows: pd.DataFrame,
    statsbomb_defaults: dict[str, float],
    params: StatsBombGoalParams,
) -> pd.DataFrame:
    goal_profile_state = fit_goal_profile_state(training_matches)
    if not scored.empty:
        statsbomb_histories, statsbomb_index = _initial_statsbomb_state(
            statsbomb_rows,
            pd.Timestamp(scored["date"].min()),
        )
    else:
        statsbomb_histories, statsbomb_index = defaultdict(deque), 0
    prediction_rows: list[dict[str, Any]] = []

    for _, row in scored.sort_values("date").iterrows():
        match_date = pd.Timestamp(row["date"])
        if not statsbomb_rows.empty:
            statsbomb_index = update_statsbomb_histories_until(
                statsbomb_rows,
                statsbomb_histories,
                statsbomb_index,
                match_date,
            )

        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        adjusted_elo_diff = float(row["adjusted_elo_diff"])
        total_goals = total_goals_for_elo_gap(average_total_goals, adjusted_elo_diff)
        average_team_goals = total_goals / 2.0
        home_multiplier, away_multiplier = goal_profile_multipliers(
            goal_profile_state,
            home_team,
            away_team,
            average_team_goals,
            profile_weight=params.goal_profile_weight,
        )
        base_home_lambda, base_away_lambda = expected_goals_from_elo_expected(
            float(row["expected_home"]),
            total_goals,
            home_goal_multiplier=home_multiplier,
            away_goal_multiplier=away_multiplier,
        )
        home_lambda, away_lambda, diagnostics = _adjust_lambdas_with_statsbomb(
            base_home_lambda,
            base_away_lambda,
            home_team,
            away_team,
            statsbomb_histories,
            statsbomb_defaults,
            average_team_goals,
            params,
        )
        score_matrix = independent_poisson_score_matrix(
            home_lambda,
            away_lambda,
            max_goals=POISSON_MAX_GOALS,
        )
        score_matrix = apply_dixon_coles_adjustment(
            score_matrix,
            home_lambda,
            away_lambda,
            params.dixon_coles_rho,
        )
        home_win_prob, draw_prob, away_win_prob = outcome_probabilities_from_score_matrix(score_matrix)
        home_win_prob, draw_prob, away_win_prob = normalize_probabilities(
            [home_win_prob, draw_prob, away_win_prob]
        )

        prediction_row = row.to_dict()
        prediction_row.update(
            {
                "base_home_expected_goals": base_home_lambda,
                "base_away_expected_goals": base_away_lambda,
                "home_expected_goals": home_lambda,
                "away_expected_goals": away_lambda,
                "total_expected_goals": home_lambda + away_lambda,
                "home_goal_profile_multiplier": home_multiplier,
                "away_goal_profile_multiplier": away_multiplier,
                "home_win_prob": home_win_prob,
                "draw_prob": draw_prob,
                "away_win_prob": away_win_prob,
                "xg_strength": params.xg_strength,
                "total_goal_strength": params.total_goal_strength,
                "dixon_coles_rho": params.dixon_coles_rho,
                "statsbomb_window": params.statsbomb_window,
                "goal_profile_weight": params.goal_profile_weight,
                **diagnostics,
            }
        )
        prediction_rows.append(prediction_row)

        update_goal_profile_state(
            goal_profile_state,
            home_team,
            away_team,
            int(row["home_score"]),
            int(row["away_score"]),
        )

    predictions = pd.DataFrame(prediction_rows)
    predictions["actual_outcome"] = predictions.apply(_actual_outcome, axis=1)
    predictions["top_probability_outcome"] = predictions.apply(_top_probability_outcome, axis=1)
    predictions["decision_outcome"] = predictions.apply(_decision_outcome, axis=1)
    return predictions


def _metrics_for_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    metrics = score_prediction_rows(predictions)
    metrics["top_probability_accuracy"] = metrics["accuracy"]
    metrics.update(outcome_diagnostics(predictions))
    return metrics


def tune_statsbomb_goal_params(
    matches: pd.DataFrame,
    statsbomb_team_match_features: pd.DataFrame | None,
    validation_year: int = 2018,
    output_path: str = "results/statsbomb_goal_model_tuning_results.csv",
) -> tuple[StatsBombGoalParams, pd.DataFrame]:
    """Tune the StatsBomb goal model on one pre-holdout World Cup."""
    validation_matches = matches[
        (matches["date"].dt.year == validation_year)
        & (matches["tournament"].astype(str).eq("FIFA World Cup"))
    ].copy()
    if validation_matches.empty:
        fallback = StatsBombGoalParams()
        rows = pd.DataFrame([{**fallback.__dict__, "selected": True}])
        rows.to_csv(output_path, index=False)
        return fallback, rows

    training_matches = matches[matches["date"] < validation_matches["date"].min()].copy()
    calibration, scored = _build_elo_calibration_and_scored(training_matches, validation_matches)
    average_total_goals = fit_average_total_goals(calibration if not calibration.empty else training_matches)
    statsbomb_rows, statsbomb_defaults = _prepare_statsbomb_rows(statsbomb_team_match_features)

    rows: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_params = PARAMETER_GRID[0]

    for params in PARAMETER_GRID:
        predictions = _predict_scored_matches(
            scored,
            training_matches,
            average_total_goals,
            statsbomb_rows,
            statsbomb_defaults,
            params,
        )
        metrics = _metrics_for_predictions(predictions)
        key = (metrics["log_loss"], -metrics["accuracy"])
        if best_key is None or key < best_key:
            best_key = key
            best_params = params
        rows.append({**params.__dict__, **metrics, "selected": False})

    results = pd.DataFrame(rows).sort_values(["log_loss", "brier_score"]).reset_index(drop=True)
    selected_mask = (
        results["xg_strength"].eq(best_params.xg_strength)
        & results["total_goal_strength"].eq(best_params.total_goal_strength)
        & results["dixon_coles_rho"].eq(best_params.dixon_coles_rho)
        & results["statsbomb_window"].eq(best_params.statsbomb_window)
        & results["goal_profile_weight"].eq(best_params.goal_profile_weight)
    )
    results.loc[selected_mask, "selected"] = True
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return best_params, results


def run_statsbomb_goal_backtest(
    matches: pd.DataFrame,
    statsbomb_team_match_features: pd.DataFrame | None,
    params: StatsBombGoalParams,
    start_year: int = 2022,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Backtest the StatsBomb-informed goal model from ``start_year`` onward."""
    training_matches = matches[matches["date"].dt.year < start_year].copy()
    holdout_matches = matches[matches["date"].dt.year >= start_year].copy()
    if holdout_matches.empty:
        raise ValueError(f"No matches available for StatsBomb goal backtest from {start_year} onward.")

    calibration, scored = _build_elo_calibration_and_scored(training_matches, holdout_matches)
    average_total_goals = fit_average_total_goals(calibration if not calibration.empty else training_matches)
    statsbomb_rows, statsbomb_defaults = _prepare_statsbomb_rows(statsbomb_team_match_features)
    predictions = _predict_scored_matches(
        scored,
        training_matches,
        average_total_goals,
        statsbomb_rows,
        statsbomb_defaults,
        params,
    )
    metrics = _metrics_for_predictions(predictions)
    metrics["start_year"] = float(start_year)
    metrics["average_total_goals"] = float(average_total_goals)
    metrics.update(params.__dict__)
    return metrics, predictions

"""Rolling-window validation for the separate Poisson goal model."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from src.data_loader import clean_results, load_results
from src.poisson_backtest import _build_elo_calibration_and_scored
from src.poisson_model import fit_average_total_goals
from src.utils import ensure_directories
from poisson_tuning import (
    DECISION_MAX_WIN_PROB_GAP_GRID,
    DECISION_THRESHOLD_GRID,
    DIXON_COLES_RHO_GRID,
    DRAW_INFLATION_GRID,
    GOAL_PROFILE_WEIGHT_GRID,
    _score_parameters,
)


RESULTS_PATH = Path("data/raw/results.csv")
ROLLING_OUTPUT_PATH = Path("results/poisson_rolling_validation_results.csv")
ROLLING_FOLD_OUTPUT_PATH = Path("results/poisson_rolling_validation_folds.csv")


VALIDATION_WINDOWS = [
    (2018, 2019),
    (2020, 2021),
    (2022, 2023),
    (2024, 2026),
]


def _weighted_average(rows: list[dict[str, float]], column: str) -> float:
    total_matches = sum(row["matches"] for row in rows)
    if total_matches <= 0:
        return 0.0
    return sum(row[column] * row["matches"] for row in rows) / total_matches


def _aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate fold metrics into one weighted result row."""
    total_matches = sum(row["matches"] for row in rows)
    total_predicted_draws = sum(row["predicted_draws"] for row in rows)
    total_actual_draws = sum(row["actual_draws"] for row in rows)
    true_draw_predictions = sum(row["draw_recall"] * row["actual_draws"] for row in rows)
    return {
        "matches": total_matches,
        "log_loss": _weighted_average(rows, "log_loss"),
        "brier_score": _weighted_average(rows, "brier_score"),
        "top_probability_accuracy": _weighted_average(rows, "top_probability_accuracy"),
        "decision_accuracy": _weighted_average(rows, "decision_accuracy"),
        "predicted_draws": total_predicted_draws,
        "actual_draws": total_actual_draws,
        "draw_recall": true_draw_predictions / total_actual_draws if total_actual_draws else 0.0,
        "draw_precision": true_draw_predictions / total_predicted_draws if total_predicted_draws else 0.0,
        "non_draw_accuracy": _weighted_average(rows, "non_draw_accuracy"),
    }


def _precompute_folds(matches: pd.DataFrame) -> list[dict[str, object]]:
    """Build reusable Elo rows for each rolling validation window."""
    folds: list[dict[str, object]] = []
    for start_year, end_year in VALIDATION_WINDOWS:
        training_matches = matches[matches["date"].dt.year < start_year].copy()
        validation_matches = matches[
            (matches["date"].dt.year >= start_year)
            & (matches["date"].dt.year <= end_year)
        ].copy()
        if validation_matches.empty:
            continue

        calibration, scored_template = _build_elo_calibration_and_scored(
            training_matches,
            validation_matches,
        )
        average_total_goals = fit_average_total_goals(
            calibration if not calibration.empty else training_matches
        )
        folds.append(
            {
                "start_year": start_year,
                "end_year": end_year,
                "training_matches": training_matches,
                "scored_template": scored_template,
                "average_total_goals": average_total_goals,
            }
        )
    return folds


def main() -> int:
    """Run rolling-window validation over the Poisson tuning grid."""
    ensure_directories()
    matches = clean_results(load_results(str(RESULTS_PATH)))
    folds = _precompute_folds(matches)
    if not folds:
        print("No rolling-validation folds could be built.")
        return 1

    result_rows = []
    fold_rows = []
    for draw_inflation, goal_profile_weight, dixon_coles_rho, decision_threshold, decision_gap in product(
        DRAW_INFLATION_GRID,
        GOAL_PROFILE_WEIGHT_GRID,
        DIXON_COLES_RHO_GRID,
        DECISION_THRESHOLD_GRID,
        DECISION_MAX_WIN_PROB_GAP_GRID,
    ):
        parameter_fold_rows = []
        for fold in folds:
            metrics = _score_parameters(
                fold["scored_template"],
                fold["training_matches"],
                float(fold["average_total_goals"]),
                draw_inflation,
                goal_profile_weight,
                dixon_coles_rho,
                decision_threshold,
                decision_gap,
            )
            fold_row = {
                "start_year": fold["start_year"],
                "end_year": fold["end_year"],
                "draw_inflation": draw_inflation,
                "goal_profile_weight": goal_profile_weight,
                "dixon_coles_rho": dixon_coles_rho,
                "decision_threshold": decision_threshold,
                "decision_max_win_prob_gap": decision_gap,
            }
            fold_row.update(metrics)
            fold_rows.append(fold_row)
            parameter_fold_rows.append(metrics)

        aggregate = _aggregate_metrics(parameter_fold_rows)
        result_row = {
            "draw_inflation": draw_inflation,
            "goal_profile_weight": goal_profile_weight,
            "dixon_coles_rho": dixon_coles_rho,
            "decision_threshold": decision_threshold,
            "decision_max_win_prob_gap": decision_gap,
        }
        result_row.update(aggregate)
        result_rows.append(result_row)

    results = pd.DataFrame(result_rows)
    results = results.sort_values(["top_probability_accuracy", "decision_accuracy"], ascending=False)
    results.to_csv(ROLLING_OUTPUT_PATH, index=False)
    pd.DataFrame(fold_rows).to_csv(ROLLING_FOLD_OUTPUT_PATH, index=False)

    print("Best rolling-validation settings by top-pick accuracy:")
    print(
        results.head(10)[
            [
                "draw_inflation",
                "goal_profile_weight",
                "dixon_coles_rho",
                "decision_threshold",
                "decision_max_win_prob_gap",
                "log_loss",
                "brier_score",
                "top_probability_accuracy",
                "decision_accuracy",
                "predicted_draws",
                "draw_recall",
                "draw_precision",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote {ROLLING_OUTPUT_PATH}")
    print(f"Wrote {ROLLING_FOLD_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

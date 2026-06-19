"""Grid-search tuning for the separate Poisson goal model."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from src.config import BACKTEST_START_YEAR
from src.backtest import score_prediction_rows
from src.data_loader import clean_results, load_results
from src.poisson_backtest import (
    _actual_outcome,
    _build_elo_calibration_and_scored,
    _decision_outcome,
    _top_probability_outcome,
    outcome_diagnostics,
)
from src.poisson_model import (
    fit_average_total_goals,
    fit_goal_profile_state,
    poisson_outcome_probabilities,
    update_goal_profile_state,
)
from src.utils import ensure_directories


RESULTS_PATH = Path("data/raw/results.csv")
TUNING_OUTPUT_PATH = Path("results/poisson_tuning_results.csv")


DRAW_INFLATION_GRID = [1.2, 1.35, 1.5, 1.65]
GOAL_PROFILE_WEIGHT_GRID = [0.0, 0.2, 0.35, 0.5]
DECISION_THRESHOLD_GRID = [0.28, 0.30]
DECISION_MAX_WIN_PROB_GAP_GRID = [0.12, 0.15]
DIXON_COLES_RHO_GRID = [-0.12, -0.08, -0.04, 0.0]


def _score_parameters(
    scored_template: pd.DataFrame,
    training_matches: pd.DataFrame,
    average_total_goals: float,
    draw_inflation: float,
    goal_profile_weight: float,
    dixon_coles_rho: float,
    decision_threshold: float,
    decision_gap: float,
) -> dict[str, float]:
    """Score one Poisson parameter set using precomputed Elo rows."""
    goal_profile_state = fit_goal_profile_state(training_matches)
    rows = []

    for _, row in scored_template.sort_values("date").iterrows():
        prediction = poisson_outcome_probabilities(
            float(row["expected_home"]),
            average_total_goals,
            adjusted_elo_diff=float(row["adjusted_elo_diff"]),
            goal_profile_state=goal_profile_state,
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
            draw_inflation=draw_inflation,
            goal_profile_weight=goal_profile_weight,
            dixon_coles_rho=dixon_coles_rho,
        )
        prediction_row = row.to_dict()
        prediction_row.update(prediction)
        rows.append(prediction_row)
        update_goal_profile_state(
            goal_profile_state,
            str(row["home_team"]),
            str(row["away_team"]),
            int(row["home_score"]),
            int(row["away_score"]),
        )

    scored = pd.DataFrame(rows)
    scored["actual_outcome"] = scored.apply(_actual_outcome, axis=1)
    scored["top_probability_outcome"] = scored.apply(_top_probability_outcome, axis=1)
    scored["decision_outcome"] = scored.apply(
        lambda row: _decision_outcome(
            row,
            decision_threshold=decision_threshold,
            decision_max_win_prob_gap=decision_gap,
        ),
        axis=1,
    )
    metrics = score_prediction_rows(scored)
    metrics["top_probability_accuracy"] = metrics["accuracy"]
    metrics.update(
        outcome_diagnostics(
            scored,
            decision_threshold=decision_threshold,
            decision_max_win_prob_gap=decision_gap,
        )
    )
    return metrics


def main() -> int:
    """Run a compact grid search over separate Poisson settings."""
    ensure_directories()
    matches = clean_results(load_results(str(RESULTS_PATH)))
    training_matches = matches[matches["date"].dt.year < BACKTEST_START_YEAR].copy()
    holdout_matches = matches[matches["date"].dt.year >= BACKTEST_START_YEAR].copy()
    calibration, scored_template = _build_elo_calibration_and_scored(training_matches, holdout_matches)
    average_total_goals = fit_average_total_goals(calibration if not calibration.empty else training_matches)
    rows = []

    for draw_inflation, goal_profile_weight, dixon_coles_rho, decision_threshold, decision_gap in product(
        DRAW_INFLATION_GRID,
        GOAL_PROFILE_WEIGHT_GRID,
        DIXON_COLES_RHO_GRID,
        DECISION_THRESHOLD_GRID,
        DECISION_MAX_WIN_PROB_GAP_GRID,
    ):
        metrics = _score_parameters(
            scored_template,
            training_matches,
            average_total_goals,
            draw_inflation,
            goal_profile_weight,
            dixon_coles_rho,
            decision_threshold,
            decision_gap,
        )
        row = {
            "draw_inflation": draw_inflation,
            "goal_profile_weight": goal_profile_weight,
            "dixon_coles_rho": dixon_coles_rho,
            "decision_threshold": decision_threshold,
            "decision_max_win_prob_gap": decision_gap,
        }
        row.update(metrics)
        rows.append(row)
        print(
            "tested "
            f"draw={draw_inflation}, goal_weight={goal_profile_weight}, "
            f"rho={dixon_coles_rho}, "
            f"threshold={decision_threshold}, gap={decision_gap}: "
            f"log_loss={metrics['log_loss']:.6f}, "
            f"top_accuracy={metrics['top_probability_accuracy']:.6f}, "
            f"decision_accuracy={metrics['decision_accuracy']:.6f}"
        )

    results = pd.DataFrame(rows).sort_values(["log_loss", "brier_score"]).reset_index(drop=True)
    results.to_csv(TUNING_OUTPUT_PATH, index=False)

    print("Best by log loss:")
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
    print("Best by top-probability accuracy:")
    print(
        results.sort_values("top_probability_accuracy", ascending=False).head(10)[
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
    print("Best by decision-rule accuracy:")
    print(
        results.sort_values("decision_accuracy", ascending=False).head(10)[
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
    print(f"Wrote {TUNING_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare Basic Poisson, Dixon-Coles, and Enhanced Dixon-Coles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BACKTEST_START_YEAR,
    DIXON_COLES_RHO,
    POISSON_GOAL_PROFILE_WEIGHT,
)
from src.data_loader import clean_results, load_results
from src.enhanced_dixon_coles import (
    fit_enhanced_feature_state,
    predict_enhanced_dixon_coles_match,
    update_enhanced_feature_state,
)
from src.market_value import load_historical_world_cup_market_values
from src.poisson_backtest import _build_elo_calibration_and_scored, _score_prediction_frame
from src.poisson_model import (
    fit_average_total_goals,
    fit_goal_profile_state,
    poisson_outcome_probabilities,
    update_goal_profile_state,
)
from src.utils import ensure_directories


RESULTS_PATH = Path("data/raw/results.csv")
HISTORICAL_MARKET_VALUES_PATH = Path("data/fixtures/historical_world_cup_market_values.csv")
PREDICTIONS_OUTPUT_PATH = Path("results/enhanced_dixon_coles_predictions.csv")
SUMMARY_OUTPUT_PATH = Path("results/enhanced_dixon_coles_summary.csv")


def _prefixed(prediction: dict[str, object], prefix: str) -> dict[str, object]:
    """Return prediction values with a model prefix, excluding large matrices."""
    prefixed: dict[str, object] = {}
    for key, value in prediction.items():
        if isinstance(value, np.ndarray):
            continue
        prefixed[f"{prefix}{key}"] = value
    return prefixed


def _run_comparison_backtest(
    matches: pd.DataFrame,
    start_year: int = BACKTEST_START_YEAR,
    market_values_by_year: dict[int, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest all Poisson-family models on the same chronological rows."""
    training_matches = matches[matches["date"].dt.year < start_year].copy()
    holdout_matches = matches[matches["date"].dt.year >= start_year].copy()
    if holdout_matches.empty:
        raise ValueError(f"No matches available for enhanced Dixon-Coles backtest from {start_year} onward.")

    calibration, scored = _build_elo_calibration_and_scored(training_matches, holdout_matches)
    average_total_goals = fit_average_total_goals(calibration if not calibration.empty else training_matches)
    goal_profile_state = fit_goal_profile_state(training_matches)
    enhanced_feature_state = fit_enhanced_feature_state(training_matches)
    prediction_rows: list[dict[str, object]] = []

    for _, row in scored.sort_values("date").iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        adjusted_elo_diff = float(row["adjusted_elo_diff"])
        expected_home = float(row["expected_home"])

        basic_poisson = poisson_outcome_probabilities(
            expected_home,
            average_total_goals,
            adjusted_elo_diff=adjusted_elo_diff,
            goal_profile_state=goal_profile_state,
            home_team=home_team,
            away_team=away_team,
            draw_inflation=1.0,
            goal_profile_weight=POISSON_GOAL_PROFILE_WEIGHT,
            use_dixon_coles=False,
            dixon_coles_rho=0.0,
        )
        dixon_coles = poisson_outcome_probabilities(
            expected_home,
            average_total_goals,
            adjusted_elo_diff=adjusted_elo_diff,
            goal_profile_state=goal_profile_state,
            home_team=home_team,
            away_team=away_team,
            draw_inflation=1.0,
            goal_profile_weight=POISSON_GOAL_PROFILE_WEIGHT,
            use_dixon_coles=True,
            dixon_coles_rho=DIXON_COLES_RHO,
        )
        enhanced_dixon_coles = predict_enhanced_dixon_coles_match(
            expected_home,
            average_total_goals,
            adjusted_elo_diff,
            enhanced_feature_state,
            home_team,
            away_team,
            pd.Timestamp(row["date"]),
            tournament=str(row.get("tournament", "")),
            neutral=bool(row.get("neutral", True)),
            goal_profile_state=goal_profile_state,
            market_values_by_year=market_values_by_year,
            goal_profile_weight=POISSON_GOAL_PROFILE_WEIGHT,
            dixon_coles_rho=DIXON_COLES_RHO,
            include_score_matrix=False,
        )

        prediction_row = row.to_dict()
        prediction_row.update(_prefixed(basic_poisson, "basic_poisson_"))
        prediction_row.update(_prefixed(dixon_coles, "dixon_coles_"))
        prediction_row.update(_prefixed(enhanced_dixon_coles, "enhanced_dc_"))
        prediction_rows.append(prediction_row)

        update_goal_profile_state(
            goal_profile_state,
            home_team,
            away_team,
            int(row["home_score"]),
            int(row["away_score"]),
        )
        update_enhanced_feature_state(
            enhanced_feature_state,
            home_team,
            away_team,
            int(row["home_score"]),
            int(row["away_score"]),
            pd.Timestamp(row["date"]),
        )

    predictions = pd.DataFrame(prediction_rows)
    summaries = []
    for model_name, prefix in [
        ("basic_poisson", "basic_poisson_"),
        ("dixon_coles", "dixon_coles_"),
        ("enhanced_dixon_coles", "enhanced_dc_"),
    ]:
        metrics = _score_prediction_frame(predictions, prefix=prefix)
        metrics["model"] = model_name
        metrics["start_year"] = float(start_year)
        metrics["average_total_goals"] = float(average_total_goals)
        if model_name != "basic_poisson":
            metrics["dixon_coles_rho"] = float(DIXON_COLES_RHO)
        else:
            metrics["dixon_coles_rho"] = 0.0
        summaries.append(metrics)

    summary = pd.DataFrame(summaries)
    ordered_columns = ["model"] + [column for column in summary.columns if column != "model"]
    return summary[ordered_columns], predictions


def main() -> int:
    """Load data, run the enhanced Dixon-Coles comparison, and save outputs."""
    ensure_directories()
    matches = clean_results(load_results(str(RESULTS_PATH)))
    market_values_by_year = load_historical_world_cup_market_values(HISTORICAL_MARKET_VALUES_PATH)
    summary, predictions = _run_comparison_backtest(
        matches,
        start_year=BACKTEST_START_YEAR,
        market_values_by_year=market_values_by_year,
    )
    predictions.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    if market_values_by_year:
        print(f"Loaded historical market values for years: {', '.join(map(str, sorted(market_values_by_year)))}")
    else:
        print("No historical market values found; enhanced market-value feature is neutral.")
    print("Poisson-family comparison metrics:")
    display_columns = [
        "model",
        "matches",
        "log_loss",
        "brier_score",
        "accuracy",
        "decision_accuracy",
        "draw_recall",
        "non_draw_accuracy",
    ]
    print(summary[display_columns].to_string(index=False))
    print(f"Wrote {PREDICTIONS_OUTPUT_PATH}")
    print(f"Wrote {SUMMARY_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the StatsBomb-informed goal-model experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import clean_results, load_results
from src.statsbomb_features import load_statsbomb_team_match_features
from src.statsbomb_goal_model import (
    run_statsbomb_goal_backtest,
    tune_statsbomb_goal_params,
)
from src.utils import ensure_directories


RESULTS_PATH = Path("data/raw/results.csv")
STATSBOMB_FEATURES_PATH = Path("data/processed/statsbomb_team_match_features.csv")
PREDICTIONS_OUTPUT_PATH = Path("results/statsbomb_goal_model_predictions.csv")
METRICS_OUTPUT_PATH = Path("results/statsbomb_goal_model_metrics.csv")


def _subset_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    from src.backtest import score_prediction_rows
    from src.poisson_backtest import outcome_diagnostics

    metrics = score_prediction_rows(predictions)
    metrics.update(outcome_diagnostics(predictions))
    return metrics


def main() -> int:
    ensure_directories()
    matches = clean_results(load_results(str(RESULTS_PATH)))
    statsbomb_features = (
        load_statsbomb_team_match_features(STATSBOMB_FEATURES_PATH)
        if STATSBOMB_FEATURES_PATH.exists()
        else None
    )
    best_params, tuning_results = tune_statsbomb_goal_params(matches, statsbomb_features)
    metrics, predictions = run_statsbomb_goal_backtest(
        matches,
        statsbomb_features,
        best_params,
        start_year=2022,
    )
    predictions.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)

    world_cup_2022 = predictions[
        (pd.to_datetime(predictions["date"]).dt.year == 2022)
        & predictions["tournament"].astype(str).eq("FIFA World Cup")
    ].copy()
    wc_metrics = _subset_metrics(world_cup_2022)
    summary = {
        "model": "statsbomb_goal_model",
        **{f"all_2022_plus_{key}": value for key, value in metrics.items()},
        **{f"world_cup_2022_{key}": value for key, value in wc_metrics.items()},
    }
    pd.DataFrame([summary]).to_csv(METRICS_OUTPUT_PATH, index=False)

    print("Best StatsBomb goal-model params from 2018 World Cup validation:")
    print(tuning_results[tuning_results["selected"]].to_string(index=False))
    print("StatsBomb goal model holdout metrics:")
    print(
        pd.DataFrame(
            [
                {
                    "scope": "2022+",
                    "matches": metrics["matches"],
                    "log_loss": metrics["log_loss"],
                    "accuracy": metrics["accuracy"],
                    "draw_recall": metrics["draw_recall"],
                },
                {
                    "scope": "2022 World Cup",
                    "matches": wc_metrics["matches"],
                    "log_loss": wc_metrics["log_loss"],
                    "accuracy": wc_metrics["accuracy"],
                    "draw_recall": wc_metrics["draw_recall"],
                },
            ]
        ).to_string(index=False)
    )
    print(f"Wrote {PREDICTIONS_OUTPUT_PATH}")
    print(f"Wrote {METRICS_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

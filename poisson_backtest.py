"""Run the separate Elo-informed Poisson goal model backtest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import BACKTEST_START_YEAR
from src.data_loader import clean_results, load_results
from src.poisson_backtest import run_poisson_goal_backtest
from src.utils import ensure_directories


RESULTS_PATH = Path("data/raw/results.csv")
PREDICTIONS_OUTPUT_PATH = Path("results/poisson_backtest_predictions.csv")
SUMMARY_OUTPUT_PATH = Path("results/poisson_backtest_summary.csv")


def main() -> int:
    """Load results, run the Poisson backtest, and save outputs."""
    ensure_directories()
    matches = clean_results(load_results(str(RESULTS_PATH)))
    metrics, predictions = run_poisson_goal_backtest(matches, start_year=BACKTEST_START_YEAR)
    predictions.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    pd.DataFrame([metrics]).to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print("Poisson goal model backtest metrics:")
    for key, value in metrics.items():
        if key in {
            "matches",
            "start_year",
            "top_probability_predicted_draws",
            "predicted_draws",
            "actual_draws",
        }:
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.6f}")
    print(f"Wrote {PREDICTIONS_OUTPUT_PATH}")
    print(f"Wrote {SUMMARY_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

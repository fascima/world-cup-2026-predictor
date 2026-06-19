"""Run a chronological backtest for the current Elo model."""

from __future__ import annotations

from pathlib import Path

from src.config import BACKTEST_START_YEAR
from src.backtest import run_backtest
from src.data_loader import clean_results, load_results
from src.utils import ensure_directories


RESULTS_PATH = Path("data/raw/results.csv")
BACKTEST_OUTPUT_PATH = Path("results/backtest_predictions.csv")


def main() -> int:
    """Load results, run the backtest, and save scored predictions."""
    ensure_directories()
    matches = clean_results(load_results(str(RESULTS_PATH)))
    metrics, predictions = run_backtest(matches, start_year=BACKTEST_START_YEAR)
    predictions.to_csv(BACKTEST_OUTPUT_PATH, index=False)

    print("Backtest metrics:")
    for key, value in metrics.items():
        if key in {"matches", "start_year"}:
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.6f}")
    print(f"Wrote {BACKTEST_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Backtest historical World Cup market-value adjustments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest import run_world_cup_market_value_backtest
from src.data_loader import clean_results, load_results
from src.market_value import load_historical_world_cup_market_values
from src.utils import ensure_directories


RESULTS_PATH = Path("data/raw/results.csv")
HISTORICAL_MARKET_VALUES_PATH = Path("data/fixtures/historical_world_cup_market_values.csv")
TYPO_MARKET_VALUES_PATH = Path("data/fixtures/historical_world_cup_market_values.csv.csv")
PREDICTIONS_OUTPUT_PATH = Path("results/world_cup_market_value_backtest_predictions.csv")
SUMMARY_OUTPUT_PATH = Path("results/world_cup_market_value_backtest_summary.csv")


def _resolve_market_values_path() -> Path:
    """Return the historical market-value file path, with a typo fallback."""
    if HISTORICAL_MARKET_VALUES_PATH.exists():
        return HISTORICAL_MARKET_VALUES_PATH
    if TYPO_MARKET_VALUES_PATH.exists():
        print(
            f"Using {TYPO_MARKET_VALUES_PATH}. Rename it to {HISTORICAL_MARKET_VALUES_PATH} "
            "when convenient."
        )
        return TYPO_MARKET_VALUES_PATH
    return HISTORICAL_MARKET_VALUES_PATH


def _summary_frame(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert nested metrics into a compact CSV table."""
    rows = []
    for model_name, model_metrics in metrics.items():
        row = {"model": model_name}
        row.update(model_metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    """Load data, run the World Cup market-value backtest, and save outputs."""
    ensure_directories()

    market_values_path = _resolve_market_values_path()
    market_values_by_year = load_historical_world_cup_market_values(market_values_path)
    if not market_values_by_year:
        print(
            f"No historical market values found at {market_values_path}. Add rows with "
            "tournament_year,team,market_value_eur before running this backtest."
        )
        return 1

    # This backtest disables the default 2010 filter so older provided World
    # Cups, such as 2006, can be scored if results.csv contains them.
    matches = clean_results(load_results(str(RESULTS_PATH)), min_match_year=None)
    metrics, predictions = run_world_cup_market_value_backtest(matches, market_values_by_year)

    predictions.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    summary = _summary_frame(metrics)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print("World Cup market-value backtest:")
    for model_name in ["elo_only", "elo_plus_market_value"]:
        model_metrics = metrics[model_name]
        print(model_name)
        print(f"matches: {int(model_metrics['matches'])}")
        print(f"log_loss: {model_metrics['log_loss']:.6f}")
        print(f"brier_score: {model_metrics['brier_score']:.6f}")
        print(f"accuracy: {model_metrics['accuracy']:.6f}")

    comparison = metrics["comparison"]
    print("comparison deltas: elo_plus_market_value - elo_only")
    print(f"log_loss_delta: {comparison['log_loss_delta']:.6f}")
    print(f"brier_score_delta: {comparison['brier_score_delta']:.6f}")
    print(f"accuracy_delta: {comparison['accuracy_delta']:.6f}")
    print(f"Wrote {PREDICTIONS_OUTPUT_PATH}")
    print(f"Wrote {SUMMARY_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

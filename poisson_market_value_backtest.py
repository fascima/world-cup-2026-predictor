"""Backtest historical World Cup market values in the separate Poisson model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import clean_results, load_results
from src.market_value import load_historical_world_cup_market_values
from src.poisson_backtest import run_world_cup_poisson_market_value_backtest
from src.utils import ensure_directories


RESULTS_PATH = Path("data/raw/results.csv")
HISTORICAL_MARKET_VALUES_PATH = Path("data/fixtures/historical_world_cup_market_values.csv")
PREDICTIONS_OUTPUT_PATH = Path("results/poisson_world_cup_market_value_predictions.csv")
SUMMARY_OUTPUT_PATH = Path("results/poisson_world_cup_market_value_summary.csv")


def _summary_frame(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for model, model_metrics in metrics.items():
        row = {"model": model}
        row.update(model_metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    """Run the World Cup-only Poisson market-value backtest."""
    ensure_directories()
    market_values_by_year = load_historical_world_cup_market_values(HISTORICAL_MARKET_VALUES_PATH)
    if not market_values_by_year:
        print(f"Missing or empty historical market values: {HISTORICAL_MARKET_VALUES_PATH}")
        return 1

    matches = clean_results(load_results(str(RESULTS_PATH)), min_match_year=None)
    metrics, predictions = run_world_cup_poisson_market_value_backtest(matches, market_values_by_year)
    predictions.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    _summary_frame(metrics).to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print("Poisson World Cup market-value backtest:")
    for model in ["poisson", "poisson_plus_market_value"]:
        model_metrics = metrics[model]
        print(model)
        print(f"matches: {int(model_metrics['matches'])}")
        print(f"log_loss: {model_metrics['log_loss']:.6f}")
        print(f"brier_score: {model_metrics['brier_score']:.6f}")
        print(f"top_probability_accuracy: {model_metrics['top_probability_accuracy']:.6f}")
        print(f"decision_accuracy: {model_metrics['decision_accuracy']:.6f}")
        print(f"predicted_draws: {int(model_metrics['predicted_draws'])}")
        print(f"draw_recall: {model_metrics['draw_recall']:.6f}")

    comparison = metrics["comparison"]
    print("comparison deltas: poisson_plus_market_value - poisson")
    print(f"log_loss_delta: {comparison['log_loss_delta']:.6f}")
    print(f"brier_score_delta: {comparison['brier_score_delta']:.6f}")
    print(f"top_probability_accuracy_delta: {comparison['top_probability_accuracy_delta']:.6f}")
    print(f"decision_accuracy_delta: {comparison['decision_accuracy_delta']:.6f}")
    print(f"Wrote {PREDICTIONS_OUTPUT_PATH}")
    print(f"Wrote {SUMMARY_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

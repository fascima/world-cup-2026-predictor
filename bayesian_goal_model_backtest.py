"""Run the regularized Bayesian-style attack/defense goal model backtest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.bayesian_goal_model import score_predictions, train_bayesian_goal_model
from src.data_loader import clean_results, load_results
from src.features import build_ml_feature_dataset, save_ml_feature_dataset
from src.injury_features import load_injury_feature_index
from src.market_value import load_historical_world_cup_market_values
from src.statsbomb_features import load_statsbomb_team_match_features
from src.utils import ensure_directories


RESULTS_PATH = Path("data/raw/results.csv")
ML_FEATURES_PATH = Path("data/processed/ml_match_features.csv")
HISTORICAL_MARKET_VALUES_PATH = Path("data/fixtures/historical_world_cup_market_values.csv")
INJURED_PLAYERS_MARKET_VALUES_PATH = Path("data/fixtures/injured_players_market_value_filled.csv")
STATSBOMB_TEAM_MATCH_FEATURES_PATH = Path("data/processed/statsbomb_team_match_features.csv")
MODEL_PATH = Path("models/bayesian_goal_model.joblib")
PREDICTIONS_PATH = Path("results/bayesian_goal_model_predictions.csv")
SUMMARY_PATH = Path("results/bayesian_goal_model_summary.csv")
TUNING_PATH = Path("results/bayesian_goal_model_tuning.csv")


def _load_or_build_features() -> pd.DataFrame:
    """Load features if they already include scores, otherwise rebuild with current inputs."""
    if ML_FEATURES_PATH.exists():
        features = pd.read_csv(ML_FEATURES_PATH, parse_dates=["date"])
        if {"team_a_score", "team_b_score"}.issubset(features.columns):
            return features

    matches = clean_results(load_results(str(RESULTS_PATH)))
    market_values_by_year = load_historical_world_cup_market_values(HISTORICAL_MARKET_VALUES_PATH)
    injury_feature_index = load_injury_feature_index(
        INJURED_PLAYERS_MARKET_VALUES_PATH,
        market_values_by_year=market_values_by_year,
    )
    statsbomb_features = None
    if STATSBOMB_TEAM_MATCH_FEATURES_PATH.exists():
        statsbomb_features = load_statsbomb_team_match_features(STATSBOMB_TEAM_MATCH_FEATURES_PATH)
    features = build_ml_feature_dataset(
        matches,
        market_values_by_year=market_values_by_year,
        statsbomb_team_match_features=statsbomb_features,
        injury_feature_index=injury_feature_index,
    )
    save_ml_feature_dataset(features, str(ML_FEATURES_PATH))
    return features


def _world_cup_2022_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    mask = predictions["date"].dt.year.eq(2022) & predictions["tournament"].eq("FIFA World Cup")
    wc = predictions.loc[mask].copy()
    metrics = score_predictions(wc)
    return {f"world_cup_2022_{key}": value for key, value in metrics.items()}


def main() -> int:
    ensure_directories()
    features = _load_or_build_features()
    _, predictions, tuning = train_bayesian_goal_model(features, model_path=str(MODEL_PATH))
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    tuning.to_csv(TUNING_PATH, index=False)

    metrics = {
        "alpha": predictions.attrs.get("alpha"),
        "draw_multiplier": predictions.attrs.get("draw_multiplier"),
        "draw_decision_threshold": predictions.attrs.get("draw_decision_threshold"),
        "draw_decision_max_win_gap": predictions.attrs.get("draw_decision_max_win_gap"),
        **score_predictions(predictions),
        **_world_cup_2022_metrics(predictions),
    }
    pd.DataFrame([metrics]).to_csv(SUMMARY_PATH, index=False)

    print("Bayesian-style goal model metrics:")
    display = pd.DataFrame([metrics])
    print(display.to_string(index=False))
    print(f"Wrote {PREDICTIONS_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {TUNING_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

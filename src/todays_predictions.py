"""Today's World Cup match predictions across active model families."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from simulate_2026_model_family_predictions import build_predictors
from src.data_loader import clean_results, load_results
from src.draw_model import build_empirical_draw_model
from src.elo import build_elo_history
from src.live_world_cup import LIVE_MATCHES_PATH, load_cached_matches, normalize_team_name, todays_matches
from src.predict import predict_match_elo
from src.utils import normalize_probabilities


TODAYS_PREDICTIONS_PATH = Path("results/todays_match_predictions.csv")
CURRENT_ELO_RATINGS_PATH = Path("results/current_elo_ratings.csv")
LIVE_RESULTS_PATH = Path("data/live/results_with_live_world_cup.csv")
RAW_RESULTS_PATH = Path("data/raw/results.csv")

MODEL_DISPLAY_NAMES = {
    "elo": "Elo Model",
    "regression": "Regression Model",
    "gradient_boosting": "Gradient Boosting Model",
    "blended": "Blended Model",
    "market_adjusted_wc_elo": "Market-Adjusted WC Elo Model",
}

PREDICTION_COLUMNS = [
    "prediction_date",
    "match_id",
    "kickoff_utc",
    "local_date",
    "status",
    "stage",
    "group",
    "team_a",
    "team_b",
    "model_key",
    "model",
    "team_a_win_prob",
    "draw_prob",
    "team_b_win_prob",
    "team_a_advancement_prob",
    "team_b_advancement_prob",
]


def _ratings_lookup(path: Path = CURRENT_ELO_RATINGS_PATH) -> dict[str, float]:
    ratings = pd.read_csv(path)
    return {str(row.team): float(row.elo) for row in ratings.itertuples(index=False)}


def _draw_model() -> dict[str, object]:
    results_path = LIVE_RESULTS_PATH if LIVE_RESULTS_PATH.exists() else RAW_RESULTS_PATH
    matches = clean_results(load_results(str(results_path)))
    elo_history, _ = build_elo_history(matches)
    return build_empirical_draw_model(elo_history)


def _phase_from_stage(stage: object) -> str:
    text = str(stage or "").lower()
    return "group" if "group" in text else "knockout"


def _append_prediction_row(
    rows: list[dict[str, object]],
    match: pd.Series,
    model_key: str,
    model_name: str,
    team_a_win_prob: float,
    draw_prob: float,
    team_b_win_prob: float,
) -> None:
    team_a_adv, team_b_adv = normalize_probabilities(
        [team_a_win_prob + draw_prob / 2.0, team_b_win_prob + draw_prob / 2.0]
    )
    rows.append(
        {
            "prediction_date": date.today().isoformat(),
            "match_id": str(match.get("match_id", "")),
            "kickoff_utc": match.get("utc_date", ""),
            "local_date": match.get("local_date", ""),
            "status": match.get("status", ""),
            "stage": match.get("stage", ""),
            "group": match.get("group", ""),
            "team_a": normalize_team_name(match.get("home_team", "")),
            "team_b": normalize_team_name(match.get("away_team", "")),
            "model_key": model_key,
            "model": model_name,
            "team_a_win_prob": float(team_a_win_prob),
            "draw_prob": float(draw_prob),
            "team_b_win_prob": float(team_b_win_prob),
            "team_a_advancement_prob": float(team_a_adv),
            "team_b_advancement_prob": float(team_b_adv),
        }
    )


def build_todays_predictions(
    matches: pd.DataFrame | None = None,
    today: date | str | None = None,
    output_path: Path = TODAYS_PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Build W/D/L predictions for matches scheduled today."""
    source_matches = matches if matches is not None else load_cached_matches(LIVE_MATCHES_PATH)
    todays = todays_matches(source_matches, today=today)
    if todays.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=PREDICTION_COLUMNS)
        empty.to_csv(output_path, index=False)
        return empty

    ratings = _ratings_lookup()
    draw_model = _draw_model()
    predictors = build_predictors()
    rows: list[dict[str, object]] = []

    for _, match in todays.iterrows():
        team_a = normalize_team_name(match.get("home_team", ""))
        team_b = normalize_team_name(match.get("away_team", ""))
        if not team_a or not team_b:
            continue

        elo = predict_match_elo(team_a, team_b, ratings, neutral=True, draw_model=draw_model)
        _append_prediction_row(
            rows,
            match,
            "elo",
            MODEL_DISPLAY_NAMES["elo"],
            float(elo["team_a_win_prob"]),
            float(elo["draw_prob"]),
            float(elo["team_b_win_prob"]),
        )

        phase = _phase_from_stage(match.get("stage", ""))
        for model_key in ["regression", "gradient_boosting", "blended", "market_adjusted_wc_elo"]:
            prediction = predictors[model_key].predict(team_a, team_b, phase)
            _append_prediction_row(
                rows,
                match,
                model_key,
                MODEL_DISPLAY_NAMES[model_key],
                prediction.team_a_win_prob,
                prediction.draw_prob,
                prediction.team_b_win_prob,
            )

    df = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

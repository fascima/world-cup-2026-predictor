"""Today's World Cup match predictions across active model families."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from simulate_2026_model_family_predictions import build_predictors
from src import config
from src.data_loader import clean_results, load_results
from src.draw_model import build_empirical_draw_model
from src.live_evaluation import BACKFILL_PREDICTIONS_PATH, append_predictions_to_ledger, utc_now_iso
from src.elo import build_elo_history
from src.live_world_cup import LIVE_MATCHES_PATH, load_cached_matches, normalize_team_name, todays_matches
from src.poisson_model import (
    apply_dixon_coles_adjustment,
    fit_average_total_goals,
    fit_goal_profile_state,
    independent_poisson_score_matrix,
    poisson_outcome_probabilities,
)
from src.predict import expected_score
from src.predict import predict_match_elo
from src.utils import normalize_probabilities


TODAYS_PREDICTIONS_PATH = Path("results/todays_match_predictions.csv")
TODAYS_POISSON_SCORE_PREDICTIONS_PATH = Path("results/todays_poisson_score_predictions.csv")
FUTURE_PREDICTIONS_PATH = Path("results/world_cup_2026_upcoming_match_predictions.csv")
CURRENT_ELO_RATINGS_PATH = Path("results/current_elo_ratings.csv")
LIVE_RESULTS_PATH = Path("data/live/results_with_live_world_cup.csv")
RAW_RESULTS_PATH = Path("data/raw/results.csv")

# Match the standalone 2026 Poisson tournament script settings.
POISSON_SCORE_DRAW_INFLATION = 1.20
POISSON_SCORE_GOAL_PROFILE_WEIGHT = 0.50

MODEL_DISPLAY_NAMES = {
    "elo": "Elo Model",
    "regression": "Regression Model",
    "gradient_boosting": "Gradient Boosting Model",
    "blended": "Blended Model",
    "market_adjusted_wc_elo": "Market-Adjusted WC Elo Model",
}

PREDICTION_COLUMNS = [
    "prediction_generated_at",
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
    "model_version",
    "team_a_win_prob",
    "draw_prob",
    "team_b_win_prob",
    "team_a_advancement_prob",
    "team_b_advancement_prob",
]

POISSON_SCORE_COLUMNS = [
    "prediction_generated_at",
    "prediction_date",
    "match_id",
    "kickoff_utc",
    "local_date",
    "status",
    "stage",
    "group",
    "team_a",
    "team_b",
    "team_a_expected_goals",
    "team_b_expected_goals",
    "team_a_win_prob",
    "draw_prob",
    "team_b_win_prob",
    "scoreline_1",
    "scoreline_1_prob",
    "scoreline_2",
    "scoreline_2_prob",
    "scoreline_3",
    "scoreline_3_prob",
]


def _ratings_lookup(path: Path = CURRENT_ELO_RATINGS_PATH) -> dict[str, float]:
    ratings = pd.read_csv(path)
    return {str(row.team): float(row.elo) for row in ratings.itertuples(index=False)}


def _draw_model() -> dict[str, object]:
    results_path = LIVE_RESULTS_PATH if LIVE_RESULTS_PATH.exists() else RAW_RESULTS_PATH
    matches = clean_results(load_results(str(results_path)))
    elo_history, _ = build_elo_history(matches)
    return build_empirical_draw_model(elo_history)


def _poisson_training_state() -> tuple[float, dict[str, dict[str, float]]]:
    results_path = LIVE_RESULTS_PATH if LIVE_RESULTS_PATH.exists() else RAW_RESULTS_PATH
    matches = clean_results(load_results(str(results_path)))
    elo_history, _ = build_elo_history(matches)
    return fit_average_total_goals(elo_history), fit_goal_profile_state(matches)


def _phase_from_stage(stage: object) -> str:
    text = str(stage or "").lower()
    return "group" if "group" in text else "knockout"


def _append_prediction_row(
    rows: list[dict[str, object]],
    match: pd.Series,
    generated_at: str,
    model_key: str,
    model_name: str,
    model_version: str,
    team_a_win_prob: float,
    draw_prob: float,
    team_b_win_prob: float,
) -> None:
    team_a_adv, team_b_adv = normalize_probabilities(
        [team_a_win_prob + draw_prob / 2.0, team_b_win_prob + draw_prob / 2.0]
    )
    rows.append(
        {
            "prediction_generated_at": generated_at,
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
            "model_version": model_version,
            "team_a_win_prob": float(team_a_win_prob),
            "draw_prob": float(draw_prob),
            "team_b_win_prob": float(team_b_win_prob),
            "team_a_advancement_prob": float(team_a_adv),
            "team_b_advancement_prob": float(team_b_adv),
        }
    )


def _predict_matches(
    matches: pd.DataFrame,
    output_path: Path,
    model_version: str,
    preserve_ledger: bool,
) -> pd.DataFrame:
    """Build W/D/L prediction rows for a supplied match frame."""
    if matches.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=PREDICTION_COLUMNS)
        empty.to_csv(output_path, index=False)
        return empty

    ratings = _ratings_lookup()
    draw_model = _draw_model()
    predictors = build_predictors()
    generated_at = utc_now_iso()
    rows: list[dict[str, object]] = []

    for _, match in matches.iterrows():
        team_a = normalize_team_name(match.get("home_team", ""))
        team_b = normalize_team_name(match.get("away_team", ""))
        if not team_a or not team_b:
            continue

        elo = predict_match_elo(team_a, team_b, ratings, neutral=True, draw_model=draw_model)
        _append_prediction_row(
            rows,
            match,
            generated_at,
            "elo",
            MODEL_DISPLAY_NAMES["elo"],
            model_version,
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
                generated_at,
                model_key,
                MODEL_DISPLAY_NAMES[model_key],
                model_version,
                prediction.team_a_win_prob,
                prediction.draw_prob,
                prediction.team_b_win_prob,
            )

    df = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    if preserve_ledger:
        append_predictions_to_ledger(df, generated_at=generated_at)
    return df


def _top_scorelines(score_matrix: np.ndarray, limit: int = 3) -> list[tuple[str, float]]:
    scores = [
        (f"{team_a_goals}-{team_b_goals}", float(score_matrix[team_a_goals, team_b_goals]))
        for team_a_goals in range(score_matrix.shape[0])
        for team_b_goals in range(score_matrix.shape[1])
    ]
    return sorted(scores, key=lambda item: item[1], reverse=True)[:limit]


def _poisson_score_prediction(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    average_total_goals: float,
    goal_profile_state: dict[str, dict[str, float]],
) -> dict[str, object]:
    rating_a = float(ratings.get(team_a, config.INITIAL_ELO))
    rating_b = float(ratings.get(team_b, config.INITIAL_ELO))
    adjusted_elo_diff = rating_a - rating_b
    prediction = poisson_outcome_probabilities(
        expected_score(rating_a, rating_b),
        average_total_goals,
        adjusted_elo_diff=adjusted_elo_diff,
        goal_profile_state=goal_profile_state,
        home_team=team_a,
        away_team=team_b,
        draw_inflation=POISSON_SCORE_DRAW_INFLATION,
        goal_profile_weight=POISSON_SCORE_GOAL_PROFILE_WEIGHT,
    )
    score_matrix = independent_poisson_score_matrix(
        float(prediction["home_expected_goals"]),
        float(prediction["away_expected_goals"]),
        max_goals=config.POISSON_MAX_GOALS,
    )
    if config.USE_DIXON_COLES:
        score_matrix = apply_dixon_coles_adjustment(
            score_matrix,
            float(prediction["home_expected_goals"]),
            float(prediction["away_expected_goals"]),
            config.DIXON_COLES_RHO,
        )
    top_scores = _top_scorelines(score_matrix, limit=3)
    row: dict[str, object] = {
        "team_a_expected_goals": float(prediction["home_expected_goals"]),
        "team_b_expected_goals": float(prediction["away_expected_goals"]),
        "team_a_win_prob": float(prediction["home_win_prob"]),
        "draw_prob": float(prediction["draw_prob"]),
        "team_b_win_prob": float(prediction["away_win_prob"]),
    }
    for index in range(3):
        scoreline, probability = top_scores[index] if index < len(top_scores) else ("", 0.0)
        row[f"scoreline_{index + 1}"] = scoreline
        row[f"scoreline_{index + 1}_prob"] = probability
    return row


def build_todays_poisson_score_predictions(
    matches: pd.DataFrame | None = None,
    today: date | str | None = None,
    output_path: Path = TODAYS_POISSON_SCORE_PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Build top scoreline forecasts from the Poisson goal model for today's matches."""
    source_matches = matches if matches is not None else load_cached_matches(LIVE_MATCHES_PATH)
    todays = _known_team_matches(todays_matches(source_matches, today=today))
    if todays.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=POISSON_SCORE_COLUMNS)
        empty.to_csv(output_path, index=False)
        return empty

    ratings = _ratings_lookup()
    average_total_goals, goal_profile_state = _poisson_training_state()
    generated_at = utc_now_iso()
    rows: list[dict[str, object]] = []
    for _, match in todays.iterrows():
        team_a = normalize_team_name(match.get("home_team", ""))
        team_b = normalize_team_name(match.get("away_team", ""))
        if not team_a or not team_b:
            continue
        row = {
            "prediction_generated_at": generated_at,
            "prediction_date": date.today().isoformat(),
            "match_id": str(match.get("match_id", "")),
            "kickoff_utc": match.get("utc_date", ""),
            "local_date": match.get("local_date", ""),
            "status": match.get("status", ""),
            "stage": match.get("stage", ""),
            "group": match.get("group", ""),
            "team_a": team_a,
            "team_b": team_b,
        }
        row.update(
            _poisson_score_prediction(
                team_a,
                team_b,
                ratings,
                average_total_goals,
                goal_profile_state,
            )
        )
        rows.append(row)

    df = pd.DataFrame(rows, columns=POISSON_SCORE_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def _known_team_matches(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()
    known = matches.copy()
    known = known.dropna(subset=["home_team", "away_team"])
    known["home_team"] = known["home_team"].map(normalize_team_name)
    known["away_team"] = known["away_team"].map(normalize_team_name)
    known = known[known["home_team"].astype(str).str.len().gt(0)]
    known = known[known["away_team"].astype(str).str.len().gt(0)]
    return known


def build_todays_predictions(
    matches: pd.DataFrame | None = None,
    today: date | str | None = None,
    output_path: Path = TODAYS_PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Build W/D/L predictions for matches scheduled today."""
    source_matches = matches if matches is not None else load_cached_matches(LIVE_MATCHES_PATH)
    todays = _known_team_matches(todays_matches(source_matches, today=today))
    return _predict_matches(todays, output_path, model_version="live", preserve_ledger=True)


def build_upcoming_match_predictions(
    matches: pd.DataFrame | None = None,
    today: date | str | None = None,
    output_path: Path = FUTURE_PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Build and lock predictions for all cached known-team matches from today forward."""
    source_matches = matches if matches is not None else load_cached_matches(LIVE_MATCHES_PATH)
    if source_matches.empty or "local_date" not in source_matches.columns:
        return _predict_matches(pd.DataFrame(), output_path, model_version="live", preserve_ledger=True)
    target = str(today or date.today().isoformat())
    upcoming = source_matches[source_matches["local_date"].astype(str).ge(target)].copy()
    upcoming = _known_team_matches(upcoming)
    return _predict_matches(upcoming, output_path, model_version="live", preserve_ledger=True)


def build_cached_match_backfill_predictions(
    matches: pd.DataFrame | None = None,
    output_path: Path = BACKFILL_PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Build clearly labeled cached-fixture predictions for pre-ledger match coverage."""
    source_matches = matches if matches is not None else load_cached_matches(LIVE_MATCHES_PATH)
    known = _known_team_matches(source_matches)
    return _predict_matches(
        known,
        output_path,
        model_version="cached_fixture_backfill",
        preserve_ledger=False,
    )

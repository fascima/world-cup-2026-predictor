"""Official-bracket World Cup challenge predictions and scoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd

from simulate_2026_model_family_predictions import build_predictors
from src.live_world_cup import FINISHED_STATUSES, load_cached_matches, normalize_team_name
from src.predict import predict_match_elo
from src.todays_predictions import MODEL_DISPLAY_NAMES, _draw_model, _ratings_lookup
from src.utils import normalize_probabilities


BRACKET_SCOREBOARD_PATH = Path("results/2026_model_bracket_challenge_scoreboard.csv")
BRACKET_PICK_EVALUATION_PATH = Path("results/2026_model_bracket_challenge_picks.csv")

ROUND_POINTS = {
    "round_of_32": 1,
    "round_of_16": 2,
    "quarterfinal": 4,
    "semifinal": 8,
    "final": 16,
}
ROUND_LABELS = {
    "round_of_32": "Round of 32",
    "round_of_16": "Round of 16",
    "quarterfinal": "Quarterfinal",
    "semifinal": "Semifinal",
    "final": "Final",
}

# Official FOX image order supplied in data/fixtures/world_cup_official_bracket_image.webp.
OFFICIAL_R32_MATCHUPS = [
    ("Germany", "Paraguay"),
    ("France", "Sweden"),
    ("South Africa", "Canada"),
    ("Netherlands", "Morocco"),
    ("Portugal", "Croatia"),
    ("Spain", "Austria"),
    ("United States", "Bosnia-Herzegovina"),
    ("Belgium", "Senegal"),
    ("Brazil", "Japan"),
    ("Ivory Coast", "Norway"),
    ("Mexico", "Ecuador"),
    ("England", "DR Congo"),
    ("Argentina", "Cape Verde"),
    ("Australia", "Egypt"),
    ("Switzerland", "Algeria"),
    ("Colombia", "Ghana"),
]

MODEL_KEYS = [
    "elo",
    "regression",
    "gradient_boosting",
    "blended",
    "market_adjusted_wc_elo",
]


@dataclass(frozen=True)
class MatchPrediction:
    team_a_advancement_prob: float
    team_b_advancement_prob: float
    predicted_winner: str

    @property
    def confidence(self) -> float:
        return max(self.team_a_advancement_prob, self.team_b_advancement_prob)


def _pair_key(team_a: object, team_b: object) -> str:
    return "|".join(sorted([normalize_team_name(team_a), normalize_team_name(team_b)]))


def _actual_winner(row: pd.Series) -> str:
    winner_code = str(row.get("winner", "") or "")
    if winner_code == "HOME_TEAM":
        return normalize_team_name(row.get("home_team", ""))
    if winner_code == "AWAY_TEAM":
        return normalize_team_name(row.get("away_team", ""))

    home = pd.to_numeric(row.get("home_score"), errors="coerce")
    away = pd.to_numeric(row.get("away_score"), errors="coerce")
    if pd.isna(home) or pd.isna(away) or float(home) == float(away):
        return ""
    return normalize_team_name(row.get("home_team", "")) if float(home) > float(away) else normalize_team_name(row.get("away_team", ""))


def _fixture_lookup(matches: pd.DataFrame) -> dict[str, dict[str, object]]:
    if matches.empty:
        return {}
    fixtures = matches.copy()
    stage_text = fixtures["stage"].astype(str).str.upper()
    fixtures = fixtures[~stage_text.str.contains("GROUP", na=False)].copy()
    fixtures = fixtures.dropna(subset=["home_team", "away_team"])
    lookup: dict[str, dict[str, object]] = {}
    for _, row in fixtures.iterrows():
        key = _pair_key(row.get("home_team"), row.get("away_team"))
        actual_winner = _actual_winner(row) if str(row.get("status", "")) in FINISHED_STATUSES else ""
        lookup[key] = {
            "match_id": str(row.get("match_id", "")),
            "kickoff_utc": row.get("utc_date", ""),
            "local_date": row.get("local_date", ""),
            "status": row.get("status", ""),
            "stage": row.get("stage", ""),
            "actual_winner": actual_winner,
        }
    return lookup


def _predict_match(
    model_key: str,
    team_a: str,
    team_b: str,
    predictors: dict[str, object],
    elo_ratings: dict[str, float],
    draw_model: dict[str, object],
) -> MatchPrediction:
    if model_key == "elo":
        prediction = predict_match_elo(team_a, team_b, elo_ratings, neutral=True, draw_model=draw_model)
        team_a_win = float(prediction["team_a_win_prob"])
        draw = float(prediction["draw_prob"])
        team_b_win = float(prediction["team_b_win_prob"])
    else:
        prediction = predictors[model_key].predict(team_a, team_b, "knockout")
        team_a_win = prediction.team_a_win_prob
        draw = prediction.draw_prob
        team_b_win = prediction.team_b_win_prob

    team_a_adv, team_b_adv = normalize_probabilities([team_a_win + draw / 2.0, team_b_win + draw / 2.0])
    predicted_winner = team_a if team_a_adv >= team_b_adv else team_b
    return MatchPrediction(float(team_a_adv), float(team_b_adv), predicted_winner)


def _append_pick(
    rows: list[dict[str, object]],
    fixture_lookup: dict[str, dict[str, object]],
    model_key: str,
    model_name: str,
    round_name: str,
    node_id: str,
    bracket_index: int,
    team_a: str,
    team_b: str,
    prediction: MatchPrediction,
) -> None:
    fixture = fixture_lookup.get(_pair_key(team_a, team_b), {})
    actual_winner = str(fixture.get("actual_winner", "") or "")
    evaluated = bool(actual_winner)
    correct = evaluated and prediction.predicted_winner == actual_winner
    points_available = ROUND_POINTS[round_name]
    if actual_winner == team_a:
        actual_probability = prediction.team_a_advancement_prob
    elif actual_winner == team_b:
        actual_probability = prediction.team_b_advancement_prob
    else:
        actual_probability = float("nan")
    log_loss = -math.log(max(float(actual_probability), 1e-12)) if evaluated else float("nan")
    weighted_log_loss = points_available * log_loss if evaluated else 0.0
    rows.append(
        {
            "model_key": model_key,
            "model": model_name,
            "node_id": node_id,
            "bracket_index": bracket_index,
            "round": round_name,
            "round_label": ROUND_LABELS[round_name],
            "match_id": fixture.get("match_id", node_id),
            "kickoff_utc": fixture.get("kickoff_utc", ""),
            "local_date": fixture.get("local_date", ""),
            "status": fixture.get("status", "PREDICTED"),
            "stage": fixture.get("stage", round_name),
            "team_a": team_a,
            "team_b": team_b,
            "team_a_advancement_prob": prediction.team_a_advancement_prob,
            "team_b_advancement_prob": prediction.team_b_advancement_prob,
            "predicted_winner": prediction.predicted_winner,
            "confidence": prediction.confidence,
            "actual_winner": actual_winner,
            "actual_probability": actual_probability,
            "log_loss": log_loss,
            "weighted_log_loss": weighted_log_loss,
            "points_available": points_available,
            "points_earned": -weighted_log_loss,
            "possible_points_remaining": 0 if evaluated else points_available,
            "evaluated": evaluated,
            "correct": correct,
        }
    )


def _play_round(
    rows: list[dict[str, object]],
    fixture_lookup: dict[str, dict[str, object]],
    model_key: str,
    model_name: str,
    predictors: dict[str, object],
    elo_ratings: dict[str, float],
    draw_model: dict[str, object],
    round_name: str,
    node_prefix: str,
    teams: list[str],
) -> list[str]:
    winners = []
    for index in range(0, len(teams), 2):
        team_a = teams[index]
        team_b = teams[index + 1]
        prediction = _predict_match(model_key, team_a, team_b, predictors, elo_ratings, draw_model)
        node_id = f"{node_prefix}_{index // 2 + 1:02d}"
        _append_pick(
            rows,
            fixture_lookup,
            model_key,
            model_name,
            round_name,
            node_id,
            index // 2,
            team_a,
            team_b,
            prediction,
        )
        winners.append(prediction.predicted_winner)
    return winners


def build_official_bracket_picks(matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build full deterministic model brackets from the official Round of 32 order."""
    cached_matches = matches if matches is not None else load_cached_matches()
    fixtures = _fixture_lookup(cached_matches)
    predictors = build_predictors()
    elo_ratings = _ratings_lookup()
    draw_model = _draw_model()

    rows: list[dict[str, object]] = []
    r32_teams = [team for matchup in OFFICIAL_R32_MATCHUPS for team in matchup]
    for model_key in MODEL_KEYS:
        model_name = MODEL_DISPLAY_NAMES[model_key]
        r16 = _play_round(rows, fixtures, model_key, model_name, predictors, elo_ratings, draw_model, "round_of_32", "R32", r32_teams)
        qf = _play_round(rows, fixtures, model_key, model_name, predictors, elo_ratings, draw_model, "round_of_16", "R16", r16)
        sf = _play_round(rows, fixtures, model_key, model_name, predictors, elo_ratings, draw_model, "quarterfinal", "QF", qf)
        finalists = _play_round(rows, fixtures, model_key, model_name, predictors, elo_ratings, draw_model, "semifinal", "SF", sf)
        _play_round(rows, fixtures, model_key, model_name, predictors, elo_ratings, draw_model, "final", "F", finalists)

    return pd.DataFrame(rows)


def score_bracket_challenge(matches: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score full deterministic model brackets against completed knockout fixtures."""
    picks = build_official_bracket_picks(matches=matches)
    if picks.empty:
        empty_scoreboard = pd.DataFrame(
            columns=[
                "model_key",
                "model",
                "weighted_log_loss",
                "average_log_loss",
                "max_points",
                "possible_points",
                "evaluated_picks",
                "correct_picks",
                "wrong_picks",
            ]
        )
        return empty_scoreboard, picks

    scoreboard = (
        picks.groupby(["model_key", "model"], as_index=False)
        .agg(
            weighted_log_loss=("weighted_log_loss", "sum"),
            max_points=("points_available", "sum"),
            possible_points_remaining=("possible_points_remaining", "sum"),
            evaluated_picks=("evaluated", "sum"),
            correct_picks=("correct", "sum"),
        )
        .sort_values(["weighted_log_loss", "possible_points_remaining"], ascending=[True, False], kind="mergesort")
    )
    scoreboard["average_log_loss"] = scoreboard["weighted_log_loss"] / scoreboard["evaluated_picks"].clip(lower=1)
    scoreboard.loc[scoreboard["evaluated_picks"].eq(0), "average_log_loss"] = 0.0
    scoreboard["possible_points"] = scoreboard["possible_points_remaining"]
    scoreboard["wrong_picks"] = scoreboard["evaluated_picks"] - scoreboard["correct_picks"]
    scoreboard["rank"] = range(1, len(scoreboard) + 1)
    return scoreboard[
        [
            "rank",
            "model_key",
            "model",
            "weighted_log_loss",
            "average_log_loss",
            "max_points",
            "possible_points",
            "evaluated_picks",
            "correct_picks",
            "wrong_picks",
        ]
    ], picks


def refresh_bracket_challenge_outputs(matches: pd.DataFrame | None = None) -> dict[str, int]:
    """Predict and score official full brackets."""
    scoreboard, picks = score_bracket_challenge(matches=matches)
    BRACKET_SCOREBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    scoreboard.to_csv(BRACKET_SCOREBOARD_PATH, index=False)
    picks.to_csv(BRACKET_PICK_EVALUATION_PATH, index=False)
    return {"scoreboard_rows": len(scoreboard), "pick_rows": len(picks)}

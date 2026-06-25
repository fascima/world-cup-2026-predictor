"""Run 2026 World Cup simulations for multiple model families.

The script writes a consistent pair of outputs for each supported model:

* ``results/2026_<model>_stage_probabilities.csv``
* ``results/2026_<model>_deterministic_bracket.json``

It intentionally keeps 2026 feature construction conservative. Historical
feature medians provide neutral defaults, while current team ratings, phase
flags, and World Cup context are set per hypothetical match.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from src import config
from src.bracket import (
    FINAL_MATCH,
    QUARTERFINAL_MATCHES,
    ROUND_OF_16_MATCHES,
    ROUND_OF_32_MATCH_NUMBERS,
    SEMIFINAL_MATCHES,
    build_round_of_32_matches,
    get_group_finishers,
)
from src.data_loader import clean_results, load_results
from src.draw_model import build_empirical_draw_model
from src.elo import build_elo_history
from src.predict import predict_match_elo
from src.simulate import STAGE_COLUMNS, load_groups
from src.utils import ensure_directories, normalize_probabilities


RESULTS_PATH = Path("data/raw/results.csv")
GROUPS_PATH = Path("data/fixtures/world_cup_2026_groups.csv")
FEATURES_PATH = Path("data/processed/ml_match_features.csv")
LOGISTIC_MODEL_PATH = Path("models/logistic_match_outcome.joblib")
GRADIENT_BOOSTING_MODEL_PATH = Path("models/gradient_boosting_match_outcome.joblib")
MODEL_RATINGS_PATH = Path("results/world_cup_team_model_ratings.csv")
OUTPUT_DIR = Path("results")

N_SIMULATIONS = int(os.environ.get("SIM_2026_N", "10000"))
RNG_SEED = config.MONTE_CARLO_SEED
BLEND_LOGISTIC_WEIGHT = 0.65
DEFAULT_MODEL_KEYS = [
    "regression",
    "gradient_boosting",
    "blended",
    "market_adjusted_wc_elo",
]


@dataclass(frozen=True)
class Prediction:
    team_a_win_prob: float
    draw_prob: float
    team_b_win_prob: float

    @property
    def team_a_advancement_prob(self) -> float:
        return self.team_a_win_prob + self.draw_prob / 2.0

    @property
    def team_b_advancement_prob(self) -> float:
        return self.team_b_win_prob + self.draw_prob / 2.0


class ModelPredictor:
    """Callable model wrapper with cached W/D/L predictions."""

    def __init__(
        self,
        name: str,
        predict_fn: Callable[[str, str, str], Prediction],
        precompute_fn: Callable[[list[str]], None] | None = None,
    ) -> None:
        self.name = name
        self._predict_fn = predict_fn
        self._precompute_fn = precompute_fn
        self._cache: dict[tuple[str, str, str], Prediction] = {}

    def predict(self, team_a: str, team_b: str, phase: str) -> Prediction:
        key = (team_a, team_b, phase)
        if key not in self._cache:
            self._cache[key] = self._predict_fn(team_a, team_b, phase)
        return self._cache[key]

    def precompute(self, teams: list[str]) -> None:
        if self._precompute_fn is not None:
            self._precompute_fn(teams)
            return
        pairs = [(team_a, team_b) for team_a in teams for team_b in teams if team_a != team_b]
        for team_a, team_b in pairs:
            self.predict(team_a, team_b, "group")
            self.predict(team_a, team_b, "knockout")


def _groups_are_complete(groups: dict[str, list[str]]) -> bool:
    return set(groups) == set("ABCDEFGHIJKL") and all(len(teams) == 4 for teams in groups.values())


def _required_model_columns(model) -> tuple[list[str], list[str]]:
    preprocessor = model.named_steps["preprocess"]
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for name, _, columns in preprocessor.transformers_:
        if name == "numeric":
            numeric_columns = list(columns)
        elif name == "categorical":
            categorical_columns = list(columns)
    return numeric_columns, categorical_columns


def _feature_defaults(features: pd.DataFrame, columns: list[str]) -> dict[str, float | str]:
    defaults: dict[str, float | str] = {}
    for column in columns:
        if column not in features.columns:
            defaults[column] = 0.0
            continue
        if pd.api.types.is_numeric_dtype(features[column]) or pd.api.types.is_bool_dtype(features[column]):
            value = pd.to_numeric(features[column], errors="coerce").median()
            defaults[column] = 0.0 if pd.isna(value) else float(value)
        else:
            mode = features[column].mode(dropna=True)
            defaults[column] = str(mode.iloc[0]) if not mode.empty else "unknown"
    return defaults


def _rating_lookup() -> dict[str, float]:
    ratings = pd.read_csv("results/current_elo_ratings.csv")
    return {str(row.team): float(row.elo) for row in ratings.itertuples(index=False)}


def _market_rating_lookup(fallback_ratings: dict[str, float]) -> dict[str, float]:
    if not MODEL_RATINGS_PATH.exists():
        return fallback_ratings
    ratings = pd.read_csv(MODEL_RATINGS_PATH)
    if "model_rating" not in ratings.columns:
        return fallback_ratings
    output = dict(fallback_ratings)
    for row in ratings.itertuples(index=False):
        output[str(row.team)] = float(row.model_rating)
    return output


def _set_if_present(row: dict[str, float | str], column: str, value: float | str) -> None:
    if column in row:
        row[column] = value


def _close_gap(abs_diff: float, threshold: float) -> float:
    return float(abs_diff <= threshold)


def _ml_feature_row(
    defaults: dict[str, float | str],
    team_a: str,
    team_b: str,
    phase: str,
    ratings: dict[str, float],
) -> pd.DataFrame:
    rating_a = float(ratings.get(team_a, config.INITIAL_ELO))
    rating_b = float(ratings.get(team_b, config.INITIAL_ELO))
    diff = rating_a - rating_b
    abs_diff = abs(diff)
    is_knockout = float(phase != "group")
    is_group = float(phase == "group")

    row = dict(defaults)
    _set_if_present(row, "neutral", 1.0)
    _set_if_present(row, "tournament_type", "world_cup")
    _set_if_present(row, "team_a_pre_elo", rating_a)
    _set_if_present(row, "team_b_pre_elo", rating_b)
    _set_if_present(row, "elo_diff", diff)
    _set_if_present(row, "team_a_prediction_elo", rating_a)
    _set_if_present(row, "team_b_prediction_elo", rating_b)
    _set_if_present(row, "prediction_elo_diff", diff)
    _set_if_present(row, "adjusted_elo_diff", diff)
    _set_if_present(row, "abs_adjusted_elo_diff", abs_diff)
    for threshold in (50, 100, 150, 200, 250):
        _set_if_present(row, f"close_elo_gap_{threshold}", _close_gap(abs_diff, threshold))
    _set_if_present(row, "team_a_home_advantage", 0.0)
    _set_if_present(row, "team_b_home_advantage", 0.0)
    _set_if_present(row, "elo_diff_neutral_interaction", diff)
    _set_if_present(row, "elo_diff_home_interaction", 0.0)
    _set_if_present(row, "is_world_cup", 1.0)
    _set_if_present(row, "is_world_cup_group_stage", is_group)
    _set_if_present(row, "is_world_cup_knockout", is_knockout)
    _set_if_present(row, "world_cup_knockout_abs_adjusted_elo_diff", abs_diff * is_knockout)
    _set_if_present(row, "world_cup_knockout_close_elo_gap_100", _close_gap(abs_diff, 100) * is_knockout)
    _set_if_present(row, "world_cup_group_abs_adjusted_elo_diff", abs_diff * is_group)
    _set_if_present(row, "team_a_group_matches_played", 0.0)
    _set_if_present(row, "team_b_group_matches_played", 0.0)
    _set_if_present(row, "team_a_group_points_before", 0.0)
    _set_if_present(row, "team_b_group_points_before", 0.0)
    _set_if_present(row, "team_a_group_goal_diff_before", 0.0)
    _set_if_present(row, "team_b_group_goal_diff_before", 0.0)
    _set_if_present(row, "team_a_likely_qualified", 0.0)
    _set_if_present(row, "team_b_likely_qualified", 0.0)
    _set_if_present(row, "team_a_must_win", 0.0)
    _set_if_present(row, "team_b_must_win", 0.0)
    _set_if_present(row, "team_a_rotation_risk", 0.0)
    _set_if_present(row, "team_b_rotation_risk", 0.0)
    _set_if_present(row, "motivation_diff", 0.0)
    _set_if_present(row, "team_a_is_host", float(team_a in {"Canada", "Mexico", "United States"}))
    _set_if_present(row, "team_b_is_host", float(team_b in {"Canada", "Mexico", "United States"}))
    return pd.DataFrame([row])


def _proba_by_class(model, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features)[0]
    output = np.zeros(3, dtype=float)
    for index, class_label in enumerate(model.named_steps["classifier"].classes_):
        output[int(class_label)] = raw[index]
    return np.asarray(normalize_probabilities(output), dtype=float)


def _make_ml_predictor(
    name: str,
    model,
    features: pd.DataFrame,
    ratings: dict[str, float],
) -> ModelPredictor:
    numeric_columns, categorical_columns = _required_model_columns(model)
    defaults = _feature_defaults(features, numeric_columns + categorical_columns)

    def predict_fn(team_a: str, team_b: str, phase: str) -> Prediction:
        row = _ml_feature_row(defaults, team_a, team_b, phase, ratings)
        probs = _proba_by_class(model, row[numeric_columns + categorical_columns])
        return Prediction(float(probs[0]), float(probs[1]), float(probs[2]))

    def precompute_fn(teams: list[str]) -> None:
        pairs = [(team_a, team_b) for team_a in teams for team_b in teams if team_a != team_b]
        for phase in ["group", "knockout"]:
            rows = [
                _ml_feature_row(defaults, team_a, team_b, phase, ratings).iloc[0].to_dict()
                for team_a, team_b in pairs
            ]
            frame = pd.DataFrame(rows)
            raw = model.predict_proba(frame[numeric_columns + categorical_columns])
            class_probs = np.zeros((len(frame), 3), dtype=float)
            for index, class_label in enumerate(model.named_steps["classifier"].classes_):
                class_probs[:, int(class_label)] = raw[:, index]
            class_probs = np.clip(class_probs, 1e-12, None)
            class_probs = class_probs / class_probs.sum(axis=1, keepdims=True)
            for (team_a, team_b), probs in zip(pairs, class_probs, strict=False):
                predictor._cache[(team_a, team_b, phase)] = Prediction(
                    float(probs[0]), float(probs[1]), float(probs[2])
                )

    predictor = ModelPredictor(name, predict_fn)
    predictor._precompute_fn = precompute_fn
    return predictor


def _make_blend_predictor(logistic: ModelPredictor, gradient_boosting: ModelPredictor) -> ModelPredictor:
    def predict_fn(team_a: str, team_b: str, phase: str) -> Prediction:
        log = logistic.predict(team_a, team_b, phase)
        gb = gradient_boosting.predict(team_a, team_b, phase)
        gb_weight = 1.0 - BLEND_LOGISTIC_WEIGHT
        probs = np.asarray(
            [
                BLEND_LOGISTIC_WEIGHT * log.team_a_win_prob + gb_weight * gb.team_a_win_prob,
                BLEND_LOGISTIC_WEIGHT * log.draw_prob + gb_weight * gb.draw_prob,
                BLEND_LOGISTIC_WEIGHT * log.team_b_win_prob + gb_weight * gb.team_b_win_prob,
            ],
            dtype=float,
        )
        probs = np.asarray(normalize_probabilities(probs), dtype=float)
        return Prediction(float(probs[0]), float(probs[1]), float(probs[2]))

    def precompute_fn(teams: list[str]) -> None:
        logistic.precompute(teams)
        gradient_boosting.precompute(teams)
        pairs = [(team_a, team_b) for team_a in teams for team_b in teams if team_a != team_b]
        for phase in ["group", "knockout"]:
            for team_a, team_b in pairs:
                predictor._cache[(team_a, team_b, phase)] = predict_fn(team_a, team_b, phase)

    predictor = ModelPredictor("blended", predict_fn)
    predictor._precompute_fn = precompute_fn
    return predictor


def _make_market_adjusted_elo_predictor(
    ratings: dict[str, float],
    draw_model: dict[str, object],
) -> ModelPredictor:
    def predict_fn(team_a: str, team_b: str, phase: str) -> Prediction:
        pred = predict_match_elo(team_a, team_b, ratings, neutral=True, draw_model=draw_model)
        return Prediction(
            float(pred["team_a_win_prob"]),
            float(pred["draw_prob"]),
            float(pred["team_b_win_prob"]),
        )

    return ModelPredictor("market_adjusted_wc_elo", predict_fn)


def _empty_group_stats(teams: list[str]) -> dict[str, dict[str, int | str]]:
    return {
        team: {
            "team": team,
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
        }
        for team in teams
    }


def _record_match(
    stats: dict[str, dict[str, int | str]],
    team_a: str,
    team_b: str,
    goals_a: int,
    goals_b: int,
) -> None:
    stats[team_a]["goals_for"] = int(stats[team_a]["goals_for"]) + goals_a
    stats[team_a]["goals_against"] = int(stats[team_a]["goals_against"]) + goals_b
    stats[team_b]["goals_for"] = int(stats[team_b]["goals_for"]) + goals_b
    stats[team_b]["goals_against"] = int(stats[team_b]["goals_against"]) + goals_a
    stats[team_a]["goal_difference"] = int(stats[team_a]["goals_for"]) - int(stats[team_a]["goals_against"])
    stats[team_b]["goal_difference"] = int(stats[team_b]["goals_for"]) - int(stats[team_b]["goals_against"])

    if goals_a > goals_b:
        stats[team_a]["points"] = int(stats[team_a]["points"]) + 3
        stats[team_a]["wins"] = int(stats[team_a]["wins"]) + 1
        stats[team_b]["losses"] = int(stats[team_b]["losses"]) + 1
    elif goals_a < goals_b:
        stats[team_b]["points"] = int(stats[team_b]["points"]) + 3
        stats[team_b]["wins"] = int(stats[team_b]["wins"]) + 1
        stats[team_a]["losses"] = int(stats[team_a]["losses"]) + 1
    else:
        stats[team_a]["points"] = int(stats[team_a]["points"]) + 1
        stats[team_b]["points"] = int(stats[team_b]["points"]) + 1
        stats[team_a]["draws"] = int(stats[team_a]["draws"]) + 1
        stats[team_b]["draws"] = int(stats[team_b]["draws"]) + 1


def _sample_scoreline(prediction: Prediction, rng: np.random.Generator) -> tuple[int, int]:
    outcome = str(
        rng.choice(
            ["team_a_win", "draw", "team_b_win"],
            p=[prediction.team_a_win_prob, prediction.draw_prob, prediction.team_b_win_prob],
        )
    )
    if outcome == "draw":
        goals = int(rng.choice([0, 1, 2, 3], p=[0.30, 0.45, 0.20, 0.05]))
        return goals, goals
    margin = int(rng.choice([1, 2, 3, 4], p=[0.60, 0.25, 0.10, 0.05]))
    loser_goals = int(rng.choice([0, 1, 2], p=[0.45, 0.40, 0.15]))
    if outcome == "team_a_win":
        return loser_goals + margin, loser_goals
    return loser_goals, loser_goals + margin


def _simulate_group(
    group: str,
    teams: list[str],
    predictor: ModelPredictor,
    rng: np.random.Generator,
) -> dict[str, object]:
    stats = _empty_group_stats(teams)
    matches = []
    for team_a, team_b in combinations(teams, 2):
        prediction = predictor.predict(team_a, team_b, "group")
        goals_a, goals_b = _sample_scoreline(prediction, rng)
        _record_match(stats, team_a, team_b, goals_a, goals_b)
        matches.append({"team_a": team_a, "team_b": team_b, "goals_a": goals_a, "goals_b": goals_b})

    standings = []
    for record in stats.values():
        row = dict(record)
        row["random_tiebreaker"] = float(rng.random())
        standings.append(row)
    standings = sorted(
        standings,
        key=lambda row: (
            -int(row["points"]),
            -int(row["goal_difference"]),
            -int(row["goals_for"]),
            float(row["random_tiebreaker"]),
        ),
    )
    for row in standings:
        row.pop("random_tiebreaker")
    return {"standings": standings, "matches": matches}


def _simulate_group_stage(
    groups: dict[str, list[str]],
    predictor: ModelPredictor,
    rng: np.random.Generator,
) -> dict[str, object]:
    group_results: dict[str, object] = {}
    third_place_records = []

    for group in sorted(groups):
        result = _simulate_group(group, groups[group], predictor, rng)
        group_results[group] = result
        third_record = dict(result["standings"][2])
        third_record["group"] = group
        third_record["random_tiebreaker"] = float(rng.random())
        third_place_records.append(third_record)

    third_place_records = sorted(
        third_place_records,
        key=lambda row: (
            -int(row["points"]),
            -int(row["goal_difference"]),
            -int(row["goals_for"]),
            float(row["random_tiebreaker"]),
        ),
    )
    group_results["advancing_third_place_groups"] = [
        str(row["group"]) for row in third_place_records[:8]
    ]
    return group_results


def _simulate_knockout_match(
    team_a: str,
    team_b: str,
    predictor: ModelPredictor,
    rng: np.random.Generator,
) -> dict[str, object]:
    prediction = predictor.predict(team_a, team_b, "knockout")
    team_a_adv, team_b_adv = normalize_probabilities(
        [prediction.team_a_advancement_prob, prediction.team_b_advancement_prob]
    )
    winner = str(rng.choice([team_a, team_b], p=[team_a_adv, team_b_adv]))
    return {"team_a": team_a, "team_b": team_b, "winner": winner}


def _simulate_numbered_round(
    match_specs: list[tuple[int, int, int]],
    previous_winners: dict[int, str],
    predictor: ModelPredictor,
    rng: np.random.Generator,
) -> tuple[dict[int, str], list[dict[str, object]]]:
    winners = {}
    matches = []
    for match_number, source_a, source_b in match_specs:
        match = _simulate_knockout_match(previous_winners[source_a], previous_winners[source_b], predictor, rng)
        match["match_number"] = match_number
        winners[match_number] = str(match["winner"])
        matches.append(match)
    return winners, matches


def _simulate_knockout(
    group_results: dict[str, object],
    predictor: ModelPredictor,
    rng: np.random.Generator,
) -> dict[str, object]:
    finishers = get_group_finishers(group_results)
    r32_matches = build_round_of_32_matches(
        finishers,
        list(group_results["advancing_third_place_groups"]),
    )

    r32_winners = {}
    r32_details = []
    for match_number, (team_a, team_b) in zip(ROUND_OF_32_MATCH_NUMBERS, r32_matches):
        match = _simulate_knockout_match(team_a, team_b, predictor, rng)
        match["match_number"] = match_number
        r32_winners[match_number] = str(match["winner"])
        r32_details.append(match)

    r16_winners, r16_details = _simulate_numbered_round(ROUND_OF_16_MATCHES, r32_winners, predictor, rng)
    qf_winners, qf_details = _simulate_numbered_round(QUARTERFINAL_MATCHES, r16_winners, predictor, rng)
    sf_winners, sf_details = _simulate_numbered_round(SEMIFINAL_MATCHES, qf_winners, predictor, rng)

    final_number, source_a, source_b = FINAL_MATCH
    final_match = _simulate_knockout_match(sf_winners[source_a], sf_winners[source_b], predictor, rng)
    final_match["match_number"] = final_number

    return {
        "round_of_16": list(r32_winners.values()),
        "quarterfinal": list(r16_winners.values()),
        "semifinal": list(qf_winners.values()),
        "final": list(sf_winners.values()),
        "champion": str(final_match["winner"]),
        "bracket": {
            "round_of_32": r32_details,
            "round_of_16": r16_details,
            "quarterfinal": qf_details,
            "semifinal": sf_details,
            "final": [final_match],
        },
    }


def _simulate_tournament(
    groups: dict[str, list[str]],
    predictor: ModelPredictor,
    rng: np.random.Generator,
) -> dict[str, object]:
    group_results = _simulate_group_stage(groups, predictor, rng)
    knockout = _simulate_knockout(group_results, predictor, rng)
    return {
        "round_of_32": [
            team
            for group, result in group_results.items()
            if group != "advancing_third_place_groups"
            for team in [result["standings"][0]["team"], result["standings"][1]["team"]]
        ]
        + [
            group_results[group]["standings"][2]["team"]
            for group in group_results["advancing_third_place_groups"]
        ],
        "round_of_16": knockout["round_of_16"],
        "quarterfinal": knockout["quarterfinal"],
        "semifinal": knockout["semifinal"],
        "final": knockout["final"],
        "champion": knockout["champion"],
    }


def run_monte_carlo(
    groups: dict[str, list[str]],
    predictor: ModelPredictor,
    n_simulations: int,
    seed: int | None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = sorted({team for group_teams in groups.values() for team in group_teams})
    counts = {team: {stage: 0 for stage in STAGE_COLUMNS} for team in teams}

    for _ in range(n_simulations):
        result = _simulate_tournament(groups, predictor, rng)
        for stage in ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"]:
            for team in result[stage]:
                counts[team][stage] += 1
        counts[str(result["champion"])]["champion"] += 1

    rows = []
    for team in teams:
        rows.append(
            {
                "team": team,
                "round_of_32_prob": counts[team]["round_of_32"] / n_simulations,
                "round_of_16_prob": counts[team]["round_of_16"] / n_simulations,
                "quarterfinal_prob": counts[team]["quarterfinal"] / n_simulations,
                "semifinal_prob": counts[team]["semifinal"] / n_simulations,
                "final_prob": counts[team]["final"] / n_simulations,
                "champion_prob": counts[team]["champion"] / n_simulations,
            }
        )
    return pd.DataFrame(rows).sort_values("champion_prob", ascending=False).reset_index(drop=True)


RANKING_COLUMNS = [
    "champion_prob",
    "final_prob",
    "semifinal_prob",
    "quarterfinal_prob",
    "round_of_16_prob",
    "round_of_32_prob",
]


def _probability_lookup(probabilities: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(row["team"]): {column: float(row[column]) for column in RANKING_COLUMNS}
        for _, row in probabilities.iterrows()
    }


def _ranking_key(team: str, lookup: dict[str, dict[str, float]]) -> tuple[float, ...]:
    values = lookup.get(team)
    if values is None:
        return tuple(0.0 for _ in RANKING_COLUMNS)
    return tuple(values[column] for column in RANKING_COLUMNS)


def _build_deterministic_group_results(
    groups: dict[str, list[str]],
    lookup: dict[str, dict[str, float]],
) -> dict[str, object]:
    group_results: dict[str, object] = {}
    third_place_rows = []
    for group in sorted(groups):
        ranked_teams = sorted(groups[group], key=lambda team: (_ranking_key(team, lookup), team), reverse=True)
        standings = []
        for rank, team in enumerate(ranked_teams, start=1):
            standings.append(
                {
                    "team": team,
                    "points": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "goal_difference": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "probability_rank": rank,
                }
            )
        group_results[group] = {"standings": standings, "matches": []}
        third_place_rows.append({"group": group, "team": standings[2]["team"], **lookup[standings[2]["team"]]})

    third_place_rows = sorted(
        third_place_rows,
        key=lambda row: (tuple(float(row[column]) for column in RANKING_COLUMNS), str(row["team"])),
        reverse=True,
    )
    group_results["advancing_third_place_groups"] = [str(row["group"]) for row in third_place_rows[:8]]
    return group_results


def _pick_deterministic_winner(team_a: str, team_b: str, lookup: dict[str, dict[str, float]]) -> str:
    key_a = _ranking_key(team_a, lookup)
    key_b = _ranking_key(team_b, lookup)
    if key_a == key_b:
        return min(team_a, team_b)
    return team_a if key_a > key_b else team_b


def _play_deterministic_round(
    round_name: str,
    match_specs: list[tuple[int, int, int]],
    previous_winners: dict[int, str],
    lookup: dict[str, dict[str, float]],
) -> tuple[dict[int, str], list[dict[str, object]]]:
    winners = {}
    matches = []
    for match_number, source_a, source_b in match_specs:
        team_a = previous_winners[source_a]
        team_b = previous_winners[source_b]
        winner = _pick_deterministic_winner(team_a, team_b, lookup)
        winners[match_number] = winner
        matches.append({"round": round_name, "match_number": match_number, "team_a": team_a, "team_b": team_b, "winner": winner})
    return winners, matches


def build_deterministic_bracket(
    groups: dict[str, list[str]],
    probabilities: pd.DataFrame,
) -> dict[str, object]:
    lookup = _probability_lookup(probabilities)
    group_results = _build_deterministic_group_results(groups, lookup)
    finishers = get_group_finishers(group_results)
    r32_matches = build_round_of_32_matches(finishers, list(group_results["advancing_third_place_groups"]))

    r32_winners = {}
    r32_details = []
    for match_number, (team_a, team_b) in zip(ROUND_OF_32_MATCH_NUMBERS, r32_matches):
        winner = _pick_deterministic_winner(team_a, team_b, lookup)
        r32_winners[match_number] = winner
        r32_details.append({"round": "round_of_32", "match_number": match_number, "team_a": team_a, "team_b": team_b, "winner": winner})

    r16_winners, r16_details = _play_deterministic_round("round_of_16", ROUND_OF_16_MATCHES, r32_winners, lookup)
    qf_winners, qf_details = _play_deterministic_round("quarterfinal", QUARTERFINAL_MATCHES, r16_winners, lookup)
    sf_winners, sf_details = _play_deterministic_round("semifinal", SEMIFINAL_MATCHES, qf_winners, lookup)

    final_number, source_a, source_b = FINAL_MATCH
    finalist_a = sf_winners[source_a]
    finalist_b = sf_winners[source_b]
    champion = _pick_deterministic_winner(finalist_a, finalist_b, lookup)
    final_details = [{"round": "final", "match_number": final_number, "team_a": finalist_a, "team_b": finalist_b, "winner": champion}]
    return {
        "champion": champion,
        "advancing_third_place_groups": group_results["advancing_third_place_groups"],
        "round_of_32": r32_details,
        "round_of_16": r16_details,
        "quarterfinal": qf_details,
        "semifinal": sf_details,
        "final": final_details,
    }


def build_predictors() -> dict[str, ModelPredictor]:
    features = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    ratings = _rating_lookup()
    raw_results = clean_results(load_results(str(RESULTS_PATH)))
    elo_history, _ = build_elo_history(raw_results)
    draw_model = build_empirical_draw_model(elo_history)

    logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
    gradient_boosting_model = joblib.load(GRADIENT_BOOSTING_MODEL_PATH)

    logistic = _make_ml_predictor("regression", logistic_model, features, ratings)
    gradient_boosting = _make_ml_predictor("gradient_boosting", gradient_boosting_model, features, ratings)
    return {
        "regression": logistic,
        "gradient_boosting": gradient_boosting,
        "blended": _make_blend_predictor(logistic, gradient_boosting),
        "market_adjusted_wc_elo": _make_market_adjusted_elo_predictor(_market_rating_lookup(ratings), draw_model),
    }


def write_outputs(model_key: str, groups: dict[str, list[str]], predictor: ModelPredictor) -> None:
    probabilities = run_monte_carlo(groups, predictor, N_SIMULATIONS, RNG_SEED)
    bracket = build_deterministic_bracket(groups, probabilities)

    probabilities_path = OUTPUT_DIR / f"2026_{model_key}_stage_probabilities.csv"
    bracket_path = OUTPUT_DIR / f"2026_{model_key}_deterministic_bracket.json"
    probabilities.to_csv(probabilities_path, index=False)
    with bracket_path.open("w", encoding="utf-8") as f:
        json.dump(bracket, f, indent=2)

    print(f"\n{model_key}")
    print(probabilities.head(8)[["team", "champion_prob", "final_prob"]].to_string(index=False))
    print(f"Champion: {bracket['champion']}")
    print(f"Wrote {probabilities_path}")
    print(f"Wrote {bracket_path}")


def precompute_predictions(
    groups: dict[str, list[str]],
    predictors: dict[str, ModelPredictor],
    model_keys: list[str],
) -> None:
    teams = sorted({team for group_teams in groups.values() for team in group_teams})
    for model_key in model_keys:
        predictor = predictors[model_key]
        predictor.precompute(teams)
        print(f"Precomputed {model_key}: {len(predictor._cache)} cached match states")


def main() -> int:
    ensure_directories()
    groups = load_groups(str(GROUPS_PATH))
    if not _groups_are_complete(groups):
        raise ValueError(f"{GROUPS_PATH} must contain 12 complete groups of 4 teams.")

    predictors = build_predictors()
    model_keys = [
        key.strip()
        for key in os.environ.get("SIM_2026_MODELS", ",".join(DEFAULT_MODEL_KEYS)).split(",")
        if key.strip()
    ]
    precompute_predictions(groups, predictors, model_keys)
    for model_key in model_keys:
        write_outputs(model_key, groups, predictors[model_key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

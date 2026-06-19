"""World Cup 2026 tournament simulation using the Elo model only."""

from __future__ import annotations

from itertools import combinations

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
from src.predict import predict_knockout_advancement, predict_match_elo
from src.utils import normalize_probabilities


PLACEHOLDER_TERMS = ("team tbd", "tbd", "playoff winner", "play-off winner", "placeholder")
STAGE_COLUMNS = [
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
    "champion",
]


def _is_placeholder_team(team: object) -> bool:
    """Return True when a team name is empty or a placeholder."""
    if pd.isna(team):
        return True
    text = str(team).strip().lower()
    if not text:
        return True
    return any(term in text for term in PLACEHOLDER_TERMS)


def load_groups(path: str) -> dict[str, list[str]]:
    """Load World Cup groups from a CSV with columns group and team."""
    df = pd.read_csv(path)
    required_columns = {"group", "team"}
    if not required_columns.issubset(df.columns):
        raise ValueError("group fixture file must contain columns: group, team")

    groups: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        if _is_placeholder_team(row["team"]):
            continue
        group = str(row["group"]).strip().upper()
        team = str(row["team"]).strip()
        if not group:
            continue
        groups.setdefault(group, []).append(team)
    return groups


def _choice_from_distribution(distribution: dict[int, float], rng: np.random.Generator) -> int:
    values = list(distribution.keys())
    probs = normalize_probabilities(list(distribution.values()))
    return int(rng.choice(values, p=probs))


def _favorite_margin_distribution(abs_elo_gap: float) -> dict[int, float]:
    if abs_elo_gap < 100:
        return config.FAVORITE_MARGIN_PROBS["close"]
    if abs_elo_gap <= 250:
        return config.FAVORITE_MARGIN_PROBS["medium"]
    return config.FAVORITE_MARGIN_PROBS["large"]


def simulate_scoreline_from_elo(
    team_a: str,
    team_b: str,
    outcome: str,
    ratings: dict[str, float],
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Generate a simple Elo-informed scoreline for a sampled outcome."""
    if outcome not in {"team_a_win", "draw", "team_b_win"}:
        raise ValueError("outcome must be team_a_win, draw, or team_b_win")

    team_a_elo = float(ratings.get(team_a, config.INITIAL_ELO))
    team_b_elo = float(ratings.get(team_b, config.INITIAL_ELO))
    elo_diff = team_a_elo - team_b_elo

    if outcome == "draw":
        goals = _choice_from_distribution(config.DRAW_SCORELINE_PROBS, rng)
        return goals, goals

    team_a_is_winner = outcome == "team_a_win"
    winner_is_favorite = (team_a_is_winner and elo_diff >= 0) or (not team_a_is_winner and elo_diff <= 0)
    if winner_is_favorite:
        margin_probs = _favorite_margin_distribution(abs(elo_diff))
    else:
        margin_probs = config.UNDERDOG_MARGIN_PROBS

    margin = _choice_from_distribution(margin_probs, rng)
    loser_goals = _choice_from_distribution(config.LOSER_GOALS_PROBS, rng)
    winner_goals = loser_goals + margin
    if team_a_is_winner:
        return winner_goals, loser_goals
    return loser_goals, winner_goals


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
    """Update group table stats for one simulated match."""
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


def _simulate_group(
    group: str,
    teams: list[str],
    ratings: dict[str, float],
    rng: np.random.Generator,
    draw_model: dict[str, object] | None = None,
) -> dict:
    if len(teams) != 4:
        raise ValueError(f"group {group} must contain exactly 4 non-placeholder teams; found {len(teams)}")

    stats = _empty_group_stats(teams)
    matches = []
    for team_a, team_b in combinations(teams, 2):
        prediction = predict_match_elo(team_a, team_b, ratings, neutral=True, draw_model=draw_model)
        outcome = str(
            rng.choice(
                ["team_a_win", "draw", "team_b_win"],
                p=[
                    float(prediction["team_a_win_prob"]),
                    float(prediction["draw_prob"]),
                    float(prediction["team_b_win_prob"]),
                ],
            )
        )
        goals_a, goals_b = simulate_scoreline_from_elo(team_a, team_b, outcome, ratings, rng)
        _record_match(stats, team_a, team_b, goals_a, goals_b)
        matches.append({"team_a": team_a, "team_b": team_b, "goals_a": goals_a, "goals_b": goals_b})

    standings = pd.DataFrame(stats.values())
    standings["random_tiebreaker"] = rng.random(len(standings))
    standings = standings.sort_values(
        ["points", "goal_difference", "goals_for", "random_tiebreaker"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    standings = standings.drop(columns=["random_tiebreaker"])

    return {"standings": standings.to_dict("records"), "matches": matches}


def simulate_group_stage(
    groups: dict[str, list[str]],
    ratings: dict[str, float],
    rng: np.random.Generator,
    draw_model: dict[str, object] | None = None,
) -> tuple[list[str], dict]:
    """Simulate the group stage and return Round of 32 teams plus details."""
    group_results: dict[str, object] = {}
    third_place_records = []
    round_of_32_teams: list[str] = []

    for group in sorted(groups):
        result = _simulate_group(group, groups[group], ratings, rng, draw_model=draw_model)
        group_results[group] = result
        standings = result["standings"]
        round_of_32_teams.extend([standings[0]["team"], standings[1]["team"]])
        third_record = dict(standings[2])
        third_record["group"] = group
        third_record["random_tiebreaker"] = float(rng.random())
        third_place_records.append(third_record)

    third_df = pd.DataFrame(third_place_records)
    third_df = third_df.sort_values(
        ["points", "goal_difference", "goals_for", "random_tiebreaker"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best_thirds = third_df.head(8)
    round_of_32_teams.extend(best_thirds["team"].tolist())
    group_results["advancing_third_place_groups"] = best_thirds["group"].tolist()
    return round_of_32_teams, group_results


def _simulate_knockout_match(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    rng: np.random.Generator,
    draw_model: dict[str, object] | None = None,
) -> dict[str, object]:
    prediction = predict_knockout_advancement(team_a, team_b, ratings, neutral=True, draw_model=draw_model)
    winner = str(
        rng.choice(
            [team_a, team_b],
            p=[float(prediction["team_a_advancement_prob"]), float(prediction["team_b_advancement_prob"])],
        )
    )
    return {"team_a": team_a, "team_b": team_b, "winner": winner, "prediction": prediction}


def _simulate_numbered_round(
    match_specs: list[tuple[int, int, int]],
    previous_winners: dict[int, str],
    ratings: dict[str, float],
    rng: np.random.Generator,
    draw_model: dict[str, object] | None = None,
) -> tuple[dict[int, str], list[dict[str, object]]]:
    round_winners: dict[int, str] = {}
    round_matches = []
    for match_number, source_a, source_b in match_specs:
        team_a = previous_winners[source_a]
        team_b = previous_winners[source_b]
        match = _simulate_knockout_match(team_a, team_b, ratings, rng, draw_model=draw_model)
        match["match_number"] = match_number
        match["source_match_a"] = source_a
        match["source_match_b"] = source_b
        round_winners[match_number] = str(match["winner"])
        round_matches.append(match)
    return round_winners, round_matches


def simulate_knockout(
    group_results: dict,
    ratings: dict[str, float],
    rng: np.random.Generator,
    draw_model: dict[str, object] | None = None,
) -> dict:
    """Simulate knockout stages from a slot-based Round of 32 bracket."""
    group_finishers = get_group_finishers(group_results)
    advancing_third_groups = list(group_results["advancing_third_place_groups"])
    round_of_32_matches = build_round_of_32_matches(group_finishers, advancing_third_groups)

    r32_winners: dict[int, str] = {}
    r32_match_details = []
    for match_number, (team_a, team_b) in zip(ROUND_OF_32_MATCH_NUMBERS, round_of_32_matches):
        match = _simulate_knockout_match(team_a, team_b, ratings, rng, draw_model=draw_model)
        match["match_number"] = match_number
        r32_winners[match_number] = str(match["winner"])
        r32_match_details.append(match)

    r16_winners, r16_matches = _simulate_numbered_round(
        ROUND_OF_16_MATCHES,
        r32_winners,
        ratings,
        rng,
        draw_model=draw_model,
    )
    qf_winners, qf_matches = _simulate_numbered_round(
        QUARTERFINAL_MATCHES,
        r16_winners,
        ratings,
        rng,
        draw_model=draw_model,
    )
    sf_winners, sf_matches = _simulate_numbered_round(
        SEMIFINAL_MATCHES,
        qf_winners,
        ratings,
        rng,
        draw_model=draw_model,
    )

    final_number, semifinal_a, semifinal_b = FINAL_MATCH
    final_match = _simulate_knockout_match(
        sf_winners[semifinal_a],
        sf_winners[semifinal_b],
        ratings,
        rng,
        draw_model=draw_model,
    )
    final_match["match_number"] = final_number
    final_match["source_match_a"] = semifinal_a
    final_match["source_match_b"] = semifinal_b

    return {
        "round_of_16": list(r32_winners.values()),
        "quarterfinal": list(r16_winners.values()),
        "semifinal": list(qf_winners.values()),
        "final": list(sf_winners.values()),
        "champion": str(final_match["winner"]),
        "bracket": {
            "round_of_32": r32_match_details,
            "round_of_16": r16_matches,
            "quarterfinal": qf_matches,
            "semifinal": sf_matches,
            "final": [final_match],
        },
    }


def simulate_tournament(
    groups: dict[str, list[str]],
    ratings: dict[str, float],
    rng: np.random.Generator,
    draw_model: dict[str, object] | None = None,
) -> dict:
    """Simulate one full tournament."""
    round_of_32, group_results = simulate_group_stage(groups, ratings, rng, draw_model=draw_model)
    knockout = simulate_knockout(group_results, ratings, rng, draw_model=draw_model)
    return {
        "round_of_32": round_of_32,
        "round_of_16": knockout["round_of_16"],
        "quarterfinal": knockout["quarterfinal"],
        "semifinal": knockout["semifinal"],
        "final": knockout["final"],
        "champion": knockout["champion"],
        "group_results": group_results,
        "bracket": knockout["bracket"],
    }


def run_monte_carlo(
    groups: dict[str, list[str]],
    ratings: dict[str, float],
    n_simulations: int = 10000,
    seed: int | None = None,
    draw_model: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run repeated tournament simulations and return stage probabilities."""
    rng = np.random.default_rng(seed)
    teams = sorted({team for group_teams in groups.values() for team in group_teams})
    counts = {team: {stage: 0 for stage in STAGE_COLUMNS} for team in teams}

    for _ in range(n_simulations):
        result = simulate_tournament(groups, ratings, rng, draw_model=draw_model)
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

"""Simulate the 2026 World Cup with the separate Poisson goal model."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

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
from src.elo import build_elo_history
from src.poisson_model import (
    fit_average_total_goals,
    fit_goal_profile_state,
    poisson_outcome_probabilities,
)
from src.predict import expected_score
from src.simulate import STAGE_COLUMNS, load_groups
from src.utils import ensure_directories, normalize_probabilities


RESULTS_PATH = Path("data/raw/results.csv")
GROUPS_PATH = Path("data/fixtures/world_cup_2026_groups.csv")
OUTPUT_PATH = Path("results/poisson_team_stage_probabilities.csv")

# Best rolling-validation decision-accuracy setting found so far.
POISSON_SIM_DRAW_INFLATION = 1.20
POISSON_SIM_GOAL_PROFILE_WEIGHT = 0.50
N_SIMULATIONS = 1000


def _groups_are_complete(groups: dict[str, list[str]]) -> bool:
    return set(groups) == set("ABCDEFGHIJKL") and all(len(teams) == 4 for teams in groups.values())


def _poisson_prediction(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    average_total_goals: float,
    goal_profile_state: dict[str, dict[str, float]],
) -> dict[str, float]:
    rating_a = float(ratings.get(team_a, config.INITIAL_ELO))
    rating_b = float(ratings.get(team_b, config.INITIAL_ELO))
    adjusted_elo_diff = rating_a - rating_b
    return poisson_outcome_probabilities(
        expected_score(rating_a, rating_b),
        average_total_goals,
        adjusted_elo_diff=adjusted_elo_diff,
        goal_profile_state=goal_profile_state,
        home_team=team_a,
        away_team=team_b,
        draw_inflation=POISSON_SIM_DRAW_INFLATION,
        goal_profile_weight=POISSON_SIM_GOAL_PROFILE_WEIGHT,
    )


def _sample_poisson_scoreline(
    prediction: dict[str, float],
    rng: np.random.Generator,
) -> tuple[int, int]:
    home_goals = int(rng.poisson(float(prediction["home_expected_goals"])))
    away_goals = int(rng.poisson(float(prediction["away_expected_goals"])))
    return min(home_goals, 10), min(away_goals, 10)


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


def _simulate_group(
    group: str,
    teams: list[str],
    ratings: dict[str, float],
    average_total_goals: float,
    goal_profile_state: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> dict[str, object]:
    if len(teams) != 4:
        raise ValueError(f"group {group} must have exactly 4 teams")

    stats = _empty_group_stats(teams)
    matches = []
    for team_a, team_b in combinations(teams, 2):
        prediction = _poisson_prediction(team_a, team_b, ratings, average_total_goals, goal_profile_state)
        goals_a, goals_b = _sample_poisson_scoreline(prediction, rng)
        _record_match(stats, team_a, team_b, goals_a, goals_b)
        matches.append({"team_a": team_a, "team_b": team_b, "goals_a": goals_a, "goals_b": goals_b})

    standings = pd.DataFrame(stats.values())
    standings["random_tiebreaker"] = rng.random(len(standings))
    standings = standings.sort_values(
        ["points", "goal_difference", "goals_for", "random_tiebreaker"],
        ascending=[False, False, False, True],
    ).drop(columns=["random_tiebreaker"])
    return {"standings": standings.to_dict("records"), "matches": matches}


def _simulate_group_stage(
    groups: dict[str, list[str]],
    ratings: dict[str, float],
    average_total_goals: float,
    goal_profile_state: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> tuple[list[str], dict[str, object]]:
    group_results: dict[str, object] = {}
    third_place_records = []
    round_of_32_teams = []

    for group in sorted(groups):
        result = _simulate_group(group, groups[group], ratings, average_total_goals, goal_profile_state, rng)
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
    )
    best_thirds = third_df.head(8)
    round_of_32_teams.extend(best_thirds["team"].tolist())
    group_results["advancing_third_place_groups"] = best_thirds["group"].tolist()
    return round_of_32_teams, group_results


def _knockout_advancement_probs(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    average_total_goals: float,
    goal_profile_state: dict[str, dict[str, float]],
) -> tuple[float, float, dict[str, float]]:
    prediction = _poisson_prediction(team_a, team_b, ratings, average_total_goals, goal_profile_state)
    team_a_adv = float(prediction["home_win_prob"]) + float(prediction["draw_prob"]) / 2.0
    team_b_adv = float(prediction["away_win_prob"]) + float(prediction["draw_prob"]) / 2.0
    team_a_adv, team_b_adv = normalize_probabilities([team_a_adv, team_b_adv])
    return team_a_adv, team_b_adv, prediction


def _simulate_knockout_match(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    average_total_goals: float,
    goal_profile_state: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> dict[str, object]:
    team_a_adv, team_b_adv, prediction = _knockout_advancement_probs(
        team_a,
        team_b,
        ratings,
        average_total_goals,
        goal_profile_state,
    )
    winner = str(rng.choice([team_a, team_b], p=[team_a_adv, team_b_adv]))
    return {"team_a": team_a, "team_b": team_b, "winner": winner, "prediction": prediction}


def _simulate_numbered_round(
    match_specs: list[tuple[int, int, int]],
    previous_winners: dict[int, str],
    ratings: dict[str, float],
    average_total_goals: float,
    goal_profile_state: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> tuple[dict[int, str], list[dict[str, object]]]:
    winners = {}
    matches = []
    for match_number, source_a, source_b in match_specs:
        match = _simulate_knockout_match(
            previous_winners[source_a],
            previous_winners[source_b],
            ratings,
            average_total_goals,
            goal_profile_state,
            rng,
        )
        match["match_number"] = match_number
        winners[match_number] = str(match["winner"])
        matches.append(match)
    return winners, matches


def _simulate_knockout(
    group_results: dict[str, object],
    ratings: dict[str, float],
    average_total_goals: float,
    goal_profile_state: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> dict[str, object]:
    finishers = get_group_finishers(group_results)
    round_of_32_matches = build_round_of_32_matches(
        finishers,
        list(group_results["advancing_third_place_groups"]),
    )

    r32_winners = {}
    r32_details = []
    for match_number, (team_a, team_b) in zip(ROUND_OF_32_MATCH_NUMBERS, round_of_32_matches):
        match = _simulate_knockout_match(team_a, team_b, ratings, average_total_goals, goal_profile_state, rng)
        match["match_number"] = match_number
        r32_winners[match_number] = str(match["winner"])
        r32_details.append(match)

    r16_winners, r16_matches = _simulate_numbered_round(
        ROUND_OF_16_MATCHES, r32_winners, ratings, average_total_goals, goal_profile_state, rng
    )
    qf_winners, qf_matches = _simulate_numbered_round(
        QUARTERFINAL_MATCHES, r16_winners, ratings, average_total_goals, goal_profile_state, rng
    )
    sf_winners, sf_matches = _simulate_numbered_round(
        SEMIFINAL_MATCHES, qf_winners, ratings, average_total_goals, goal_profile_state, rng
    )

    final_number, source_a, source_b = FINAL_MATCH
    final_match = _simulate_knockout_match(
        sf_winners[source_a],
        sf_winners[source_b],
        ratings,
        average_total_goals,
        goal_profile_state,
        rng,
    )
    final_match["match_number"] = final_number
    return {
        "round_of_16": list(r32_winners.values()),
        "quarterfinal": list(r16_winners.values()),
        "semifinal": list(qf_winners.values()),
        "final": list(sf_winners.values()),
        "champion": str(final_match["winner"]),
        "bracket": {
            "round_of_32": r32_details,
            "round_of_16": r16_matches,
            "quarterfinal": qf_matches,
            "semifinal": sf_matches,
            "final": [final_match],
        },
    }


def _simulate_tournament(
    groups: dict[str, list[str]],
    ratings: dict[str, float],
    average_total_goals: float,
    goal_profile_state: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> dict[str, object]:
    round_of_32, group_results = _simulate_group_stage(
        groups,
        ratings,
        average_total_goals,
        goal_profile_state,
        rng,
    )
    knockout = _simulate_knockout(group_results, ratings, average_total_goals, goal_profile_state, rng)
    return {
        "round_of_32": round_of_32,
        "round_of_16": knockout["round_of_16"],
        "quarterfinal": knockout["quarterfinal"],
        "semifinal": knockout["semifinal"],
        "final": knockout["final"],
        "champion": knockout["champion"],
    }


def _run_monte_carlo(
    groups: dict[str, list[str]],
    ratings: dict[str, float],
    average_total_goals: float,
    goal_profile_state: dict[str, dict[str, float]],
    n_simulations: int,
    seed: int | None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = sorted({team for group_teams in groups.values() for team in group_teams})
    counts = {team: {stage: 0 for stage in STAGE_COLUMNS} for team in teams}

    for _ in range(n_simulations):
        result = _simulate_tournament(groups, ratings, average_total_goals, goal_profile_state, rng)
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


def main() -> int:
    """Run the Poisson 2026 World Cup simulation."""
    ensure_directories()
    groups = load_groups(str(GROUPS_PATH))
    if not _groups_are_complete(groups):
        print(f"{GROUPS_PATH} must contain 12 complete groups of 4 teams.")
        return 1

    matches = clean_results(load_results(str(RESULTS_PATH)))
    elo_history, ratings = build_elo_history(matches)
    average_total_goals = fit_average_total_goals(elo_history)
    goal_profile_state = fit_goal_profile_state(matches)

    probabilities = _run_monte_carlo(
        groups,
        ratings,
        average_total_goals,
        goal_profile_state,
        n_simulations=N_SIMULATIONS,
        seed=config.MONTE_CARLO_SEED,
    )
    probabilities.to_csv(OUTPUT_PATH, index=False)

    print("Top 12 teams by Poisson World Cup champion probability:")
    print(probabilities.head(12)[["team", "champion_prob"]].to_string(index=False))
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

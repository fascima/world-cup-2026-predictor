"""Build a deterministic 2026 bracket from Poisson stage probabilities.

This script does not simulate matches. It reads
``results/poisson_team_stage_probabilities.csv`` and always advances the team
with the highest probability ranking.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.bracket import (
    FINAL_MATCH,
    QUARTERFINAL_MATCHES,
    ROUND_OF_16_MATCHES,
    ROUND_OF_32_MATCH_NUMBERS,
    SEMIFINAL_MATCHES,
    build_round_of_32_matches,
    get_group_finishers,
)
from src.simulate import load_groups
from src.utils import ensure_directories


GROUPS_PATH = Path("data/fixtures/world_cup_2026_groups.csv")
PROBABILITIES_PATH = Path("results/poisson_team_stage_probabilities.csv")
GROUP_RANKINGS_OUTPUT_PATH = Path("results/deterministic_poisson_group_rankings.csv")
BRACKET_OUTPUT_PATH = Path("results/deterministic_poisson_bracket.json")

RANKING_COLUMNS = [
    "champion_prob",
    "final_prob",
    "semifinal_prob",
    "quarterfinal_prob",
    "round_of_16_prob",
    "round_of_32_prob",
]


def _load_probability_table(path: Path) -> pd.DataFrame:
    """Load team stage probabilities and validate required columns."""
    probabilities = pd.read_csv(path)
    required_columns = {"team", *RANKING_COLUMNS}
    missing = required_columns - set(probabilities.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return probabilities


def _probability_lookup(probabilities: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return probability rows keyed by team."""
    return {
        str(row["team"]): {column: float(row[column]) for column in RANKING_COLUMNS}
        for _, row in probabilities.iterrows()
    }


def _ranking_key(team: str, lookup: dict[str, dict[str, float]]) -> tuple[float, ...]:
    """Return a descending ranking key for a team."""
    values = lookup.get(team)
    if values is None:
        return tuple(0.0 for _ in RANKING_COLUMNS)
    return tuple(values[column] for column in RANKING_COLUMNS)


def _rank_group(
    group: str,
    teams: list[str],
    lookup: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    """Rank one group by the configured probability columns."""
    ranked_teams = sorted(teams, key=lambda team: (_ranking_key(team, lookup), team), reverse=True)
    rows = []
    for rank, team in enumerate(ranked_teams, start=1):
        row: dict[str, object] = {
            "group": group,
            "rank": rank,
            "team": team,
        }
        row.update(lookup.get(team, {column: 0.0 for column in RANKING_COLUMNS}))
        rows.append(row)
    return rows


def build_deterministic_group_results(
    groups: dict[str, list[str]],
    lookup: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build group rankings and bracket-compatible group results."""
    ranking_rows = []
    group_results: dict[str, object] = {}
    third_place_rows = []

    for group in sorted(groups):
        rows = _rank_group(group, groups[group], lookup)
        ranking_rows.extend(rows)
        standings = []
        for row in rows:
            standings.append(
                {
                    "team": row["team"],
                    "points": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "goal_difference": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "probability_rank": row["rank"],
                }
            )
        group_results[group] = {"standings": standings, "matches": []}
        third_place = dict(rows[2])
        third_place_rows.append(third_place)

    third_place_rows = sorted(
        third_place_rows,
        key=lambda row: (tuple(float(row[column]) for column in RANKING_COLUMNS), str(row["team"])),
        reverse=True,
    )
    advancing_third_place_groups = [str(row["group"]) for row in third_place_rows[:8]]
    group_results["advancing_third_place_groups"] = advancing_third_place_groups
    return pd.DataFrame(ranking_rows), group_results


def _pick_winner(team_a: str, team_b: str, lookup: dict[str, dict[str, float]]) -> str:
    """Pick the higher-probability team."""
    key_a = _ranking_key(team_a, lookup)
    key_b = _ranking_key(team_b, lookup)
    if key_a == key_b:
        return min(team_a, team_b)
    return team_a if key_a > key_b else team_b


def _play_numbered_round(
    round_name: str,
    match_specs: list[tuple[int, int, int]],
    previous_winners: dict[int, str],
    lookup: dict[str, dict[str, float]],
) -> tuple[dict[int, str], list[dict[str, object]]]:
    """Play one deterministic knockout round."""
    winners = {}
    matches = []
    for match_number, source_a, source_b in match_specs:
        team_a = previous_winners[source_a]
        team_b = previous_winners[source_b]
        winner = _pick_winner(team_a, team_b, lookup)
        winners[match_number] = winner
        matches.append(
            {
                "round": round_name,
                "match_number": match_number,
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner,
                "team_a_champion_prob": lookup[team_a]["champion_prob"],
                "team_b_champion_prob": lookup[team_b]["champion_prob"],
            }
        )
    return winners, matches


def build_deterministic_bracket(
    group_results: dict[str, object],
    lookup: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Build the deterministic knockout bracket."""
    finishers = get_group_finishers(group_results)
    r32_matches = build_round_of_32_matches(
        finishers,
        list(group_results["advancing_third_place_groups"]),
    )

    r32_winners = {}
    r32_details = []
    for match_number, (team_a, team_b) in zip(ROUND_OF_32_MATCH_NUMBERS, r32_matches):
        winner = _pick_winner(team_a, team_b, lookup)
        r32_winners[match_number] = winner
        r32_details.append(
            {
                "round": "round_of_32",
                "match_number": match_number,
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner,
                "team_a_champion_prob": lookup[team_a]["champion_prob"],
                "team_b_champion_prob": lookup[team_b]["champion_prob"],
            }
        )

    r16_winners, r16_details = _play_numbered_round("round_of_16", ROUND_OF_16_MATCHES, r32_winners, lookup)
    qf_winners, qf_details = _play_numbered_round("quarterfinal", QUARTERFINAL_MATCHES, r16_winners, lookup)
    sf_winners, sf_details = _play_numbered_round("semifinal", SEMIFINAL_MATCHES, qf_winners, lookup)

    final_number, source_a, source_b = FINAL_MATCH
    finalist_a = sf_winners[source_a]
    finalist_b = sf_winners[source_b]
    champion = _pick_winner(finalist_a, finalist_b, lookup)
    final_details = [
        {
            "round": "final",
            "match_number": final_number,
            "team_a": finalist_a,
            "team_b": finalist_b,
            "winner": champion,
            "team_a_champion_prob": lookup[finalist_a]["champion_prob"],
            "team_b_champion_prob": lookup[finalist_b]["champion_prob"],
        }
    ]

    return {
        "champion": champion,
        "advancing_third_place_groups": group_results["advancing_third_place_groups"],
        "round_of_32": r32_details,
        "round_of_16": r16_details,
        "quarterfinal": qf_details,
        "semifinal": sf_details,
        "final": final_details,
    }


def main() -> int:
    """Create deterministic group rankings and bracket outputs."""
    ensure_directories()
    groups = load_groups(str(GROUPS_PATH))
    probabilities = _load_probability_table(PROBABILITIES_PATH)
    lookup = _probability_lookup(probabilities)

    missing_teams = sorted({team for teams in groups.values() for team in teams} - set(lookup))
    if missing_teams:
        raise ValueError(
            "The probability file is missing teams from the fixture file: "
            + ", ".join(missing_teams)
        )

    group_rankings, group_results = build_deterministic_group_results(groups, lookup)
    bracket = build_deterministic_bracket(group_results, lookup)

    group_rankings.to_csv(GROUP_RANKINGS_OUTPUT_PATH, index=False)
    with BRACKET_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(bracket, f, indent=2)

    print("Deterministic group rankings:")
    print(group_rankings[["group", "rank", "team", "champion_prob"]].to_string(index=False))
    print("\nDeterministic knockout bracket winners:")
    for round_name in ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"]:
        print(round_name)
        for match in bracket[round_name]:
            print(f"  {match['team_a']} vs {match['team_b']} -> {match['winner']}")
    print(f"\nChampion: {bracket['champion']}")
    print(f"Wrote {GROUP_RANKINGS_OUTPUT_PATH}")
    print(f"Wrote {BRACKET_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

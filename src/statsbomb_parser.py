"""StatsBomb Open Data parsing utilities."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_STATSBOMB_ROOT = Path("data/external/statsbomb_open_data")
DEFAULT_OUTPUT_PATH = Path("data/processed/statsbomb_team_match_features.csv")

NON_PENALTY_SHOT_TYPES = {"Open Play", "Free Kick", "Corner", "Throw-in", "Kick Off"}
SET_PIECE_PATTERNS = {
    "From Corner",
    "From Free Kick",
    "From Throw In",
    "From Goal Kick",
    "From Keeper",
}
SHOT_ON_TARGET_OUTCOMES = {"Goal", "Saved", "Saved To Post"}
SUCCESSFUL_DRIBBLE_OUTCOMES = {"Complete"}
WON_DUEL_OUTCOMES = {"Won", "Success In Play", "Success Out"}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _nested_name(value: dict[str, Any] | None, default: str = "") -> str:
    if not isinstance(value, dict):
        return default
    name = value.get("name", default)
    return str(name) if name is not None else default


def _metadata_team_name(match: dict[str, Any], side: str) -> str:
    team = match.get(f"{side}_team", {})
    if not isinstance(team, dict):
        return ""
    value = team.get(f"{side}_team_name", team.get("name", ""))
    return str(value) if value is not None else ""


def _metadata_name(value: dict[str, Any] | None, key: str, default: str = "") -> str:
    if not isinstance(value, dict):
        return default
    name = value.get(key, value.get("name", default))
    return str(name) if name is not None else default


def _location_x(event: dict[str, Any], key: str = "location") -> float | None:
    value = event.get(key)
    if isinstance(value, list) and value:
        try:
            return float(value[0])
        except (TypeError, ValueError):
            return None
    return None


def _end_location_x(payload: dict[str, Any], key: str = "end_location") -> float | None:
    value = payload.get(key)
    if isinstance(value, list) and value:
        try:
            return float(value[0])
        except (TypeError, ValueError):
            return None
    return None


def _end_location_y(payload: dict[str, Any], key: str = "end_location") -> float | None:
    value = payload.get(key)
    if isinstance(value, list) and len(value) > 1:
        try:
            return float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _is_box_entry(start_x: float | None, end_x: float | None, end_y: float | None) -> bool:
    if start_x is None or end_x is None or end_y is None:
        return False
    return start_x < 102.0 <= end_x and 18.0 <= end_y <= 62.0


def _is_final_third_entry(start_x: float | None, end_x: float | None) -> bool:
    if start_x is None or end_x is None:
        return False
    return start_x < 80.0 <= end_x


def _empty_stats() -> dict[str, float]:
    return {
        "events": 0.0,
        "possessions": 0.0,
        "passes": 0.0,
        "completed_passes": 0.0,
        "progressive_passes": 0.0,
        "final_third_entries": 0.0,
        "box_entries": 0.0,
        "crosses": 0.0,
        "completed_crosses": 0.0,
        "through_balls": 0.0,
        "switches": 0.0,
        "key_passes": 0.0,
        "assists": 0.0,
        "carries": 0.0,
        "progressive_carries": 0.0,
        "shots": 0.0,
        "shots_on_target": 0.0,
        "penalty_shots": 0.0,
        "goals_from_shots": 0.0,
        "own_goals_for": 0.0,
        "own_goals_against": 0.0,
        "xg": 0.0,
        "non_penalty_xg": 0.0,
        "open_play_xg": 0.0,
        "set_piece_xg": 0.0,
        "penalty_xg": 0.0,
        "free_kick_xg": 0.0,
        "headers": 0.0,
        "pressures": 0.0,
        "counterpressures": 0.0,
        "ball_recoveries": 0.0,
        "interceptions": 0.0,
        "clearances": 0.0,
        "blocks": 0.0,
        "duels": 0.0,
        "duels_won": 0.0,
        "dribbles": 0.0,
        "successful_dribbles": 0.0,
        "fouls_committed": 0.0,
        "fouls_won": 0.0,
        "miscontrols": 0.0,
        "dispossessed": 0.0,
        "goalkeeper_events": 0.0,
        "substitutions": 0.0,
    }


def load_match_metadata(statsbomb_root: Path = DEFAULT_STATSBOMB_ROOT) -> dict[int, dict[str, Any]]:
    """Load all available StatsBomb match metadata under the local data root."""
    matches_root = statsbomb_root / "matches"
    if not matches_root.exists():
        raise FileNotFoundError(f"StatsBomb matches directory not found: {matches_root}")

    metadata: dict[int, dict[str, Any]] = {}
    for path in sorted(matches_root.glob("*/*.json")):
        for match in _load_json(path):
            match_id = int(match["match_id"])
            metadata[match_id] = match
    return metadata


def _match_team_names(match: dict[str, Any], events: list[dict[str, Any]]) -> list[str]:
    home_team = _metadata_team_name(match, "home")
    away_team = _metadata_team_name(match, "away")
    teams = [team for team in [home_team, away_team] if team]
    if len(teams) == 2:
        return teams

    event_teams = sorted(
        {
            _nested_name(event.get("team"))
            for event in events
            if _nested_name(event.get("team"))
        }
    )
    if len(event_teams) != 2:
        raise ValueError(f"Expected two teams from match/events, got {event_teams}")
    return event_teams


def _aggregate_events(events: list[dict[str, Any]], teams: list[str]) -> dict[str, dict[str, float]]:
    stats = {team: _empty_stats() for team in teams}
    possession_ids: dict[str, set[int]] = {team: set() for team in teams}

    for event in events:
        if int(event.get("period", 0) or 0) > 4:
            continue

        team = _nested_name(event.get("team"))
        if team not in stats:
            continue

        team_stats = stats[team]
        event_type = _nested_name(event.get("type"))
        play_pattern = _nested_name(event.get("play_pattern"))
        possession_team = _nested_name(event.get("possession_team"))

        team_stats["events"] += 1.0
        if possession_team == team and event.get("possession") is not None:
            possession_ids[team].add(int(event["possession"]))
        if event.get("counterpress"):
            team_stats["counterpressures"] += 1.0

        if event_type == "Pass":
            pass_payload = event.get("pass", {})
            if not isinstance(pass_payload, dict):
                pass_payload = {}
            outcome = _nested_name(pass_payload.get("outcome"), default="Complete")
            is_complete = outcome == "Complete"
            start_x = _location_x(event)
            end_x = _end_location_x(pass_payload)
            end_y = _end_location_y(pass_payload)

            team_stats["passes"] += 1.0
            if is_complete:
                team_stats["completed_passes"] += 1.0
            if start_x is not None and end_x is not None and end_x - start_x >= 20.0:
                team_stats["progressive_passes"] += 1.0
            if is_complete and _is_final_third_entry(start_x, end_x):
                team_stats["final_third_entries"] += 1.0
            if is_complete and _is_box_entry(start_x, end_x, end_y):
                team_stats["box_entries"] += 1.0
            if pass_payload.get("cross"):
                team_stats["crosses"] += 1.0
                if is_complete:
                    team_stats["completed_crosses"] += 1.0
            if pass_payload.get("through_ball"):
                team_stats["through_balls"] += 1.0
            if pass_payload.get("switch"):
                team_stats["switches"] += 1.0
            if pass_payload.get("shot_assist"):
                team_stats["key_passes"] += 1.0
            if pass_payload.get("goal_assist"):
                team_stats["assists"] += 1.0
            continue

        if event_type == "Carry":
            carry_payload = event.get("carry", {})
            if not isinstance(carry_payload, dict):
                carry_payload = {}
            start_x = _location_x(event)
            end_x = _end_location_x(carry_payload)
            end_y = _end_location_y(carry_payload)

            team_stats["carries"] += 1.0
            if start_x is not None and end_x is not None and end_x - start_x >= 15.0:
                team_stats["progressive_carries"] += 1.0
            if _is_final_third_entry(start_x, end_x):
                team_stats["final_third_entries"] += 1.0
            if _is_box_entry(start_x, end_x, end_y):
                team_stats["box_entries"] += 1.0
            continue

        if event_type == "Shot":
            shot_payload = event.get("shot", {})
            if not isinstance(shot_payload, dict):
                shot_payload = {}
            shot_type = _nested_name(shot_payload.get("type"))
            outcome = _nested_name(shot_payload.get("outcome"))
            body_part = _nested_name(shot_payload.get("body_part"))
            xg = float(shot_payload.get("statsbomb_xg", 0.0) or 0.0)

            team_stats["shots"] += 1.0
            team_stats["xg"] += xg
            if shot_type != "Penalty":
                team_stats["non_penalty_xg"] += xg
            if shot_type == "Penalty":
                team_stats["penalty_shots"] += 1.0
                team_stats["penalty_xg"] += xg
            if shot_type == "Free Kick":
                team_stats["free_kick_xg"] += xg
            if shot_type == "Open Play" and play_pattern not in SET_PIECE_PATTERNS:
                team_stats["open_play_xg"] += xg
            if shot_type in NON_PENALTY_SHOT_TYPES and play_pattern in SET_PIECE_PATTERNS:
                team_stats["set_piece_xg"] += xg
            if outcome in SHOT_ON_TARGET_OUTCOMES:
                team_stats["shots_on_target"] += 1.0
            if outcome == "Goal":
                team_stats["goals_from_shots"] += 1.0
            if body_part == "Head":
                team_stats["headers"] += 1.0
            continue

        if event_type == "Own Goal For":
            team_stats["own_goals_for"] += 1.0
        elif event_type == "Own Goal Against":
            team_stats["own_goals_against"] += 1.0
        elif event_type == "Pressure":
            team_stats["pressures"] += 1.0
        elif event_type == "Ball Recovery":
            team_stats["ball_recoveries"] += 1.0
        elif event_type == "Interception":
            team_stats["interceptions"] += 1.0
        elif event_type == "Clearance":
            team_stats["clearances"] += 1.0
        elif event_type == "Block":
            team_stats["blocks"] += 1.0
        elif event_type == "Duel":
            duel_payload = event.get("duel", {})
            if not isinstance(duel_payload, dict):
                duel_payload = {}
            team_stats["duels"] += 1.0
            if _nested_name(duel_payload.get("outcome")) in WON_DUEL_OUTCOMES:
                team_stats["duels_won"] += 1.0
        elif event_type == "Dribble":
            dribble_payload = event.get("dribble", {})
            if not isinstance(dribble_payload, dict):
                dribble_payload = {}
            team_stats["dribbles"] += 1.0
            if _nested_name(dribble_payload.get("outcome")) in SUCCESSFUL_DRIBBLE_OUTCOMES:
                team_stats["successful_dribbles"] += 1.0
        elif event_type == "Foul Committed":
            team_stats["fouls_committed"] += 1.0
        elif event_type == "Foul Won":
            team_stats["fouls_won"] += 1.0
        elif event_type == "Miscontrol":
            team_stats["miscontrols"] += 1.0
        elif event_type == "Dispossessed":
            team_stats["dispossessed"] += 1.0
        elif event_type == "Goal Keeper":
            team_stats["goalkeeper_events"] += 1.0
        elif event_type == "Substitution":
            team_stats["substitutions"] += 1.0

    for team in teams:
        stats[team]["possessions"] = float(len(possession_ids[team]))

    return stats


def _result_label(team_score: int, opponent_score: int) -> str:
    if team_score > opponent_score:
        return "win"
    if team_score == opponent_score:
        return "draw"
    return "loss"


def _points(team_score: int, opponent_score: int) -> int:
    if team_score > opponent_score:
        return 3
    if team_score == opponent_score:
        return 1
    return 0


def _rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _team_match_base(match: dict[str, Any], team: str, opponent: str) -> dict[str, Any]:
    home_team = _metadata_team_name(match, "home")
    away_team = _metadata_team_name(match, "away")
    is_home = team == home_team
    team_score = int(match.get("home_score", 0) if is_home else match.get("away_score", 0))
    opponent_score = int(match.get("away_score", 0) if is_home else match.get("home_score", 0))
    competition = match.get("competition", {})
    season = match.get("season", {})
    stage = match.get("competition_stage", {})

    return {
        "match_id": int(match["match_id"]),
        "match_date": match.get("match_date"),
        "kick_off": match.get("kick_off"),
        "competition_id": competition.get("competition_id"),
        "competition_name": _metadata_name(competition, "competition_name"),
        "season_id": season.get("season_id"),
        "season_name": _metadata_name(season, "season_name"),
        "competition_stage": _metadata_name(stage, "name"),
        "match_week": match.get("match_week"),
        "team": team,
        "opponent": opponent,
        "is_home": is_home,
        "team_score": team_score,
        "opponent_score": opponent_score,
        "goal_difference": team_score - opponent_score,
        "result": _result_label(team_score, opponent_score),
        "points": _points(team_score, opponent_score),
    }


def _derived_metrics(row: dict[str, Any]) -> None:
    row["pass_completion_rate"] = _rate(row["completed_passes"], row["passes"])
    row["cross_completion_rate"] = _rate(row["completed_crosses"], row["crosses"])
    row["shots_on_target_rate"] = _rate(row["shots_on_target"], row["shots"])
    row["xg_per_shot"] = _rate(row["xg"], row["shots"])
    row["non_penalty_xg_per_shot"] = _rate(row["non_penalty_xg"], row["shots"] - row["penalty_shots"])
    row["dribble_success_rate"] = _rate(row["successful_dribbles"], row["dribbles"])
    row["duel_win_rate"] = _rate(row["duels_won"], row["duels"])
    row["xg_difference"] = row["xg"] - row["opponent_xg"]
    row["non_penalty_xg_difference"] = row["non_penalty_xg"] - row["opponent_non_penalty_xg"]
    row["shot_difference"] = row["shots"] - row["opponent_shots"]
    row["pressure_difference"] = row["pressures"] - row["opponent_pressures"]
    row["final_third_entry_difference"] = row["final_third_entries"] - row["opponent_final_third_entries"]
    row["box_entry_difference"] = row["box_entries"] - row["opponent_box_entries"]


def parse_statsbomb_team_match_features(
    statsbomb_root: Path = DEFAULT_STATSBOMB_ROOT,
    event_files: list[Path] | None = None,
) -> pd.DataFrame:
    """Parse StatsBomb event files into one team-level row per match."""
    metadata = load_match_metadata(statsbomb_root)
    events_root = statsbomb_root / "events"
    if event_files is None:
        event_files = sorted(events_root.glob("*.json"))
    if not event_files:
        raise FileNotFoundError(f"No StatsBomb event files found under: {events_root}")

    rows: list[dict[str, Any]] = []
    missing_metadata: list[int] = []

    for event_path in sorted(event_files):
        match_id = int(event_path.stem)
        match = metadata.get(match_id)
        if match is None:
            missing_metadata.append(match_id)
            continue

        events = _load_json(event_path)
        teams = _match_team_names(match, events)
        team_stats = _aggregate_events(events, teams)

        for team in teams:
            opponent = teams[1] if teams[0] == team else teams[0]
            row = _team_match_base(match, team, opponent)
            row.update(team_stats[team])
            row.update({f"opponent_{key}": value for key, value in team_stats[opponent].items()})
            _derived_metrics(row)
            rows.append(row)

    if missing_metadata:
        raise ValueError(
            "StatsBomb event files are missing match metadata for match_id values: "
            + ", ".join(str(match_id) for match_id in sorted(missing_metadata))
        )

    features = pd.DataFrame(rows)
    if features.empty:
        raise ValueError("No StatsBomb team-match rows were parsed.")

    features["match_date"] = pd.to_datetime(features["match_date"], errors="coerce")
    features = features.sort_values(["match_date", "match_id", "team"]).reset_index(drop=True)
    return features


def save_statsbomb_team_match_features(
    features: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Save parsed StatsBomb team-match features."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert StatsBomb Open Data events into team-match features."
    )
    parser.add_argument(
        "--statsbomb-root",
        type=Path,
        default=DEFAULT_STATSBOMB_ROOT,
        help="Local StatsBomb Open Data root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV path for parsed team-match features.",
    )
    args = parser.parse_args(argv)

    features = parse_statsbomb_team_match_features(args.statsbomb_root)
    save_statsbomb_team_match_features(features, args.output)
    print(
        f"Wrote {len(features)} rows from {features['match_id'].nunique()} matches "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Rolling pre-match features from parsed StatsBomb team-match data."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_STATSBOMB_TEAM_MATCH_FEATURES_PATH = Path(
    "data/processed/statsbomb_team_match_features.csv"
)
DEFAULT_STATSBOMB_ROLLING_FEATURES_PATH = Path(
    "data/processed/statsbomb_team_rolling_features.csv"
)
STATSBOMB_WINDOWS = (3, 5)
ROLLING_METRIC_COLUMNS = {
    "xg_for": "xg",
    "xg_against": "opponent_xg",
    "non_penalty_xg_for": "non_penalty_xg",
    "non_penalty_xg_against": "opponent_non_penalty_xg",
    "shots_for": "shots",
    "shots_against": "opponent_shots",
    "box_entries_for": "box_entries",
    "box_entries_against": "opponent_box_entries",
    "pressures_for": "pressures",
    "xg_diff": "xg_difference",
    "non_penalty_xg_diff": "non_penalty_xg_difference",
    "shot_diff": "shot_difference",
    "box_entry_diff": "box_entry_difference",
    "pressure_diff": "pressure_difference",
}
TEAM_FEATURE_METRICS = ("xg_for", "xg_against", "xg_diff")
DIFFERENCE_FEATURE_METRICS = (
    "xg_diff",
    "non_penalty_xg_diff",
    "shot_diff",
    "box_entry_diff",
    "pressure_diff",
)


def load_statsbomb_team_match_features(
    path: Path = DEFAULT_STATSBOMB_TEAM_MATCH_FEATURES_PATH,
) -> pd.DataFrame:
    """Load parsed StatsBomb team-match features."""
    if not path.exists():
        raise FileNotFoundError(f"StatsBomb team-match feature file not found: {path}")
    features = pd.read_csv(path, parse_dates=["match_date"])
    required = {"match_date", "match_id", "team", "opponent"} | set(ROLLING_METRIC_COLUMNS.values())
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(
            f"StatsBomb team-match feature file is missing required columns: {', '.join(missing)}"
        )
    return features


def statsbomb_metric_defaults(team_match_features: pd.DataFrame) -> dict[str, float]:
    """Return neutral fallback values for StatsBomb rolling metrics."""
    defaults: dict[str, float] = {}
    for metric, source_column in ROLLING_METRIC_COLUMNS.items():
        if metric.endswith("_diff"):
            defaults[metric] = 0.0
        else:
            defaults[metric] = float(
                pd.to_numeric(team_match_features[source_column], errors="coerce").mean()
            )
    return defaults


def _history_entry(row: pd.Series) -> dict[str, float]:
    return {
        metric: float(pd.to_numeric(row[source_column], errors="coerce"))
        for metric, source_column in ROLLING_METRIC_COLUMNS.items()
    }


def _rolling_mean(
    history: Iterable[dict[str, float]],
    metric: str,
    window: int,
    defaults: dict[str, float],
) -> float:
    recent = list(history)[-window:]
    if not recent:
        return float(defaults.get(metric, 0.0))
    values = [float(row[metric]) for row in recent]
    return float(sum(values) / len(values))


def _team_rolling_features(
    history: deque[dict[str, float]],
    defaults: dict[str, float],
) -> dict[str, float]:
    rows = list(history)
    features: dict[str, float] = {
        "statsbomb_matches_before": float(len(rows)),
        "has_statsbomb_features": float(len(rows) > 0),
    }
    for window in STATSBOMB_WINDOWS:
        for metric in ROLLING_METRIC_COLUMNS:
            features[f"statsbomb_{metric}_last_{window}"] = _rolling_mean(
                rows,
                metric,
                window,
                defaults,
            )
    return features


def build_statsbomb_pair_features(
    team_a: str,
    team_b: str,
    histories: dict[str, deque[dict[str, float]]],
    defaults: dict[str, float],
) -> dict[str, float]:
    """Return selected pre-match StatsBomb priors for a two-team feature row."""
    team_a_features = _team_rolling_features(histories[team_a], defaults)
    team_b_features = _team_rolling_features(histories[team_b], defaults)
    features: dict[str, float] = {
        "team_a_statsbomb_matches_before": team_a_features["statsbomb_matches_before"],
        "team_b_statsbomb_matches_before": team_b_features["statsbomb_matches_before"],
        "team_a_has_statsbomb_features": team_a_features["has_statsbomb_features"],
        "team_b_has_statsbomb_features": team_b_features["has_statsbomb_features"],
        "both_teams_have_statsbomb_features": float(
            team_a_features["has_statsbomb_features"] > 0
            and team_b_features["has_statsbomb_features"] > 0
        ),
    }

    for window in STATSBOMB_WINDOWS:
        for metric in TEAM_FEATURE_METRICS:
            team_a_value = team_a_features[f"statsbomb_{metric}_last_{window}"]
            team_b_value = team_b_features[f"statsbomb_{metric}_last_{window}"]
            features[f"team_a_statsbomb_{metric}_last_{window}"] = team_a_value
            features[f"team_b_statsbomb_{metric}_last_{window}"] = team_b_value

        for metric in DIFFERENCE_FEATURE_METRICS:
            team_a_value = team_a_features[f"statsbomb_{metric}_last_{window}"]
            team_b_value = team_b_features[f"statsbomb_{metric}_last_{window}"]
            features[f"statsbomb_{metric}_delta_last_{window}"] = team_a_value - team_b_value

    return features


def update_statsbomb_histories_until(
    statsbomb_rows: pd.DataFrame,
    histories: dict[str, deque[dict[str, float]]],
    current_index: int,
    current_date: pd.Timestamp,
) -> int:
    """Append StatsBomb rows dated strictly before ``current_date`` to histories."""
    while current_index < len(statsbomb_rows):
        row = statsbomb_rows.iloc[current_index]
        row_date = pd.Timestamp(row["match_date"])
        if row_date >= current_date:
            break
        histories[str(row["team"])].append(_history_entry(row))
        current_index += 1
    return current_index


def build_statsbomb_rolling_feature_dataset(
    team_match_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build one pre-match rolling StatsBomb row per team per StatsBomb match."""
    working = team_match_features.copy()
    working["match_date"] = pd.to_datetime(working["match_date"], errors="coerce")
    working = working.dropna(subset=["match_date", "team"]).sort_values(
        ["match_date", "match_id", "team"],
    )
    defaults = statsbomb_metric_defaults(working)
    histories: dict[str, deque[dict[str, float]]] = defaultdict(deque)
    rows: list[dict[str, Any]] = []

    for match_date, date_rows in working.groupby("match_date", sort=True):
        for _, row in date_rows.iterrows():
            team = str(row["team"])
            features = _team_rolling_features(histories[team], defaults)
            rows.append(
                {
                    "match_id": int(row["match_id"]),
                    "match_date": pd.Timestamp(match_date),
                    "team": team,
                    "opponent": str(row["opponent"]),
                    **features,
                }
            )

        for _, row in date_rows.iterrows():
            histories[str(row["team"])].append(_history_entry(row))

    return pd.DataFrame(rows).sort_values(["match_date", "match_id", "team"]).reset_index(drop=True)


def save_statsbomb_rolling_feature_dataset(
    features: pd.DataFrame,
    output_path: Path = DEFAULT_STATSBOMB_ROLLING_FEATURES_PATH,
) -> None:
    """Save rolling StatsBomb pre-match features."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)

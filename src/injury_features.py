"""Team-level injury feature aggregation for World Cup models."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.market_value import canonical_team_name


INJURED_PLAYERS_MARKET_VALUE_PATH = Path("data/fixtures/injured_players_market_value_filled.csv")
POSITION_GROUPS = ("gk", "df", "mf", "fw")
KEY_PLAYER_VALUE_EUR = 20_000_000.0
KEY_PLAYER_TEAM_VALUE_SHARE = 0.05


def _neutral_team_injury_features(prefix: str) -> dict[str, float]:
    features = {
        f"{prefix}_injured_players_count": 0.0,
        f"{prefix}_injured_market_value_eur": 0.0,
        f"{prefix}_injured_market_value_share": 0.0,
        f"{prefix}_key_injured_players_count": 0.0,
        f"{prefix}_max_injured_player_market_value_eur": 0.0,
    }
    for position in POSITION_GROUPS:
        features[f"{prefix}_injured_{position}_count"] = 0.0
        features[f"{prefix}_injured_{position}_market_value_eur"] = 0.0
    return features


def neutral_pair_injury_features() -> dict[str, float]:
    """Return neutral pair-level injury features."""
    features = {
        "has_injury_data": 0.0,
        **_neutral_team_injury_features("team_a"),
        **_neutral_team_injury_features("team_b"),
    }
    features.update(_injury_difference_features(features))
    return features


def _position_key(value: object) -> str | None:
    text = str(value).strip().lower()
    if text in POSITION_GROUPS:
        return text
    aliases = {
        "goalkeeper": "gk",
        "defender": "df",
        "midfielder": "mf",
        "forward": "fw",
        "attacker": "fw",
        "striker": "fw",
    }
    return aliases.get(text)


def _safe_numeric(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)


def load_injury_feature_index(
    path: str | Path = INJURED_PLAYERS_MARKET_VALUE_PATH,
    market_values_by_year: dict[int, dict[str, float]] | None = None,
) -> dict[tuple[int, str], dict[str, float]]:
    """Load injured player market values aggregated by World Cup year and team."""
    injury_path = Path(path)
    if not injury_path.exists():
        return {}

    data = pd.read_csv(injury_path)
    if data.empty:
        return {}

    if "tournament_year" not in data.columns and "tournament" in data.columns:
        data["tournament_year"] = pd.to_numeric(
            data["tournament"].astype(str).str.extract(r"(\d{4})", expand=False),
            errors="coerce",
        )
    required = {"tournament_year", "national_team", "player_name", "market_value_eur"}
    missing = required - set(data.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"injury market value file missing required columns: {missing_text}")

    data["tournament_year"] = pd.to_numeric(data["tournament_year"], errors="coerce")
    data["team"] = data["national_team"].map(canonical_team_name)
    data["market_value_eur"] = pd.to_numeric(data["market_value_eur"], errors="coerce").fillna(0.0)
    data = data.dropna(subset=["tournament_year", "team", "player_name"])

    aggregates: dict[tuple[int, str], dict[str, float]] = defaultdict(_empty_team_aggregate)
    for _, row in data.iterrows():
        year = int(row["tournament_year"])
        team = str(row["team"])
        key = (year, team)
        value = max(float(row["market_value_eur"]), 0.0)
        aggregate = aggregates[key]
        aggregate["injured_players_count"] += 1.0
        aggregate["injured_market_value_eur"] += value
        aggregate["max_injured_player_market_value_eur"] = max(
            aggregate["max_injured_player_market_value_eur"],
            value,
        )
        position = _position_key(row.get("player_position"))
        if position:
            aggregate[f"injured_{position}_count"] += 1.0
            aggregate[f"injured_{position}_market_value_eur"] += value

    market_values_by_year = market_values_by_year or {}
    for (year, team), aggregate in aggregates.items():
        team_market_value = float(market_values_by_year.get(year, {}).get(team, 0.0) or 0.0)
        injured_value = aggregate["injured_market_value_eur"]
        injured_share = injured_value / team_market_value if team_market_value > 0 else 0.0
        aggregate["injured_market_value_share"] = injured_share

        team_rows = data[
            data["tournament_year"].astype(int).eq(year)
            & data["team"].eq(team)
        ]
        key_count = 0.0
        for _, row in team_rows.iterrows():
            value = max(_safe_numeric(row["market_value_eur"]), 0.0)
            player_share = value / team_market_value if team_market_value > 0 else 0.0
            if value >= KEY_PLAYER_VALUE_EUR or player_share >= KEY_PLAYER_TEAM_VALUE_SHARE:
                key_count += 1.0
        aggregate["key_injured_players_count"] = key_count

    return dict(aggregates)


def _empty_team_aggregate() -> dict[str, float]:
    aggregate = {
        "injured_players_count": 0.0,
        "injured_market_value_eur": 0.0,
        "injured_market_value_share": 0.0,
        "key_injured_players_count": 0.0,
        "max_injured_player_market_value_eur": 0.0,
    }
    for position in POSITION_GROUPS:
        aggregate[f"injured_{position}_count"] = 0.0
        aggregate[f"injured_{position}_market_value_eur"] = 0.0
    return aggregate


def build_pair_injury_features(
    team_a: str,
    team_b: str,
    year: int,
    injury_feature_index: dict[tuple[int, str], dict[str, float]] | None,
) -> dict[str, float]:
    """Return pair-level team injury features for one match."""
    if not injury_feature_index:
        return neutral_pair_injury_features()

    team_a_key = (year, canonical_team_name(team_a))
    team_b_key = (year, canonical_team_name(team_b))
    team_a_features = injury_feature_index.get(team_a_key, _empty_team_aggregate())
    team_b_features = injury_feature_index.get(team_b_key, _empty_team_aggregate())
    features = {"has_injury_data": float(team_a_key in injury_feature_index or team_b_key in injury_feature_index)}
    for name, value in team_a_features.items():
        features[f"team_a_{name}"] = float(value)
    for name, value in team_b_features.items():
        features[f"team_b_{name}"] = float(value)
    features.update(_injury_difference_features(features))
    return features


def _injury_difference_features(features: dict[str, float]) -> dict[str, float]:
    difference_features = {
        "injured_players_count_diff": (
            features["team_a_injured_players_count"] - features["team_b_injured_players_count"]
        ),
        "injured_market_value_diff": (
            features["team_a_injured_market_value_eur"] - features["team_b_injured_market_value_eur"]
        ),
        "injured_market_value_share_diff": (
            features["team_a_injured_market_value_share"] - features["team_b_injured_market_value_share"]
        ),
        "key_injured_players_count_diff": (
            features["team_a_key_injured_players_count"] - features["team_b_key_injured_players_count"]
        ),
        "max_injured_player_market_value_diff": (
            features["team_a_max_injured_player_market_value_eur"]
            - features["team_b_max_injured_player_market_value_eur"]
        ),
    }
    for position in POSITION_GROUPS:
        difference_features[f"injured_{position}_count_diff"] = (
            features[f"team_a_injured_{position}_count"]
            - features[f"team_b_injured_{position}_count"]
        )
        difference_features[f"injured_{position}_market_value_diff"] = (
            features[f"team_a_injured_{position}_market_value_eur"]
            - features[f"team_b_injured_{position}_market_value_eur"]
        )
    return difference_features

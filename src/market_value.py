"""Optional squad market-value adjustments for tournament simulations."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

from src.config import (
    INITIAL_ELO,
    MARKET_VALUE_ELO_SCALE,
    MAX_MARKET_VALUE_ELO_ADJUSTMENT,
    MIN_MARKET_VALUE_EUR,
    USE_MARKET_VALUE_ADJUSTMENT,
)


TEAM_NAME_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Czechia": "Czech Republic",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Curacao": "Curaçao",
    "Democratic Republic of the Congo": "DR Congo",
    "Serbia and Montenegro": "Serbia",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "USA": "United States",
}


def canonical_team_name(team: object) -> str:
    """Return a project-standard team name for market-value matching."""
    name = str(team).strip()
    return TEAM_NAME_ALIASES.get(name, name)


def parse_market_value(value: object) -> float | None:
    """Parse numeric or human-readable market values into euros.

    Accepted examples: 1200000000, "1,200,000,000", "€1.2bn", "900m".
    """
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        amount = float(value)
        return amount if amount >= MIN_MARKET_VALUE_EUR else None

    text = str(value).strip().lower()
    if not text:
        return None

    multiplier = 1.0
    if any(suffix in text for suffix in ["bn", "billion", "b"]):
        multiplier = 1_000_000_000.0
    elif any(suffix in text for suffix in ["m", "million"]):
        multiplier = 1_000_000.0
    elif any(suffix in text for suffix in ["k", "thousand"]):
        multiplier = 1_000.0

    cleaned = re.sub(r"[^0-9.]", "", text)
    if not cleaned:
        return None
    try:
        amount = float(cleaned) * multiplier
    except ValueError:
        return None
    return amount if amount >= MIN_MARKET_VALUE_EUR else None


def load_market_values(path: str | Path) -> dict[str, float]:
    """Load team market values from a CSV with columns team, market_value_eur."""
    market_path = Path(path)
    if not market_path.exists():
        return {}

    df = pd.read_csv(market_path)
    if df.empty:
        return {}
    if "team" not in df.columns or "market_value_eur" not in df.columns:
        raise ValueError("market value file must contain columns: team, market_value_eur")

    market_values: dict[str, float] = {}
    for _, row in df.iterrows():
        team = canonical_team_name(row["team"])
        value = parse_market_value(row["market_value_eur"])
        if team and value is not None:
            market_values[team] = value
    return market_values


def load_historical_world_cup_market_values(path: str | Path) -> dict[int, dict[str, float]]:
    """Load World Cup market values grouped by tournament year.

    The CSV must contain ``tournament_year``, ``team``, and
    ``market_value_eur`` columns. A ``year`` column is also accepted as an
    alias for ``tournament_year``.
    """
    market_path = Path(path)
    if not market_path.exists():
        return {}

    df = pd.read_csv(market_path)
    if df.empty:
        return {}

    if "tournament_year" not in df.columns and "year" in df.columns:
        df = df.rename(columns={"year": "tournament_year"})

    required_columns = {"tournament_year", "team", "market_value_eur"}
    missing = required_columns - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"historical market value file missing required columns: {missing_text}")

    values_by_year: dict[int, dict[str, float]] = {}
    for _, row in df.iterrows():
        year = pd.to_numeric(row["tournament_year"], errors="coerce")
        team = canonical_team_name(row["team"])
        value = parse_market_value(row["market_value_eur"])
        if pd.isna(year) or not team or value is None:
            continue
        values_by_year.setdefault(int(year), {})[team] = value

    return values_by_year


def _median(values: list[float]) -> float:
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def median_market_value(values: list[float]) -> float:
    """Return the median market value from a non-empty list."""
    if not values:
        raise ValueError("cannot compute median market value from an empty list")
    return _median(values)


def market_value_adjustment(value: float, baseline_value: float) -> float:
    """Convert market value into a capped Elo-point adjustment."""
    if not USE_MARKET_VALUE_ADJUSTMENT or value <= 0 or baseline_value <= 0:
        return 0.0
    raw_adjustment = math.log(value / baseline_value) * MARKET_VALUE_ELO_SCALE
    return max(
        -MAX_MARKET_VALUE_ELO_ADJUSTMENT,
        min(MAX_MARKET_VALUE_ELO_ADJUSTMENT, raw_adjustment),
    )


def apply_market_value_adjustments(
    ratings: dict[str, float],
    market_values: dict[str, float],
    teams: set[str] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Apply optional market-value adjustments to a ratings dictionary."""
    if not USE_MARKET_VALUE_ADJUSTMENT or not market_values:
        eligible_teams = teams if teams is not None else set(ratings)
        adjusted_ratings = dict(ratings)
        rows = []
        for team in sorted(eligible_teams):
            elo = float(ratings.get(team, INITIAL_ELO))
            adjusted_ratings[team] = elo
            rows.append(
                {
                    "team": team,
                    "elo": elo,
                    "market_value_eur": None,
                    "market_value_baseline_eur": None,
                    "market_value_adjustment": 0.0,
                    "model_rating": elo,
                }
            )
        table = pd.DataFrame(rows).sort_values("model_rating", ascending=False, na_position="last")
        return adjusted_ratings, table

    eligible_teams = teams if teams is not None else set(ratings)
    eligible_values = [
        market_values[team]
        for team in eligible_teams
        if team in market_values and market_values[team] >= MIN_MARKET_VALUE_EUR
    ]
    if not eligible_values:
        return apply_market_value_adjustments(ratings, {}, teams=teams)

    baseline_value = _median(eligible_values)
    adjusted_ratings = dict(ratings)
    rows = []

    for team in sorted(eligible_teams):
        elo = float(ratings.get(team, INITIAL_ELO))
        value = market_values.get(team)
        adjustment = market_value_adjustment(value, baseline_value) if value is not None else 0.0
        model_rating = elo + adjustment
        adjusted_ratings[team] = model_rating
        rows.append(
            {
                "team": team,
                "elo": elo,
                "market_value_eur": value,
                "market_value_baseline_eur": baseline_value,
                "market_value_adjustment": adjustment,
                "model_rating": model_rating,
            }
        )

    table = pd.DataFrame(rows).sort_values("model_rating", ascending=False, na_position="last")
    return adjusted_ratings, table

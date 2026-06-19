"""Data loading and cleaning for historical international results."""

from __future__ import annotations

import pandas as pd

from src.config import MIN_MATCH_YEAR


def load_results(path: str) -> pd.DataFrame:
    """Load raw match results from a CSV file."""
    return pd.read_csv(path)


def _parse_neutral(value: object) -> bool:
    """Convert common neutral-site values to a boolean."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "t", "1", "yes", "y"}


def clean_results(df: pd.DataFrame, min_match_year: int | None = MIN_MATCH_YEAR) -> pd.DataFrame:
    """Clean raw results and add columns needed by the Elo model.

    The function handles common CSV issues: string scores, missing neutral
    flags, and mixed-case boolean neutral values.
    """
    required_columns = ["date", "home_team", "away_team", "home_score", "away_score"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"results data missing required columns: {', '.join(missing)}")

    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["home_score"] = pd.to_numeric(cleaned["home_score"], errors="coerce")
    cleaned["away_score"] = pd.to_numeric(cleaned["away_score"], errors="coerce")

    cleaned = cleaned.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    cleaned["home_score"] = cleaned["home_score"].astype(int)
    cleaned["away_score"] = cleaned["away_score"].astype(int)

    if "neutral" not in cleaned.columns:
        cleaned["neutral"] = False
    cleaned["neutral"] = cleaned["neutral"].apply(_parse_neutral).astype(bool)

    if "tournament" not in cleaned.columns:
        cleaned["tournament"] = ""

    if min_match_year is not None:
        cleaned = cleaned[cleaned["date"].dt.year >= min_match_year].copy()
    cleaned["result"] = 0.5
    cleaned.loc[cleaned["home_score"] > cleaned["away_score"], "result"] = 1.0
    cleaned.loc[cleaned["home_score"] < cleaned["away_score"], "result"] = 0.0
    cleaned["goal_diff"] = cleaned["home_score"] - cleaned["away_score"]

    return cleaned.sort_values("date").reset_index(drop=True)

"""Validation helpers for historical World Cup injury data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.market_value import canonical_team_name


EXPECTED_WORLD_CUP_YEARS = {2014, 2018, 2022}
DEFAULT_INJURY_DATA_PATH = Path("data/fixtures/team_injuries.csv")
HISTORICAL_BACKFILL_INJURY_DATA_PATH = Path(
    "data/fixtures/team_injuries_historical_backfill_2014_2018_2022.csv"
)
INJURY_DATA_PATH = (
    DEFAULT_INJURY_DATA_PATH
    if DEFAULT_INJURY_DATA_PATH.exists()
    else HISTORICAL_BACKFILL_INJURY_DATA_PATH
)
RAW_RESULTS_PATH = Path("data/raw/results.csv")
REPORT_PATH = Path("results/injury_data_validation_report.json")

MINIMUM_COLUMNS = {"team", "player_name"}
PREFERRED_COLUMNS = {
    "tournament_year",
    "match_date",
    "team",
    "player_name",
    "position",
    "absence_reason",
    "injury_start_date",
    "expected_return_date",
    "ruled_out",
    "in_preliminary_squad",
    "in_final_squad",
    "player_market_value_eur",
    "source_name",
    "source_url",
    "source_published_at",
}
DATE_COLUMNS = ["match_date", "injury_start_date", "expected_return_date", "source_published_at"]
BOOLEAN_COLUMNS = ["ruled_out", "in_preliminary_squad", "in_final_squad"]
NUMERIC_COLUMNS = ["player_market_value_eur"]

COLUMN_ALIASES = {
    "national_team": "team",
    "player_position": "position",
    "ruled_out_match": "ruled_out",
    "published_at": "source_published_at",
}


def _load_world_cup_teams(path: Path = RAW_RESULTS_PATH) -> dict[int, set[str]]:
    if not path.exists():
        return {}

    results = pd.read_csv(path, parse_dates=["date"])
    world_cups = results[
        results["tournament"].eq("FIFA World Cup")
        & results["date"].dt.year.isin(EXPECTED_WORLD_CUP_YEARS)
    ].copy()

    teams_by_year: dict[int, set[str]] = {}
    for year, group in world_cups.groupby(world_cups["date"].dt.year):
        teams = set(group["home_team"].map(canonical_team_name))
        teams.update(group["away_team"].map(canonical_team_name))
        teams_by_year[int(year)] = teams
    return teams_by_year


def _parse_boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(
        {
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "1": True,
            "0": False,
            "y": True,
            "n": False,
        }
    )


def _normalize_schema(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]
    for source_column, target_column in COLUMN_ALIASES.items():
        if source_column in data.columns and target_column not in data.columns:
            data[target_column] = data[source_column]
    return data


def _parse_partial_date(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    empty = series.isna() | text.eq("")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    full_mask = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    month_mask = text.str.fullmatch(r"\d{4}-\d{2}", na=False)
    year_mask = text.str.fullmatch(r"\d{4}", na=False)

    parsed.loc[full_mask] = pd.to_datetime(text.loc[full_mask], errors="coerce")
    parsed.loc[month_mask] = pd.to_datetime(text.loc[month_mask] + "-01", errors="coerce")
    parsed.loc[year_mask] = pd.to_datetime(text.loc[year_mask] + "-01-01", errors="coerce")

    remaining_mask = ~(empty | full_mask | month_mask | year_mask)
    parsed.loc[remaining_mask] = pd.to_datetime(text.loc[remaining_mask], errors="coerce")
    return parsed


def _derive_tournament_year(data: pd.DataFrame) -> pd.Series:
    if "tournament_year" in data.columns:
        return pd.to_numeric(data["tournament_year"], errors="coerce")
    if "match_date" in data.columns:
        match_years = _parse_partial_date(data["match_date"]).dt.year
        if match_years.notna().any():
            return match_years
    if "tournament" in data.columns:
        return pd.to_numeric(
            data["tournament"].astype(str).str.extract(r"(\d{4})", expand=False),
            errors="coerce",
        )
    return pd.Series([pd.NA] * len(data), index=data.index, dtype="Int64")


def validate_injury_data(
    injury_path: Path = INJURY_DATA_PATH,
    raw_results_path: Path = RAW_RESULTS_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(injury_path),
        "exists": injury_path.exists(),
        "usable": False,
        "errors": [],
        "warnings": [],
        "summary": {},
    }

    if not injury_path.exists():
        report["errors"].append("Missing injury data file.")
        _write_report(report, report_path)
        return report

    data = _normalize_schema(pd.read_csv(injury_path))
    missing_minimum = sorted(MINIMUM_COLUMNS - set(data.columns))
    missing_preferred = sorted(PREFERRED_COLUMNS - set(data.columns))

    if missing_minimum:
        report["errors"].append(f"Missing minimum columns: {', '.join(missing_minimum)}")
    if "tournament_year" not in data.columns and "match_date" not in data.columns:
        report["errors"].append("Need either tournament_year or match_date.")
    if missing_preferred:
        report["warnings"].append(f"Missing preferred columns: {', '.join(missing_preferred)}")

    data["tournament_year"] = _derive_tournament_year(data)
    invalid_years = data["tournament_year"].isna()
    if invalid_years.any():
        report["errors"].append(f"Rows with invalid tournament_year: {int(invalid_years.sum())}")

    for column in DATE_COLUMNS:
        if column not in data.columns:
            continue
        parsed = _parse_partial_date(data[column])
        invalid = data[column].notna() & parsed.isna()
        if invalid.any():
            report["errors"].append(f"Rows with invalid {column}: {int(invalid.sum())}")

    for column in BOOLEAN_COLUMNS:
        if column not in data.columns:
            continue
        parsed = _parse_boolean(data[column])
        invalid = data[column].notna() & parsed.isna()
        if invalid.any():
            report["warnings"].append(f"Rows with non-standard boolean {column}: {int(invalid.sum())}")

    for column in NUMERIC_COLUMNS:
        if column not in data.columns:
            continue
        parsed = pd.to_numeric(data[column], errors="coerce")
        invalid = data[column].notna() & parsed.isna()
        negative = parsed.lt(0).fillna(False)
        if invalid.any():
            report["errors"].append(f"Rows with invalid numeric {column}: {int(invalid.sum())}")
        if negative.any():
            report["errors"].append(f"Rows with negative {column}: {int(negative.sum())}")

    blank_team = data.get("team", pd.Series(dtype=object)).astype(str).str.strip().eq("")
    blank_player = data.get("player_name", pd.Series(dtype=object)).astype(str).str.strip().eq("")
    if blank_team.any():
        report["errors"].append(f"Rows with blank team: {int(blank_team.sum())}")
    if blank_player.any():
        report["errors"].append(f"Rows with blank player_name: {int(blank_player.sum())}")

    key_columns = [column for column in ["tournament_year", "match_date", "team", "player_name"] if column in data.columns]
    if key_columns:
        duplicate_keys = data.duplicated(key_columns, keep=False)
        if duplicate_keys.any():
            report["warnings"].append(
                f"Rows with duplicate injury keys ({', '.join(key_columns)}): {int(duplicate_keys.sum())}"
            )

    exact_duplicates = data.duplicated(keep=False)
    if exact_duplicates.any():
        report["warnings"].append(f"Exact duplicate rows: {int(exact_duplicates.sum())}")

    observed_years = set(pd.to_numeric(data["tournament_year"], errors="coerce").dropna().astype(int))
    missing_years = sorted(EXPECTED_WORLD_CUP_YEARS - observed_years)
    unexpected_years = sorted(observed_years - EXPECTED_WORLD_CUP_YEARS)
    if missing_years:
        report["errors"].append(f"Missing expected World Cup years: {', '.join(map(str, missing_years))}")
    if unexpected_years:
        report["warnings"].append(f"Unexpected tournament years present: {', '.join(map(str, unexpected_years))}")

    if "team" in data.columns:
        data["canonical_team"] = data["team"].map(canonical_team_name)
        teams_by_year = _load_world_cup_teams(raw_results_path)
        unknown_rows = []
        for index, row in data.iterrows():
            year = row.get("tournament_year")
            if pd.isna(year):
                continue
            valid_teams = teams_by_year.get(int(year), set())
            if valid_teams and row["canonical_team"] not in valid_teams:
                unknown_rows.append(index)
        if unknown_rows:
            report["warnings"].append(
                f"Rows with teams not found in matching World Cup field: {len(unknown_rows)}"
            )

    report["summary"] = {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "observed_years": sorted(int(year) for year in observed_years),
        "records_by_year": {
            str(int(year)): int(count)
            for year, count in data.groupby("tournament_year", dropna=False).size().items()
            if not pd.isna(year)
        },
        "teams_by_year": {
            str(int(year)): int(count)
            for year, count in data.groupby("tournament_year")["team"].nunique().items()
            if not pd.isna(year)
        }
        if "team" in data.columns
        else {},
        "preferred_columns_present": sorted(PREFERRED_COLUMNS & set(data.columns)),
        "preferred_columns_missing": missing_preferred,
    }
    report["usable"] = not report["errors"]
    _write_report(report, report_path)
    return report


def _write_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

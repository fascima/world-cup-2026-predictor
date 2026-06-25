"""Live World Cup fixture/result ingestion helpers."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
DEFAULT_COMPETITION = "WC"
DEFAULT_SEASON = 2026
LIVE_DATA_DIR = Path("data/live")
LIVE_MATCHES_PATH = LIVE_DATA_DIR / "world_cup_matches.csv"
LIVE_RESULTS_PATH = LIVE_DATA_DIR / "results_with_live_world_cup.csv"

TEAM_NAME_ALIASES = {
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Curaçao": "Curaçao",
    "Czechia": "Czech Republic",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "USA": "United States",
}

FINISHED_STATUSES = {"FINISHED", "AWARDED"}


def normalize_team_name(name: object) -> str:
    """Return the repo's preferred team name for an API team value."""
    text = "" if pd.isna(name) else str(name).strip()
    return TEAM_NAME_ALIASES.get(text, text)


def local_date_from_utc(value: object) -> str:
    """Return the New York local date for a football-data UTC datetime."""
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return timestamp.tz_convert("America/New_York").date().isoformat()


def _team_name(team: dict[str, Any] | None) -> str:
    if not team:
        return ""
    return normalize_team_name(team.get("name") or team.get("shortName") or team.get("tla") or "")


def _score_value(score: dict[str, Any], side: str) -> int | None:
    value = score.get("fullTime", {}).get(side)
    if value is None:
        value = score.get("regularTime", {}).get(side)
    return None if value is None else int(value)


def normalize_match_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Normalize a football-data matches response into the app's live schema."""
    rows = []
    fetched_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    for match in payload.get("matches", []):
        score = match.get("score") or {}
        home_score = _score_value(score, "home")
        away_score = _score_value(score, "away")
        rows.append(
            {
                "match_id": str(match.get("id", "")),
                "utc_date": match.get("utcDate", ""),
                "local_date": local_date_from_utc(match.get("utcDate", "")),
                "status": match.get("status", ""),
                "stage": match.get("stage", ""),
                "group": match.get("group", ""),
                "matchday": match.get("matchday", ""),
                "home_team": _team_name(match.get("homeTeam")),
                "away_team": _team_name(match.get("awayTeam")),
                "home_score": home_score,
                "away_score": away_score,
                "winner": score.get("winner", ""),
                "last_updated": match.get("lastUpdated", ""),
                "fetched_at": fetched_at,
                "source": "football-data.org",
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["utc_date", "match_id"], kind="mergesort").reset_index(drop=True)
    return df


def fetch_world_cup_matches(
    date_from: date | str | None = None,
    date_to: date | str | None = None,
    api_key: str | None = None,
    competition: str | None = None,
    season: int | None = None,
) -> pd.DataFrame:
    """Fetch World Cup matches from football-data.org."""
    token = api_key or os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not token:
        raise RuntimeError("FOOTBALL_DATA_API_KEY is not set.")

    params: dict[str, str | int] = {"season": season or DEFAULT_SEASON}
    if date_from is not None:
        params["dateFrom"] = str(date_from)
    if date_to is not None:
        params["dateTo"] = str(date_to)
    query = urlencode(params)
    competition_id = competition or os.environ.get("FOOTBALL_DATA_COMPETITION", DEFAULT_COMPETITION)
    url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{competition_id}/matches?{query}"
    request = Request(url, headers={"X-Auth-Token": token, "Accept": "application/json"})

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"football-data.org request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"football-data.org request failed: {exc.reason}") from exc
    return normalize_match_payload(payload)


def load_cached_matches(path: Path = LIVE_MATCHES_PATH) -> pd.DataFrame:
    """Load cached live matches if present."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save_cached_matches(matches: pd.DataFrame, path: Path = LIVE_MATCHES_PATH) -> None:
    """Write live matches to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(path, index=False)


def merge_match_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge cached and freshly fetched matches, preferring the newest row by match id."""
    frames = [frame for frame in [existing, incoming] if not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if "match_id" in merged.columns:
        merged["match_id"] = merged["match_id"].astype(str)
        merged = merged.drop_duplicates(subset=["match_id"], keep="last")
    return merged.sort_values(["utc_date", "match_id"], kind="mergesort").reset_index(drop=True)


def refresh_cached_matches(
    days_back: int = 7,
    days_forward: int = 7,
    today: date | None = None,
) -> pd.DataFrame:
    """Fetch a rolling World Cup window and merge it into the local cache."""
    anchor = today or date.today()
    incoming = fetch_world_cup_matches(
        date_from=anchor - timedelta(days=days_back),
        date_to=anchor + timedelta(days=days_forward),
    )
    merged = merge_match_frames(load_cached_matches(), incoming)
    save_cached_matches(merged)
    return merged


def todays_matches(matches: pd.DataFrame, today: date | str | None = None) -> pd.DataFrame:
    """Return matches scheduled for the requested local date."""
    if matches.empty or "local_date" not in matches.columns:
        return pd.DataFrame()
    target = str(today or date.today())
    return matches[matches["local_date"].astype(str).eq(target)].copy()


def completed_matches_to_results(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert finished live matches into the historical results.csv shape."""
    if matches.empty:
        return pd.DataFrame()
    finished = matches[matches["status"].astype(str).isin(FINISHED_STATUSES)].copy()
    finished = finished.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    if finished.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "date": pd.to_datetime(finished["utc_date"], utc=True, errors="coerce").dt.date.astype(str),
            "home_team": finished["home_team"].map(normalize_team_name),
            "away_team": finished["away_team"].map(normalize_team_name),
            "home_score": pd.to_numeric(finished["home_score"], errors="coerce").astype("Int64"),
            "away_score": pd.to_numeric(finished["away_score"], errors="coerce").astype("Int64"),
            "tournament": "FIFA World Cup",
            "city": "",
            "country": "",
            "neutral": True,
        }
    ).dropna(subset=["date", "home_score", "away_score"])


def write_results_with_live_matches(
    raw_results_path: Path = Path("data/raw/results.csv"),
    output_path: Path = LIVE_RESULTS_PATH,
    matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Write historical results plus completed live World Cup matches."""
    if not raw_results_path.exists():
        raise FileNotFoundError(raw_results_path)
    raw = pd.read_csv(raw_results_path)
    live_results = completed_matches_to_results(matches if matches is not None else load_cached_matches())
    if live_results.empty:
        combined = raw.copy()
    else:
        combined = pd.concat([raw, live_results], ignore_index=True)
        key_columns = ["date", "home_team", "away_team", "tournament"]
        combined = combined.drop_duplicates(subset=key_columns, keep="last")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined

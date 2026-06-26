"""End-to-end live World Cup refresh workflow."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from main import GROUPS_PATH, MARKET_VALUES_PATH, MODEL_RATINGS_PATH
from src.data_loader import clean_results, load_results
from src.elo import build_elo_history, save_elo_ratings
from src.live_world_cup import (
    LIVE_RESULTS_PATH,
    current_display_date,
    refresh_cached_matches,
    write_results_with_live_matches,
)
from src.market_value import apply_market_value_adjustments, load_market_values
from src.simulate import load_groups
from src.todays_predictions import build_todays_predictions


ELO_RATINGS_PATH = Path("results/current_elo_ratings.csv")
WORLD_CUP_FINAL_DATE = date(2026, 7, 19)


def world_cup_updates_are_active(today: date | None = None) -> bool:
    """Return True while automated World Cup refreshes should run."""
    return (today or current_display_date()) <= WORLD_CUP_FINAL_DATE


def _world_cup_teams() -> set[str]:
    if not GROUPS_PATH.exists():
        return set()
    groups = load_groups(str(GROUPS_PATH))
    return {team for teams in groups.values() for team in teams}


def rebuild_live_elo_state(results_path: Path = LIVE_RESULTS_PATH) -> pd.DataFrame:
    """Rebuild current Elo/model ratings from historical plus completed live matches."""
    matches = clean_results(load_results(str(results_path)))
    _, ratings = build_elo_history(matches)
    save_elo_ratings(ratings, str(ELO_RATINGS_PATH))

    ratings_df = pd.read_csv(ELO_RATINGS_PATH)
    world_cup_teams = _world_cup_teams()
    market_values = load_market_values(MARKET_VALUES_PATH)
    _, model_ratings = apply_market_value_adjustments(
        ratings,
        market_values,
        teams=world_cup_teams or None,
    )
    MODEL_RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_ratings.to_csv(MODEL_RATINGS_PATH, index=False)
    return ratings_df


def refresh_live_outputs(
    days_back: int = 7,
    days_forward: int = 7,
    today: date | None = None,
) -> dict[str, int | str]:
    """Refresh live fixtures/results, ratings, and today's prediction file."""
    anchor = today or current_display_date()
    if not world_cup_updates_are_active(anchor):
        return {
            "status": "inactive",
            "message": f"Live World Cup updates stopped after {WORLD_CUP_FINAL_DATE.isoformat()}.",
        }

    matches = refresh_cached_matches(
        days_back=days_back,
        days_forward=days_forward,
        today=anchor,
    )
    combined = write_results_with_live_matches(matches=matches)
    ratings = rebuild_live_elo_state()
    predictions = build_todays_predictions(matches=matches, today=anchor)
    return {
        "status": "updated",
        "matches": len(matches),
        "results": len(combined),
        "ratings": len(ratings),
        "predictions": len(predictions),
    }

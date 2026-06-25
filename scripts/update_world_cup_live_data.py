"""Refresh live World Cup data and today's match predictions."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import GROUPS_PATH, MARKET_VALUES_PATH, MODEL_RATINGS_PATH
from src.data_loader import clean_results, load_results
from src.elo import build_elo_history, save_elo_ratings
from src.live_world_cup import (
    LIVE_RESULTS_PATH,
    refresh_cached_matches,
    write_results_with_live_matches,
)
from src.market_value import apply_market_value_adjustments, load_market_values
from src.simulate import load_groups
from src.todays_predictions import build_todays_predictions


ELO_RATINGS_PATH = Path("results/current_elo_ratings.csv")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days-back", type=int, default=7, help="Days before today to refresh.")
    parser.add_argument("--days-forward", type=int, default=7, help="Days after today to refresh.")
    parser.add_argument("--today", default=None, help="Override today's local date as YYYY-MM-DD.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    today = date.fromisoformat(args.today) if args.today else date.today()
    matches = refresh_cached_matches(
        days_back=args.days_back,
        days_forward=args.days_forward,
        today=today,
    )
    combined = write_results_with_live_matches(matches=matches)
    ratings = rebuild_live_elo_state()
    predictions = build_todays_predictions(matches=matches, today=today)

    print(f"Cached {len(matches)} World Cup matches.")
    print(f"Wrote {len(combined)} historical/live result rows to {LIVE_RESULTS_PATH}.")
    print(f"Updated {len(ratings)} current Elo ratings.")
    print(f"Wrote {len(predictions)} today's prediction rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

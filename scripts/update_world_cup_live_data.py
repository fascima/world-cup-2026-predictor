"""Refresh live World Cup data and today's match predictions."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_update import LIVE_RESULTS_PATH, refresh_live_outputs
from src.live_world_cup import current_display_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days-back", type=int, default=7, help="Days before today to refresh.")
    parser.add_argument("--days-forward", type=int, default=7, help="Days after today to refresh.")
    parser.add_argument("--today", default=None, help="Override today's local date as YYYY-MM-DD.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    today = date.fromisoformat(args.today) if args.today else current_display_date()
    summary = refresh_live_outputs(
        days_back=args.days_back,
        days_forward=args.days_forward,
        today=today,
    )
    if summary.get("status") == "inactive":
        print(summary["message"])
        return 0

    print(f"Cached {summary['matches']} World Cup matches.")
    print(f"Wrote {summary['results']} historical/live result rows to {LIVE_RESULTS_PATH}.")
    print(f"Updated {summary['ratings']} current Elo ratings.")
    print(f"Wrote {summary['predictions']} today's prediction rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Empirical draw probability calibration from historical Elo gaps."""

from __future__ import annotations

import pandas as pd

from src.config import (
    BASE_DRAW_PROB,
    DRAW_ELO_BINS,
    DRAW_ELO_SCALE,
    EMPIRICAL_DRAW_PRIOR_MATCHES,
    MIN_DRAW_PROB,
    USE_EMPIRICAL_DRAW_PROB,
)


def heuristic_draw_probability(elo_diff: float) -> float:
    """Return the configured heuristic draw probability."""
    return max(MIN_DRAW_PROB, BASE_DRAW_PROB - abs(elo_diff) / DRAW_ELO_SCALE)


def _bucket_index(abs_elo_diff: float) -> int:
    """Return the empirical draw bucket index for an absolute Elo gap."""
    for index in range(len(DRAW_ELO_BINS) - 1):
        if DRAW_ELO_BINS[index] <= abs_elo_diff < DRAW_ELO_BINS[index + 1]:
            return index
    return len(DRAW_ELO_BINS) - 2


def build_empirical_draw_model(history: pd.DataFrame) -> dict[str, object]:
    """Build smoothed draw rates by absolute adjusted Elo-gap bucket."""
    if history.empty:
        return {"overall_draw_rate": BASE_DRAW_PROB, "bucket_rates": {}}

    calibration = history.copy()
    calibration["is_draw"] = (calibration["home_score"] == calibration["away_score"]).astype(float)
    overall_draw_rate = float(calibration["is_draw"].mean())
    bucket_rates: dict[int, float] = {}
    bucket_counts: dict[int, int] = {}

    for index in range(len(DRAW_ELO_BINS) - 1):
        lower = DRAW_ELO_BINS[index]
        upper = DRAW_ELO_BINS[index + 1]
        bucket = calibration[
            (calibration["adjusted_elo_diff"].abs() >= lower)
            & (calibration["adjusted_elo_diff"].abs() < upper)
        ]
        draw_count = float(bucket["is_draw"].sum())
        match_count = int(len(bucket))
        smoothed_rate = (
            draw_count + overall_draw_rate * EMPIRICAL_DRAW_PRIOR_MATCHES
        ) / (match_count + EMPIRICAL_DRAW_PRIOR_MATCHES)
        bucket_rates[index] = max(MIN_DRAW_PROB, float(smoothed_rate))
        bucket_counts[index] = match_count

    return {
        "overall_draw_rate": overall_draw_rate,
        "bucket_rates": bucket_rates,
        "bucket_counts": bucket_counts,
    }


def empirical_draw_probability(elo_diff: float, draw_model: dict[str, object] | None) -> float:
    """Return empirical draw probability for an Elo gap, with fallback."""
    if not USE_EMPIRICAL_DRAW_PROB or not draw_model:
        return heuristic_draw_probability(elo_diff)

    bucket_rates = draw_model.get("bucket_rates", {})
    bucket = _bucket_index(abs(elo_diff))
    if bucket not in bucket_rates:
        return heuristic_draw_probability(elo_diff)
    return float(bucket_rates[bucket])

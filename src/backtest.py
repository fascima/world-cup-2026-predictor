"""Backtesting utilities for Elo match probability quality."""

from __future__ import annotations

import math

import pandas as pd

from src.config import BACKTEST_START_YEAR, ELO_SCALE, HOME_ADVANTAGE_ELO, ITERATIVE_ELO_PASSES, USE_ITERATIVE_ELO
from src.draw_model import build_empirical_draw_model, empirical_draw_probability
from src.elo import classify_tournament
from src.elo import build_elo_history, build_elo_history_from_state, initialize_elo_state
from src.market_value import market_value_adjustment, median_market_value
from src.utils import normalize_probabilities


def _probabilities_from_expected(expected_home: float, elo_diff: float) -> tuple[float, float, float]:
    """Convert Elo expected score into home/draw/away probabilities."""
    draw_prob = empirical_draw_probability(elo_diff, None)
    home_win_prob = (1.0 - draw_prob) * expected_home
    away_win_prob = (1.0 - draw_prob) * (1.0 - expected_home)
    home_win_prob, draw_prob, away_win_prob = normalize_probabilities(
        [home_win_prob, draw_prob, away_win_prob]
    )
    return home_win_prob, draw_prob, away_win_prob


def _actual_index(row: pd.Series) -> int:
    """Return 0 for home win, 1 for draw, 2 for away win."""
    if row["home_score"] > row["away_score"]:
        return 0
    if row["home_score"] == row["away_score"]:
        return 1
    return 2


def score_prediction_rows(rows: pd.DataFrame) -> dict[str, float]:
    """Score rows containing W/D/L probabilities and actual scores."""
    eps = 1e-15
    log_losses = []
    brier_scores = []
    correct = 0

    for _, row in rows.iterrows():
        probs = [
            float(row["home_win_prob"]),
            float(row["draw_prob"]),
            float(row["away_win_prob"]),
        ]
        actual = _actual_index(row)
        log_losses.append(-math.log(max(eps, probs[actual])))
        brier_scores.append(
            sum((probs[index] - (1.0 if index == actual else 0.0)) ** 2 for index in range(3))
        )
        if probs.index(max(probs)) == actual:
            correct += 1

    n_matches = len(rows)
    return {
        "matches": float(n_matches),
        "log_loss": float(sum(log_losses) / n_matches),
        "brier_score": float(sum(brier_scores) / n_matches),
        "accuracy": float(correct / n_matches),
    }


def run_backtest(matches: pd.DataFrame, start_year: int = BACKTEST_START_YEAR) -> tuple[dict[str, float], pd.DataFrame]:
    """Backtest chronological Elo predictions from ``start_year`` onward."""
    training_matches = matches[matches["date"].dt.year < start_year].copy()
    holdout_matches = matches[matches["date"].dt.year >= start_year].copy()
    if USE_ITERATIVE_ELO:
        state = initialize_elo_state()
        for pass_number in range(1, max(1, ITERATIVE_ELO_PASSES)):
            build_elo_history_from_state(
                training_matches,
                state,
                collect_rows=False,
                pass_number=pass_number,
            )
        calibration, _ = build_elo_history_from_state(
            training_matches,
            state,
            collect_rows=True,
            pass_number=max(1, ITERATIVE_ELO_PASSES),
        )
        scored, _ = build_elo_history_from_state(
            holdout_matches,
            state,
            collect_rows=True,
            pass_number=max(1, ITERATIVE_ELO_PASSES) + 1,
        )
    else:
        elo_history, _ = build_elo_history(matches)
        calibration = elo_history[elo_history["date"].dt.year < start_year].copy()
        scored = elo_history[elo_history["date"].dt.year >= start_year].copy()

    if scored.empty:
        raise ValueError(f"No matches available for backtest from {start_year} onward.")

    draw_model = build_empirical_draw_model(calibration)
    probabilities = scored.apply(
        lambda row: _probabilities_from_expected_with_model(
            row["expected_home"],
            row["adjusted_elo_diff"],
            draw_model,
        ),
        axis=1,
        result_type="expand",
    )
    probabilities.columns = ["home_win_prob", "draw_prob", "away_win_prob"]
    scored = pd.concat([scored.reset_index(drop=True), probabilities.reset_index(drop=True)], axis=1)

    metrics = score_prediction_rows(scored)
    metrics["start_year"] = float(start_year)
    return metrics, scored


def _build_training_state_and_draw_model(training_matches: pd.DataFrame) -> tuple[dict[str, dict], dict[str, object]]:
    """Build a pre-holdout Elo state and draw model without future matches."""
    state = initialize_elo_state()
    if training_matches.empty:
        return state, build_empirical_draw_model(pd.DataFrame())

    if USE_ITERATIVE_ELO:
        for pass_number in range(1, max(1, ITERATIVE_ELO_PASSES)):
            build_elo_history_from_state(
                training_matches,
                state,
                collect_rows=False,
                pass_number=pass_number,
            )
        calibration, _ = build_elo_history_from_state(
            training_matches,
            state,
            collect_rows=True,
            pass_number=max(1, ITERATIVE_ELO_PASSES),
        )
    else:
        calibration, _ = build_elo_history_from_state(
            training_matches,
            state,
            collect_rows=True,
            pass_number=1,
        )

    return state, build_empirical_draw_model(calibration)


def _expected_from_adjusted_diff(adjusted_elo_diff: float) -> float:
    """Convert an adjusted Elo difference into expected home score."""
    return 1.0 / (1.0 + 10.0 ** (-adjusted_elo_diff / ELO_SCALE))


def run_world_cup_market_value_backtest(
    matches: pd.DataFrame,
    market_values_by_year: dict[int, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Compare Elo-only and Elo-plus-market-value predictions in past World Cups.

    For each tournament year in ``market_values_by_year``, ratings are trained
    only on matches before that World Cup starts. World Cup matches are then
    scored chronologically. Market values are tournament-year values, so this
    avoids using current squad values for past matches.
    """
    if not market_values_by_year:
        raise ValueError("No historical World Cup market values were provided.")

    world_cup_matches = matches[matches["tournament"].apply(classify_tournament).eq("world_cup")].copy()
    scored_rows = []

    for year in sorted(market_values_by_year):
        year_matches = world_cup_matches[world_cup_matches["date"].dt.year == year].copy()
        if year_matches.empty:
            continue

        year_market_values = market_values_by_year[year]
        tournament_teams = set(year_matches["home_team"]) | set(year_matches["away_team"])
        available_market_values = [
            value
            for team, value in year_market_values.items()
            if team in tournament_teams
        ]
        if not available_market_values:
            continue

        baseline_market_value = median_market_value(available_market_values)
        tournament_start = year_matches["date"].min()
        training_matches = matches[matches["date"] < tournament_start].copy()
        state, draw_model = _build_training_state_and_draw_model(training_matches)
        year_scored, _ = build_elo_history_from_state(
            year_matches,
            state,
            collect_rows=True,
            pass_number=max(1, ITERATIVE_ELO_PASSES) + 1 if USE_ITERATIVE_ELO else 2,
        )

        for _, row in year_scored.iterrows():
            home_team = str(row["home_team"])
            away_team = str(row["away_team"])
            home_market_value = year_market_values.get(home_team)
            away_market_value = year_market_values.get(away_team)
            if home_market_value is None or away_market_value is None:
                continue

            home_market_adjustment = market_value_adjustment(home_market_value, baseline_market_value)
            away_market_adjustment = market_value_adjustment(away_market_value, baseline_market_value)
            market_home_prediction_elo = float(row["home_prediction_elo"]) + home_market_adjustment
            market_away_prediction_elo = float(row["away_prediction_elo"]) + away_market_adjustment

            market_adjusted_elo_diff = market_home_prediction_elo - market_away_prediction_elo
            if not bool(row.get("neutral", False)):
                market_adjusted_elo_diff += HOME_ADVANTAGE_ELO
            market_expected_home = _expected_from_adjusted_diff(market_adjusted_elo_diff)

            elo_probs = _probabilities_from_expected_with_model(
                float(row["expected_home"]),
                float(row["adjusted_elo_diff"]),
                draw_model,
            )
            market_probs = _probabilities_from_expected_with_model(
                market_expected_home,
                market_adjusted_elo_diff,
                draw_model,
            )

            scored_row = row.to_dict()
            scored_row.update(
                {
                    "tournament_year": year,
                    "market_value_baseline_eur": baseline_market_value,
                    "home_market_value_eur": home_market_value,
                    "away_market_value_eur": away_market_value,
                    "home_market_value_adjustment": home_market_adjustment,
                    "away_market_value_adjustment": away_market_adjustment,
                    "market_home_prediction_elo": market_home_prediction_elo,
                    "market_away_prediction_elo": market_away_prediction_elo,
                    "market_adjusted_elo_diff": market_adjusted_elo_diff,
                    "market_expected_home": market_expected_home,
                    "elo_home_win_prob": elo_probs[0],
                    "elo_draw_prob": elo_probs[1],
                    "elo_away_win_prob": elo_probs[2],
                    "market_home_win_prob": market_probs[0],
                    "market_draw_prob": market_probs[1],
                    "market_away_win_prob": market_probs[2],
                }
            )
            scored_rows.append(scored_row)

    predictions = pd.DataFrame(scored_rows)
    if predictions.empty:
        raise ValueError(
            "No World Cup matches could be scored. Check that results.csv has actual FIFA World Cup "
            "matches and that team names match the historical market-value file."
        )

    elo_rows = predictions.rename(
        columns={
            "elo_home_win_prob": "home_win_prob",
            "elo_draw_prob": "draw_prob",
            "elo_away_win_prob": "away_win_prob",
        }
    )
    market_rows = predictions.rename(
        columns={
            "market_home_win_prob": "home_win_prob",
            "market_draw_prob": "draw_prob",
            "market_away_win_prob": "away_win_prob",
        }
    )
    elo_metrics = score_prediction_rows(elo_rows)
    market_metrics = score_prediction_rows(market_rows)
    comparison = {
        "matches": market_metrics["matches"],
        "log_loss_delta": market_metrics["log_loss"] - elo_metrics["log_loss"],
        "brier_score_delta": market_metrics["brier_score"] - elo_metrics["brier_score"],
        "accuracy_delta": market_metrics["accuracy"] - elo_metrics["accuracy"],
    }

    return {
        "elo_only": elo_metrics,
        "elo_plus_market_value": market_metrics,
        "comparison": comparison,
    }, predictions


def _probabilities_from_expected_with_model(
    expected_home: float,
    elo_diff: float,
    draw_model: dict[str, object] | None,
) -> tuple[float, float, float]:
    """Convert Elo expected score into probabilities using a draw model."""
    draw_prob = empirical_draw_probability(elo_diff, draw_model)
    home_win_prob = (1.0 - draw_prob) * expected_home
    away_win_prob = (1.0 - draw_prob) * (1.0 - expected_home)
    home_win_prob, draw_prob, away_win_prob = normalize_probabilities(
        [home_win_prob, draw_prob, away_win_prob]
    )
    return home_win_prob, draw_prob, away_win_prob

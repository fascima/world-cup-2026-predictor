"""Backtesting utilities for the separate Poisson goal model."""

from __future__ import annotations

import pandas as pd

from src.backtest import score_prediction_rows
from src.config import (
    BACKTEST_START_YEAR,
    DIXON_COLES_RHO,
    ELO_SCALE,
    ITERATIVE_ELO_PASSES,
    POISSON_DRAW_DECISION_MAX_WIN_PROB_GAP,
    POISSON_DRAW_DECISION_THRESHOLD,
    POISSON_DRAW_INFLATION,
    POISSON_GOAL_PROFILE_WEIGHT,
    POISSON_USE_DRAW_DECISION_RULE,
    USE_ITERATIVE_ELO,
)
from src.elo import build_elo_history, build_elo_history_from_state, classify_tournament, initialize_elo_state
from src.market_value import market_value_adjustment, median_market_value
from src.poisson_model import (
    fit_average_total_goals,
    fit_goal_profile_state,
    poisson_outcome_probabilities,
    update_goal_profile_state,
)


def _actual_outcome(row: pd.Series) -> str:
    """Return the actual W/D/L outcome from the home team's perspective."""
    if row["home_score"] > row["away_score"]:
        return "home"
    if row["home_score"] < row["away_score"]:
        return "away"
    return "draw"


def _top_probability_outcome(row: pd.Series) -> str:
    """Return the highest-probability W/D/L outcome."""
    probs = {
        "home": float(row["home_win_prob"]),
        "draw": float(row["draw_prob"]),
        "away": float(row["away_win_prob"]),
    }
    return max(probs, key=probs.get)


def _decision_outcome(
    row: pd.Series,
    decision_threshold: float = POISSON_DRAW_DECISION_THRESHOLD,
    decision_max_win_prob_gap: float = POISSON_DRAW_DECISION_MAX_WIN_PROB_GAP,
) -> str:
    """Return the Poisson model's decision-rule outcome."""
    if (
        POISSON_USE_DRAW_DECISION_RULE
        and float(row["draw_prob"]) >= decision_threshold
        and abs(float(row["home_win_prob"]) - float(row["away_win_prob"]))
        <= decision_max_win_prob_gap
    ):
        return "draw"
    return _top_probability_outcome(row)


def outcome_diagnostics(
    rows: pd.DataFrame,
    decision_threshold: float = POISSON_DRAW_DECISION_THRESHOLD,
    decision_max_win_prob_gap: float = POISSON_DRAW_DECISION_MAX_WIN_PROB_GAP,
) -> dict[str, float]:
    """Return draw-focused diagnostics in addition to core scoring metrics."""
    if rows.empty:
        return {
            "top_probability_predicted_draws": 0.0,
            "predicted_draws": 0.0,
            "actual_draws": 0.0,
            "decision_accuracy": 0.0,
            "draw_recall": 0.0,
            "draw_precision": 0.0,
            "non_draw_accuracy": 0.0,
        }

    actual = rows.apply(_actual_outcome, axis=1)
    top_probability_predicted = rows.apply(_top_probability_outcome, axis=1)
    predicted = rows.apply(
        lambda row: _decision_outcome(
            row,
            decision_threshold=decision_threshold,
            decision_max_win_prob_gap=decision_max_win_prob_gap,
        ),
        axis=1,
    )
    top_probability_predicted_draws = top_probability_predicted.eq("draw")
    predicted_draws = predicted.eq("draw")
    actual_draws = actual.eq("draw")
    true_predicted_draws = predicted_draws & actual_draws
    non_draw_actual = ~actual_draws
    decision_accuracy = float((predicted == actual).sum() / len(rows))

    draw_recall = (
        float(true_predicted_draws.sum() / actual_draws.sum())
        if actual_draws.sum() > 0
        else 0.0
    )
    draw_precision = (
        float(true_predicted_draws.sum() / predicted_draws.sum())
        if predicted_draws.sum() > 0
        else 0.0
    )
    non_draw_accuracy = (
        float((predicted[non_draw_actual] == actual[non_draw_actual]).sum() / non_draw_actual.sum())
        if non_draw_actual.sum() > 0
        else 0.0
    )

    return {
        "top_probability_predicted_draws": float(top_probability_predicted_draws.sum()),
        "predicted_draws": float(predicted_draws.sum()),
        "actual_draws": float(actual_draws.sum()),
        "decision_accuracy": decision_accuracy,
        "draw_recall": draw_recall,
        "draw_precision": draw_precision,
        "non_draw_accuracy": non_draw_accuracy,
    }


def _build_elo_calibration_and_scored(
    training_matches: pd.DataFrame,
    holdout_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pre-match Elo rows for Poisson calibration and scoring."""
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
        return calibration, scored

    elo_history, _ = build_elo_history(pd.concat([training_matches, holdout_matches], ignore_index=True))
    calibration = elo_history[elo_history["date"].isin(training_matches["date"])].copy()
    scored = elo_history[elo_history["date"].isin(holdout_matches["date"])].copy()
    return calibration, scored


def _expected_from_adjusted_diff(adjusted_elo_diff: float) -> float:
    """Convert an adjusted Elo difference into expected home score."""
    return 1.0 / (1.0 + 10.0 ** (-adjusted_elo_diff / ELO_SCALE))


def run_poisson_goal_backtest(
    matches: pd.DataFrame,
    start_year: int = BACKTEST_START_YEAR,
    draw_inflation: float | None = None,
    goal_profile_weight: float | None = None,
    dixon_coles_rho: float | None = None,
    decision_threshold: float = POISSON_DRAW_DECISION_THRESHOLD,
    decision_max_win_prob_gap: float = POISSON_DRAW_DECISION_MAX_WIN_PROB_GAP,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Backtest the separate Elo-informed Poisson goal model."""
    training_matches = matches[matches["date"].dt.year < start_year].copy()
    holdout_matches = matches[matches["date"].dt.year >= start_year].copy()
    if holdout_matches.empty:
        raise ValueError(f"No matches available for Poisson backtest from {start_year} onward.")

    active_draw_inflation = POISSON_DRAW_INFLATION if draw_inflation is None else float(draw_inflation)
    active_goal_profile_weight = (
        POISSON_GOAL_PROFILE_WEIGHT if goal_profile_weight is None else float(goal_profile_weight)
    )
    active_dixon_coles_rho = DIXON_COLES_RHO if dixon_coles_rho is None else float(dixon_coles_rho)
    calibration, scored = _build_elo_calibration_and_scored(training_matches, holdout_matches)
    average_total_goals = fit_average_total_goals(calibration if not calibration.empty else training_matches)
    goal_profile_state = fit_goal_profile_state(training_matches)
    prediction_rows: list[dict[str, object]] = []

    for _, row in scored.sort_values("date").iterrows():
        prediction = poisson_outcome_probabilities(
            float(row["expected_home"]),
            average_total_goals,
            adjusted_elo_diff=float(row["adjusted_elo_diff"]),
            goal_profile_state=goal_profile_state,
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
            draw_inflation=active_draw_inflation,
            goal_profile_weight=active_goal_profile_weight,
            dixon_coles_rho=active_dixon_coles_rho,
        )
        prediction_row = row.to_dict()
        prediction_row.update(prediction)
        prediction_rows.append(prediction_row)
        update_goal_profile_state(
            goal_profile_state,
            str(row["home_team"]),
            str(row["away_team"]),
            int(row["home_score"]),
            int(row["away_score"]),
        )

    scored = pd.DataFrame(prediction_rows)
    scored["actual_outcome"] = scored.apply(_actual_outcome, axis=1)
    scored["top_probability_outcome"] = scored.apply(_top_probability_outcome, axis=1)
    scored["decision_outcome"] = scored.apply(
        lambda row: _decision_outcome(
            row,
            decision_threshold=decision_threshold,
            decision_max_win_prob_gap=decision_max_win_prob_gap,
        ),
        axis=1,
    )

    metrics = score_prediction_rows(scored)
    metrics["top_probability_accuracy"] = metrics["accuracy"]
    metrics.update(
        outcome_diagnostics(
            scored,
            decision_threshold=decision_threshold,
            decision_max_win_prob_gap=decision_max_win_prob_gap,
        )
    )
    metrics["start_year"] = float(start_year)
    metrics["average_total_goals"] = float(average_total_goals)
    metrics["draw_inflation"] = active_draw_inflation
    metrics["goal_profile_weight"] = active_goal_profile_weight
    metrics["dixon_coles_rho"] = active_dixon_coles_rho
    metrics["decision_threshold"] = float(decision_threshold)
    metrics["decision_max_win_prob_gap"] = float(decision_max_win_prob_gap)
    return metrics, scored


def _score_prediction_frame(rows: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    """Score prediction rows whose probability columns may have a prefix."""
    if prefix:
        scoring_rows = rows.rename(
            columns={
                f"{prefix}home_win_prob": "home_win_prob",
                f"{prefix}draw_prob": "draw_prob",
                f"{prefix}away_win_prob": "away_win_prob",
            }
        )
    else:
        scoring_rows = rows
    metrics = score_prediction_rows(scoring_rows)
    metrics["top_probability_accuracy"] = metrics["accuracy"]
    metrics.update(outcome_diagnostics(scoring_rows))
    return metrics


def run_world_cup_poisson_market_value_backtest(
    matches: pd.DataFrame,
    market_values_by_year: dict[int, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Compare Poisson predictions with and without historical market values.

    Ratings, goal profiles, and average goals are trained only on matches
    before each World Cup starts. Market values come from that tournament year.
    """
    if not market_values_by_year:
        raise ValueError("No historical World Cup market values were provided.")

    active_draw_inflation = POISSON_DRAW_INFLATION
    active_goal_profile_weight = POISSON_GOAL_PROFILE_WEIGHT
    world_cup_matches = matches[matches["tournament"].apply(classify_tournament).eq("world_cup")].copy()
    prediction_rows: list[dict[str, object]] = []

    for year in sorted(market_values_by_year):
        year_matches = world_cup_matches[world_cup_matches["date"].dt.year == year].copy()
        if year_matches.empty:
            continue

        year_market_values = market_values_by_year[year]
        tournament_teams = set(year_matches["home_team"]) | set(year_matches["away_team"])
        available_market_values = [
            value for team, value in year_market_values.items() if team in tournament_teams
        ]
        if not available_market_values:
            continue

        market_baseline = median_market_value(available_market_values)
        tournament_start = year_matches["date"].min()
        training_matches = matches[matches["date"] < tournament_start].copy()
        calibration, scored = _build_elo_calibration_and_scored(training_matches, year_matches)
        average_total_goals = fit_average_total_goals(calibration if not calibration.empty else training_matches)
        goal_profile_state = fit_goal_profile_state(training_matches)

        for _, row in scored.sort_values("date").iterrows():
            home_team = str(row["home_team"])
            away_team = str(row["away_team"])
            home_market_value = year_market_values.get(home_team)
            away_market_value = year_market_values.get(away_team)
            if home_market_value is None or away_market_value is None:
                continue

            base_prediction = poisson_outcome_probabilities(
                float(row["expected_home"]),
                average_total_goals,
                adjusted_elo_diff=float(row["adjusted_elo_diff"]),
                goal_profile_state=goal_profile_state,
                home_team=home_team,
                away_team=away_team,
                draw_inflation=active_draw_inflation,
                goal_profile_weight=active_goal_profile_weight,
                dixon_coles_rho=DIXON_COLES_RHO,
            )

            home_market_adjustment = market_value_adjustment(home_market_value, market_baseline)
            away_market_adjustment = market_value_adjustment(away_market_value, market_baseline)
            market_adjusted_elo_diff = (
                float(row["adjusted_elo_diff"]) + home_market_adjustment - away_market_adjustment
            )
            market_expected_home = _expected_from_adjusted_diff(market_adjusted_elo_diff)
            market_prediction = poisson_outcome_probabilities(
                market_expected_home,
                average_total_goals,
                adjusted_elo_diff=market_adjusted_elo_diff,
                goal_profile_state=goal_profile_state,
                home_team=home_team,
                away_team=away_team,
                draw_inflation=active_draw_inflation,
                goal_profile_weight=active_goal_profile_weight,
                dixon_coles_rho=DIXON_COLES_RHO,
            )

            prediction_row = row.to_dict()
            prediction_row.update(
                {
                    "tournament_year": year,
                    "market_value_baseline_eur": market_baseline,
                    "home_market_value_eur": home_market_value,
                    "away_market_value_eur": away_market_value,
                    "home_market_value_adjustment": home_market_adjustment,
                    "away_market_value_adjustment": away_market_adjustment,
                    "market_adjusted_elo_diff": market_adjusted_elo_diff,
                    "market_expected_home": market_expected_home,
                }
            )
            for key, value in base_prediction.items():
                prediction_row[f"poisson_{key}"] = value
            for key, value in market_prediction.items():
                prediction_row[f"poisson_market_{key}"] = value
            prediction_rows.append(prediction_row)

            update_goal_profile_state(
                goal_profile_state,
                home_team,
                away_team,
                int(row["home_score"]),
                int(row["away_score"]),
            )

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise ValueError(
            "No World Cup Poisson market-value predictions could be scored. Check that team names match."
        )

    poisson_rows = predictions.rename(
        columns={
            "poisson_home_win_prob": "home_win_prob",
            "poisson_draw_prob": "draw_prob",
            "poisson_away_win_prob": "away_win_prob",
        }
    )
    poisson_market_rows = predictions.rename(
        columns={
            "poisson_market_home_win_prob": "home_win_prob",
            "poisson_market_draw_prob": "draw_prob",
            "poisson_market_away_win_prob": "away_win_prob",
        }
    )
    poisson_metrics = _score_prediction_frame(poisson_rows)
    poisson_market_metrics = _score_prediction_frame(poisson_market_rows)
    comparison = {
        "matches": poisson_market_metrics["matches"],
        "log_loss_delta": poisson_market_metrics["log_loss"] - poisson_metrics["log_loss"],
        "brier_score_delta": poisson_market_metrics["brier_score"] - poisson_metrics["brier_score"],
        "accuracy_delta": poisson_market_metrics["accuracy"] - poisson_metrics["accuracy"],
        "top_probability_accuracy_delta": (
            poisson_market_metrics["top_probability_accuracy"] - poisson_metrics["top_probability_accuracy"]
        ),
        "decision_accuracy_delta": (
            poisson_market_metrics["decision_accuracy"] - poisson_metrics["decision_accuracy"]
        ),
    }

    return {
        "poisson": poisson_metrics,
        "poisson_plus_market_value": poisson_market_metrics,
        "comparison": comparison,
    }, predictions

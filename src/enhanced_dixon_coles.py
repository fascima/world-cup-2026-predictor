"""Enhanced Dixon-Coles expected-goals layer.

This module keeps the existing Poisson and Dixon-Coles machinery intact. It
only changes how the expected-goals lambdas are calculated before the
Dixon-Coles low-score adjustment is applied.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import MutableMapping
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DIXON_COLES_RHO,
    ENHANCED_DC_ELO_DIFF_SCALE,
    ENHANCED_DC_ELO_DIFF_WEIGHT,
    ENHANCED_DC_GOAL_RATE_WEIGHT,
    ENHANCED_DC_HOME_SITE_TOTAL_GOALS_MULTIPLIER,
    ENHANCED_DC_MARKET_VALUE_WEIGHT,
    ENHANCED_DC_MAX_LOG_LAMBDA_ADJUSTMENT,
    ENHANCED_DC_MAX_REST_DAYS,
    ENHANCED_DC_NEUTRAL_TOTAL_GOALS_MULTIPLIER,
    ENHANCED_DC_PRIOR_MATCHES,
    ENHANCED_DC_RECENT_FORM_WEIGHT,
    ENHANCED_DC_REST_DAYS_WEIGHT,
    ENHANCED_DC_ROLLING_WINDOW,
    ENHANCED_DC_TOURNAMENT_TOTAL_GOALS_MULTIPLIERS,
    POISSON_GOAL_PROFILE_WEIGHT,
    POISSON_MAX_EXPECTED_GOALS,
    POISSON_MAX_GOALS,
    POISSON_MIN_EXPECTED_GOALS,
    USE_ENHANCED_DIXON_COLES,
)
from src.elo import classify_tournament
from src.market_value import canonical_team_name
from src.poisson_model import (
    apply_dixon_coles_adjustment,
    expected_goals_from_elo_expected,
    goal_profile_multipliers,
    independent_poisson_score_matrix,
    outcome_probabilities_from_score_matrix,
    total_goals_for_elo_gap,
)
from src.utils import normalize_probabilities


ENHANCED_STATE_COLUMNS = [
    "home_recent_points_per_match",
    "away_recent_points_per_match",
    "home_rolling_goals_for",
    "home_rolling_goals_against",
    "away_rolling_goals_for",
    "away_rolling_goals_against",
    "home_rest_days",
    "away_rest_days",
    "market_value_log_diff",
    "enhanced_home_log_lambda_adjustment",
    "enhanced_away_log_lambda_adjustment",
]


def initialize_enhanced_feature_state() -> dict[str, dict[str, Any]]:
    """Create an empty state for pre-match rolling enhanced features."""
    return {}


def _team_state(state: MutableMapping[str, dict[str, Any]], team: str) -> dict[str, Any]:
    """Return a mutable rolling state for one team."""
    return state.setdefault(
        team,
        {
            "matches": deque(maxlen=ENHANCED_DC_ROLLING_WINDOW),
            "last_match_date": None,
        },
    )


def update_enhanced_feature_state(
    state: MutableMapping[str, dict[str, Any]],
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    match_date: pd.Timestamp,
) -> None:
    """Update rolling form, goal, and rest state after a completed match."""
    home_points = 3.0 if home_goals > away_goals else 1.0 if home_goals == away_goals else 0.0
    away_points = 3.0 if away_goals > home_goals else 1.0 if home_goals == away_goals else 0.0
    match_date = pd.Timestamp(match_date)

    home_state = _team_state(state, home_team)
    away_state = _team_state(state, away_team)
    home_state["matches"].append(
        {"points": home_points, "goals_for": float(home_goals), "goals_against": float(away_goals)}
    )
    away_state["matches"].append(
        {"points": away_points, "goals_for": float(away_goals), "goals_against": float(home_goals)}
    )
    home_state["last_match_date"] = match_date
    away_state["last_match_date"] = match_date


def fit_enhanced_feature_state(matches: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Fit enhanced rolling feature state from completed historical matches."""
    state = initialize_enhanced_feature_state()
    if matches.empty:
        return state

    for _, match in matches.sort_values("date").iterrows():
        update_enhanced_feature_state(
            state,
            str(match["home_team"]),
            str(match["away_team"]),
            int(match["home_score"]),
            int(match["away_score"]),
            pd.Timestamp(match["date"]),
        )
    return state


def _rolling_team_features(
    state: MutableMapping[str, dict[str, Any]],
    team: str,
    match_date: pd.Timestamp,
    average_team_goals: float,
) -> dict[str, float | None]:
    """Return only information known before ``match_date`` for one team."""
    team_state = state.get(team, {})
    recent_matches = list(team_state.get("matches", []))
    n_matches = float(len(recent_matches))
    prior = float(ENHANCED_DC_PRIOR_MATCHES)
    denominator = n_matches + prior
    neutral_points_per_match = 4.0 / 3.0
    average_team_goals = max(0.1, float(average_team_goals))

    points = sum(float(match["points"]) for match in recent_matches)
    goals_for = sum(float(match["goals_for"]) for match in recent_matches)
    goals_against = sum(float(match["goals_against"]) for match in recent_matches)
    points_per_match = (points + prior * neutral_points_per_match) / denominator
    goals_for_rate = (goals_for + prior * average_team_goals) / denominator
    goals_against_rate = (goals_against + prior * average_team_goals) / denominator

    last_match_date = team_state.get("last_match_date")
    rest_days: float | None = None
    if last_match_date is not None:
        rest_days = max(0.0, float((pd.Timestamp(match_date) - pd.Timestamp(last_match_date)).days))
        rest_days = min(float(ENHANCED_DC_MAX_REST_DAYS), rest_days)

    return {
        "points_per_match": points_per_match,
        "goals_for_rate": goals_for_rate,
        "goals_against_rate": goals_against_rate,
        "rest_days": rest_days,
    }


def _market_value_log_diff(
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp | None,
    market_values_by_year: dict[int, dict[str, float]] | None,
) -> float:
    """Return log market-value ratio when historical values are available."""
    if not match_date or not market_values_by_year:
        return 0.0

    values = market_values_by_year.get(int(pd.Timestamp(match_date).year), {})
    home_value = values.get(canonical_team_name(home_team))
    away_value = values.get(canonical_team_name(away_team))
    if home_value is None or away_value is None or home_value <= 0 or away_value <= 0:
        return 0.0
    return math.log(float(home_value) / float(away_value))


def enhanced_lambda_features(
    state: MutableMapping[str, dict[str, Any]],
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    average_team_goals: float,
    adjusted_elo_diff: float,
    tournament: str | None = None,
    neutral: bool = True,
    market_values_by_year: dict[int, dict[str, float]] | None = None,
) -> dict[str, float | str]:
    """Build pre-match features used by the enhanced lambda layer."""
    home_features = _rolling_team_features(state, home_team, match_date, average_team_goals)
    away_features = _rolling_team_features(state, away_team, match_date, average_team_goals)
    home_rest = home_features["rest_days"]
    away_rest = away_features["rest_days"]
    rest_diff = 0.0
    if home_rest is not None and away_rest is not None and ENHANCED_DC_MAX_REST_DAYS > 0:
        rest_diff = (float(home_rest) - float(away_rest)) / float(ENHANCED_DC_MAX_REST_DAYS)

    market_log_diff = _market_value_log_diff(home_team, away_team, match_date, market_values_by_year)
    tournament_class = classify_tournament(tournament or "")

    return {
        "home_recent_points_per_match": float(home_features["points_per_match"]),
        "away_recent_points_per_match": float(away_features["points_per_match"]),
        "home_rolling_goals_for": float(home_features["goals_for_rate"]),
        "home_rolling_goals_against": float(home_features["goals_against_rate"]),
        "away_rolling_goals_for": float(away_features["goals_for_rate"]),
        "away_rolling_goals_against": float(away_features["goals_against_rate"]),
        "home_rest_days": float(home_rest) if home_rest is not None else 0.0,
        "away_rest_days": float(away_rest) if away_rest is not None else 0.0,
        "rest_days_diff": rest_diff,
        "market_value_log_diff": market_log_diff,
        "tournament_class": tournament_class,
        "neutral_site": float(bool(neutral)),
        "elo_diff_feature": float(adjusted_elo_diff) / float(ENHANCED_DC_ELO_DIFF_SCALE),
    }


def _clamp_log_adjustment(value: float) -> float:
    """Clamp one log-lambda adjustment to keep predictions stable."""
    cap = float(ENHANCED_DC_MAX_LOG_LAMBDA_ADJUSTMENT)
    return max(-cap, min(cap, float(value)))


def _most_likely_score(score_matrix: np.ndarray) -> tuple[int, int, float]:
    """Return the highest-probability scoreline from a score matrix."""
    row, col = np.unravel_index(int(np.argmax(score_matrix)), score_matrix.shape)
    return int(row), int(col), float(score_matrix[row, col])


def apply_enhanced_lambda_adjustment(
    home_expected_goals: float,
    away_expected_goals: float,
    features: dict[str, float | str],
    average_team_goals: float,
) -> tuple[float, float, dict[str, float]]:
    """Adjust base expected goals using pre-match enhanced features."""
    if not USE_ENHANCED_DIXON_COLES:
        return float(home_expected_goals), float(away_expected_goals), {
            "enhanced_total_goals_multiplier": 1.0,
            "enhanced_tournament_total_goals_multiplier": 1.0,
            "enhanced_site_total_goals_multiplier": 1.0,
            "enhanced_form_diff": 0.0,
            "enhanced_home_goal_rate_feature": 0.0,
            "enhanced_away_goal_rate_feature": 0.0,
            "enhanced_symmetric_advantage": 0.0,
            "enhanced_home_log_lambda_adjustment": 0.0,
            "enhanced_away_log_lambda_adjustment": 0.0,
        }

    average_team_goals = max(0.1, float(average_team_goals))
    tournament_class = str(features.get("tournament_class", "default"))
    tournament_multiplier = float(
        ENHANCED_DC_TOURNAMENT_TOTAL_GOALS_MULTIPLIERS.get(tournament_class, 1.0)
    )
    site_multiplier = (
        ENHANCED_DC_NEUTRAL_TOTAL_GOALS_MULTIPLIER
        if bool(features.get("neutral_site", 1.0))
        else ENHANCED_DC_HOME_SITE_TOTAL_GOALS_MULTIPLIER
    )

    # Scale the total goal environment first, then redistribute with team-level
    # feature advantages. This keeps tournament/site effects separate from team
    # strength effects.
    base_home = float(home_expected_goals) * tournament_multiplier * site_multiplier
    base_away = float(away_expected_goals) * tournament_multiplier * site_multiplier

    form_diff = (
        float(features["home_recent_points_per_match"])
        - float(features["away_recent_points_per_match"])
    ) / 3.0
    home_goal_feature = (
        (float(features["home_rolling_goals_for"]) - average_team_goals) / average_team_goals
        + (float(features["away_rolling_goals_against"]) - average_team_goals) / average_team_goals
    )
    away_goal_feature = (
        (float(features["away_rolling_goals_for"]) - average_team_goals) / average_team_goals
        + (float(features["home_rolling_goals_against"]) - average_team_goals) / average_team_goals
    )

    symmetric_advantage = (
        ENHANCED_DC_ELO_DIFF_WEIGHT * float(features["elo_diff_feature"])
        + ENHANCED_DC_MARKET_VALUE_WEIGHT * float(features["market_value_log_diff"])
        + ENHANCED_DC_RECENT_FORM_WEIGHT * form_diff
        + ENHANCED_DC_REST_DAYS_WEIGHT * float(features["rest_days_diff"])
    )
    home_log_adjustment = _clamp_log_adjustment(
        symmetric_advantage + ENHANCED_DC_GOAL_RATE_WEIGHT * home_goal_feature
    )
    away_log_adjustment = _clamp_log_adjustment(
        -symmetric_advantage + ENHANCED_DC_GOAL_RATE_WEIGHT * away_goal_feature
    )

    enhanced_home = base_home * math.exp(home_log_adjustment)
    enhanced_away = base_away * math.exp(away_log_adjustment)
    enhanced_home = max(POISSON_MIN_EXPECTED_GOALS, min(POISSON_MAX_EXPECTED_GOALS, enhanced_home))
    enhanced_away = max(POISSON_MIN_EXPECTED_GOALS, min(POISSON_MAX_EXPECTED_GOALS, enhanced_away))
    diagnostics = {
        "enhanced_total_goals_multiplier": tournament_multiplier * site_multiplier,
        "enhanced_tournament_total_goals_multiplier": tournament_multiplier,
        "enhanced_site_total_goals_multiplier": site_multiplier,
        "enhanced_form_diff": form_diff,
        "enhanced_home_goal_rate_feature": home_goal_feature,
        "enhanced_away_goal_rate_feature": away_goal_feature,
        "enhanced_symmetric_advantage": symmetric_advantage,
        "enhanced_home_log_lambda_adjustment": home_log_adjustment,
        "enhanced_away_log_lambda_adjustment": away_log_adjustment,
    }
    return enhanced_home, enhanced_away, diagnostics


def predict_enhanced_dixon_coles_match(
    expected_home: float,
    average_total_goals: float,
    adjusted_elo_diff: float,
    enhanced_feature_state: MutableMapping[str, dict[str, Any]],
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    tournament: str | None = None,
    neutral: bool = True,
    goal_profile_state: dict[str, dict[str, float]] | None = None,
    market_values_by_year: dict[int, dict[str, float]] | None = None,
    goal_profile_weight: float | None = POISSON_GOAL_PROFILE_WEIGHT,
    dixon_coles_rho: float = DIXON_COLES_RHO,
    max_goals: int = POISSON_MAX_GOALS,
    include_score_matrix: bool = False,
) -> dict[str, Any]:
    """Predict a match with enhanced lambdas plus Dixon-Coles adjustment."""
    total_goals = total_goals_for_elo_gap(average_total_goals, adjusted_elo_diff)
    average_team_goals = total_goals / 2.0
    home_goal_multiplier, away_goal_multiplier = goal_profile_multipliers(
        goal_profile_state,
        home_team,
        away_team,
        average_team_goals,
        profile_weight=goal_profile_weight,
    )
    base_home_lambda, base_away_lambda = expected_goals_from_elo_expected(
        expected_home,
        total_goals,
        home_goal_multiplier=home_goal_multiplier,
        away_goal_multiplier=away_goal_multiplier,
    )
    features = enhanced_lambda_features(
        enhanced_feature_state,
        home_team,
        away_team,
        match_date,
        average_team_goals,
        adjusted_elo_diff,
        tournament=tournament,
        neutral=neutral,
        market_values_by_year=market_values_by_year,
    )
    enhanced_home_lambda, enhanced_away_lambda, diagnostics = apply_enhanced_lambda_adjustment(
        base_home_lambda,
        base_away_lambda,
        features,
        average_team_goals,
    )
    score_matrix = independent_poisson_score_matrix(
        enhanced_home_lambda,
        enhanced_away_lambda,
        max_goals=max_goals,
    )
    score_matrix = apply_dixon_coles_adjustment(
        score_matrix,
        enhanced_home_lambda,
        enhanced_away_lambda,
        dixon_coles_rho,
    )
    home_win_prob, draw_prob, away_win_prob = outcome_probabilities_from_score_matrix(score_matrix)
    home_win_prob, draw_prob, away_win_prob = normalize_probabilities(
        [home_win_prob, draw_prob, away_win_prob]
    )
    most_likely_home_goals, most_likely_away_goals, most_likely_score_prob = _most_likely_score(score_matrix)
    probabilities = {
        "home": home_win_prob,
        "draw": draw_prob,
        "away": away_win_prob,
    }
    predicted_result = max(probabilities, key=probabilities.get)

    prediction: dict[str, Any] = {
        "home_expected_goals": enhanced_home_lambda,
        "away_expected_goals": enhanced_away_lambda,
        "team_a_expected_goals": enhanced_home_lambda,
        "team_b_expected_goals": enhanced_away_lambda,
        "base_home_expected_goals": base_home_lambda,
        "base_away_expected_goals": base_away_lambda,
        "total_expected_goals": total_goals,
        "home_goal_profile_multiplier": home_goal_multiplier,
        "away_goal_profile_multiplier": away_goal_multiplier,
        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob,
        "team_a_win_prob": home_win_prob,
        "team_b_win_prob": away_win_prob,
        "predicted_result": predicted_result,
        "most_likely_score": f"{most_likely_home_goals}-{most_likely_away_goals}",
        "most_likely_score_prob": most_likely_score_prob,
        "dixon_coles_rho": float(dixon_coles_rho),
    }
    prediction.update(features)
    prediction.update(diagnostics)
    if include_score_matrix:
        prediction["score_matrix"] = score_matrix
    return prediction

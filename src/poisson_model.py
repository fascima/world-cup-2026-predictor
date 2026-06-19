"""Separate Elo-informed Poisson goal model.

This module does not replace the project's current Elo W/D/L probability
model. It is an alternate conversion layer that turns Elo expected score into
expected goals, then derives match-result probabilities from independent
Poisson scoreline probabilities.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import (
    DIXON_COLES_MAX_RHO,
    DIXON_COLES_MIN_RHO,
    DIXON_COLES_RHO,
    POISSON_DEFAULT_TOTAL_GOALS,
    POISSON_DRAW_INFLATION,
    POISSON_GOAL_PROFILE_PRIOR_MATCHES,
    POISSON_GOAL_PROFILE_WEIGHT,
    POISSON_MAX_EXPECTED_GOALS,
    POISSON_MAX_GOAL_PROFILE_MULTIPLIER,
    POISSON_MAX_GOALS,
    POISSON_MIN_EXPECTED_GOALS,
    POISSON_MIN_GOAL_PROFILE_MULTIPLIER,
    POISSON_TOTAL_GOALS_CLOSE_ADJUSTMENT,
    POISSON_TOTAL_GOALS_CLOSE_GAP,
    POISSON_TOTAL_GOALS_MEDIUM_ADJUSTMENT,
    POISSON_TOTAL_GOALS_MEDIUM_GAP,
    POISSON_TOTAL_GOALS_MISMATCH_ADJUSTMENT,
    POISSON_TOTAL_GOALS_NEUTRAL_GAP,
    POISSON_TOTAL_GOALS_SMALL_ADJUSTMENT,
    POISSON_TOTAL_GOALS_SMALL_GAP,
    POISSON_USE_ELO_GAP_TOTAL_GOALS,
    POISSON_USE_GOAL_PROFILE,
    POISSON_USE_TRAINING_AVG_TOTAL_GOALS,
    USE_DIXON_COLES,
)
from src.utils import normalize_probabilities


def fit_average_total_goals(matches: pd.DataFrame) -> float:
    """Return the average total goals used by the Poisson model."""
    if (
        not POISSON_USE_TRAINING_AVG_TOTAL_GOALS
        or matches.empty
        or "home_score" not in matches.columns
        or "away_score" not in matches.columns
    ):
        return float(POISSON_DEFAULT_TOTAL_GOALS)

    total_goals = pd.to_numeric(matches["home_score"], errors="coerce") + pd.to_numeric(
        matches["away_score"],
        errors="coerce",
    )
    total_goals = total_goals.dropna()
    if total_goals.empty:
        return float(POISSON_DEFAULT_TOTAL_GOALS)
    return max(0.5, float(total_goals.mean()))


def initialize_goal_profile_state() -> dict[str, dict[str, float]]:
    """Create an empty goal-profile state."""
    return {}


def update_goal_profile_state(
    state: dict[str, dict[str, float]],
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> None:
    """Update team goal profiles with one completed match."""
    home = state.setdefault(home_team, {"matches": 0.0, "goals_for": 0.0, "goals_against": 0.0})
    away = state.setdefault(away_team, {"matches": 0.0, "goals_for": 0.0, "goals_against": 0.0})
    home["matches"] += 1.0
    home["goals_for"] += float(home_goals)
    home["goals_against"] += float(away_goals)
    away["matches"] += 1.0
    away["goals_for"] += float(away_goals)
    away["goals_against"] += float(home_goals)


def fit_goal_profile_state(matches: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Fit goal profiles from completed historical matches."""
    state = initialize_goal_profile_state()
    if matches.empty:
        return state
    for _, match in matches.sort_values("date").iterrows():
        update_goal_profile_state(
            state,
            str(match["home_team"]),
            str(match["away_team"]),
            int(match["home_score"]),
            int(match["away_score"]),
        )
    return state


def _team_goal_rates(
    state: dict[str, dict[str, float]],
    team: str,
    average_team_goals: float,
    prior_matches: float = POISSON_GOAL_PROFILE_PRIOR_MATCHES,
) -> tuple[float, float]:
    """Return shrunken goals-for and goals-against rates for a team."""
    profile = state.get(team, {})
    matches = float(profile.get("matches", 0.0))
    goals_for = float(profile.get("goals_for", 0.0))
    goals_against = float(profile.get("goals_against", 0.0))
    denominator = matches + prior_matches
    if denominator <= 0:
        return average_team_goals, average_team_goals
    attack_rate = (goals_for + prior_matches * average_team_goals) / denominator
    defense_allowed_rate = (goals_against + prior_matches * average_team_goals) / denominator
    return attack_rate, defense_allowed_rate


def goal_profile_multipliers(
    state: dict[str, dict[str, float]] | None,
    home_team: str | None,
    away_team: str | None,
    average_team_goals: float,
    profile_weight: float | None = POISSON_GOAL_PROFILE_WEIGHT,
) -> tuple[float, float]:
    """Return goal-trained expected-goals multipliers for both teams."""
    if not POISSON_USE_GOAL_PROFILE or not state or not home_team or not away_team:
        return 1.0, 1.0

    average_team_goals = max(0.1, float(average_team_goals))
    home_attack, home_defense_allowed = _team_goal_rates(state, home_team, average_team_goals)
    away_attack, away_defense_allowed = _team_goal_rates(state, away_team, average_team_goals)

    raw_home_multiplier = (home_attack / average_team_goals) * (away_defense_allowed / average_team_goals)
    raw_away_multiplier = (away_attack / average_team_goals) * (home_defense_allowed / average_team_goals)
    active_profile_weight = POISSON_GOAL_PROFILE_WEIGHT if profile_weight is None else float(profile_weight)
    home_multiplier = 1.0 + active_profile_weight * (raw_home_multiplier - 1.0)
    away_multiplier = 1.0 + active_profile_weight * (raw_away_multiplier - 1.0)

    home_multiplier = max(
        POISSON_MIN_GOAL_PROFILE_MULTIPLIER,
        min(POISSON_MAX_GOAL_PROFILE_MULTIPLIER, home_multiplier),
    )
    away_multiplier = max(
        POISSON_MIN_GOAL_PROFILE_MULTIPLIER,
        min(POISSON_MAX_GOAL_PROFILE_MULTIPLIER, away_multiplier),
    )
    return home_multiplier, away_multiplier


def total_goals_for_elo_gap(average_total_goals: float, adjusted_elo_diff: float | None) -> float:
    """Return total expected goals adjusted by matchup Elo gap.

    Close matches are modeled as slightly lower-scoring, which raises draw
    probability through the Poisson scoreline distribution. Large mismatches
    receive a small increase because favorites are more likely to create
    separation.
    """
    total_goals = max(0.5, float(average_total_goals))
    if not POISSON_USE_ELO_GAP_TOTAL_GOALS or adjusted_elo_diff is None:
        return total_goals

    abs_gap = abs(float(adjusted_elo_diff))
    if abs_gap < POISSON_TOTAL_GOALS_CLOSE_GAP:
        total_goals += POISSON_TOTAL_GOALS_CLOSE_ADJUSTMENT
    elif abs_gap < POISSON_TOTAL_GOALS_MEDIUM_GAP:
        total_goals += POISSON_TOTAL_GOALS_MEDIUM_ADJUSTMENT
    elif abs_gap < POISSON_TOTAL_GOALS_SMALL_GAP:
        total_goals += POISSON_TOTAL_GOALS_SMALL_ADJUSTMENT
    elif abs_gap >= POISSON_TOTAL_GOALS_NEUTRAL_GAP:
        total_goals += POISSON_TOTAL_GOALS_MISMATCH_ADJUSTMENT

    return max(0.5, total_goals)


def expected_goals_from_elo_expected(
    expected_home: float,
    average_total_goals: float,
    home_goal_multiplier: float = 1.0,
    away_goal_multiplier: float = 1.0,
) -> tuple[float, float]:
    """Convert Elo expected score into home and away expected goals.

    The model treats Elo expected score as the home team's share of expected
    goals. This is intentionally simple and separate from the current Elo
    W/D/L heuristic.
    """
    expected_home = max(0.01, min(0.99, float(expected_home)))
    total_goals = max(0.5, float(average_total_goals))
    home_goals = total_goals * expected_home * max(0.01, float(home_goal_multiplier))
    away_goals = total_goals * (1.0 - expected_home) * max(0.01, float(away_goal_multiplier))
    home_goals = max(POISSON_MIN_EXPECTED_GOALS, min(POISSON_MAX_EXPECTED_GOALS, home_goals))
    away_goals = max(POISSON_MIN_EXPECTED_GOALS, min(POISSON_MAX_EXPECTED_GOALS, away_goals))
    return home_goals, away_goals


def poisson_pmf(goals: int, expected_goals: float) -> float:
    """Return P(goals) for a Poisson distribution."""
    return math.exp(-expected_goals) * expected_goals**goals / math.factorial(goals)


def _clamp_rho(rho: float) -> float:
    """Constrain Dixon-Coles rho to a stable configured range."""
    return max(DIXON_COLES_MIN_RHO, min(DIXON_COLES_MAX_RHO, float(rho)))


def dixon_coles_tau(x: int, y: int, lambda_a: float, lambda_b: float, rho: float) -> float:
    """Return the Dixon-Coles low-score adjustment factor."""
    active_rho = _clamp_rho(rho)
    lambda_a = max(0.0, float(lambda_a))
    lambda_b = max(0.0, float(lambda_b))

    if x == 0 and y == 0:
        return 1.0 - lambda_a * lambda_b * active_rho
    if x == 0 and y == 1:
        return 1.0 + lambda_a * active_rho
    if x == 1 and y == 0:
        return 1.0 + lambda_b * active_rho
    if x == 1 and y == 1:
        return 1.0 - active_rho
    return 1.0


def independent_poisson_score_matrix(
    lambda_a: float,
    lambda_b: float,
    max_goals: int = POISSON_MAX_GOALS,
) -> np.ndarray:
    """Return an independent Poisson score matrix for 0..max_goals."""
    lambda_a = max(POISSON_MIN_EXPECTED_GOALS, float(lambda_a))
    lambda_b = max(POISSON_MIN_EXPECTED_GOALS, float(lambda_b))
    team_a_probs = np.array([poisson_pmf(goals, lambda_a) for goals in range(max_goals + 1)])
    team_b_probs = np.array([poisson_pmf(goals, lambda_b) for goals in range(max_goals + 1)])
    matrix = np.outer(team_a_probs, team_b_probs)
    total = float(matrix.sum())
    if total <= 0:
        return np.full((max_goals + 1, max_goals + 1), 1.0 / ((max_goals + 1) ** 2))
    return matrix / total


def apply_dixon_coles_adjustment(
    score_matrix: np.ndarray,
    lambda_a: float,
    lambda_b: float,
    rho: float,
) -> np.ndarray:
    """Apply Dixon-Coles low-score correction and renormalize the matrix."""
    adjusted = np.array(score_matrix, dtype=float, copy=True)
    rows, cols = adjusted.shape

    for x in range(min(rows, 2)):
        for y in range(min(cols, 2)):
            tau = dixon_coles_tau(x, y, lambda_a, lambda_b, rho)
            adjusted[x, y] *= max(0.0, tau)

    adjusted = np.clip(adjusted, 0.0, None)
    total = float(adjusted.sum())
    if total <= 0:
        return np.array(score_matrix, dtype=float, copy=True)
    return adjusted / total


def outcome_probabilities_from_score_matrix(score_matrix: np.ndarray) -> tuple[float, float, float]:
    """Sum a score matrix into team-a win, draw, and team-b win probabilities."""
    team_a_win_prob = float(np.tril(score_matrix, k=-1).sum())
    draw_prob = float(np.trace(score_matrix))
    team_b_win_prob = float(np.triu(score_matrix, k=1).sum())
    team_a_win_prob, draw_prob, team_b_win_prob = normalize_probabilities(
        [team_a_win_prob, draw_prob, team_b_win_prob]
    )
    return team_a_win_prob, draw_prob, team_b_win_prob


def predict_dixon_coles_match(
    lambda_a: float,
    lambda_b: float,
    rho: float,
    max_goals: int = POISSON_MAX_GOALS,
) -> dict[str, float | np.ndarray]:
    """Predict W/D/L probabilities using a Dixon-Coles-adjusted score matrix."""
    matrix = independent_poisson_score_matrix(lambda_a, lambda_b, max_goals=max_goals)
    adjusted_matrix = apply_dixon_coles_adjustment(matrix, lambda_a, lambda_b, rho)
    team_a_win_prob, draw_prob, team_b_win_prob = outcome_probabilities_from_score_matrix(adjusted_matrix)
    return {
        "lambda_a": float(lambda_a),
        "lambda_b": float(lambda_b),
        "rho": _clamp_rho(rho),
        "team_a_win_prob": team_a_win_prob,
        "draw_prob": draw_prob,
        "team_b_win_prob": team_b_win_prob,
        "score_matrix": adjusted_matrix,
    }


def poisson_outcome_probabilities(
    expected_home: float,
    average_total_goals: float,
    adjusted_elo_diff: float | None = None,
    goal_profile_state: dict[str, dict[str, float]] | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    draw_inflation: float | None = POISSON_DRAW_INFLATION,
    goal_profile_weight: float | None = POISSON_GOAL_PROFILE_WEIGHT,
    use_dixon_coles: bool = USE_DIXON_COLES,
    dixon_coles_rho: float | None = DIXON_COLES_RHO,
    max_goals: int = POISSON_MAX_GOALS,
) -> dict[str, float]:
    """Return W/D/L probabilities by summing Poisson scoreline probabilities."""
    active_draw_inflation = POISSON_DRAW_INFLATION if draw_inflation is None else float(draw_inflation)
    total_goals = total_goals_for_elo_gap(average_total_goals, adjusted_elo_diff)
    home_goal_multiplier, away_goal_multiplier = goal_profile_multipliers(
        goal_profile_state,
        home_team,
        away_team,
        total_goals / 2.0,
        profile_weight=goal_profile_weight,
    )
    home_expected_goals, away_expected_goals = expected_goals_from_elo_expected(
        expected_home,
        total_goals,
        home_goal_multiplier=home_goal_multiplier,
        away_goal_multiplier=away_goal_multiplier,
    )
    score_matrix = independent_poisson_score_matrix(
        home_expected_goals,
        away_expected_goals,
        max_goals=max_goals,
    )
    active_rho = _clamp_rho(DIXON_COLES_RHO if dixon_coles_rho is None else dixon_coles_rho)
    if use_dixon_coles:
        score_matrix = apply_dixon_coles_adjustment(
            score_matrix,
            home_expected_goals,
            away_expected_goals,
            active_rho,
        )

    home_win_prob, draw_prob, away_win_prob = outcome_probabilities_from_score_matrix(score_matrix)
    if not use_dixon_coles and active_draw_inflation != 1.0:
        home_win_prob, draw_prob, away_win_prob = normalize_probabilities(
            [home_win_prob, draw_prob * active_draw_inflation, away_win_prob]
        )

    return {
        "home_expected_goals": home_expected_goals,
        "away_expected_goals": away_expected_goals,
        "total_expected_goals": total_goals,
        "home_goal_profile_multiplier": home_goal_multiplier,
        "away_goal_profile_multiplier": away_goal_multiplier,
        "draw_inflation": active_draw_inflation,
        "use_dixon_coles": float(use_dixon_coles),
        "dixon_coles_rho": active_rho if use_dixon_coles else 0.0,
        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob,
    }

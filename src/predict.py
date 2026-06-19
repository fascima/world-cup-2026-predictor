"""Match prediction helpers using Elo ratings."""

from __future__ import annotations

from src.config import (
    ELO_SCALE,
    HOME_ADVANTAGE_ELO,
    INITIAL_ELO,
)
from src.draw_model import empirical_draw_probability
from src.utils import normalize_probabilities


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return team A's Elo expected score against team B."""
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / ELO_SCALE))


def predict_match_elo(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    neutral: bool = True,
    draw_model: dict[str, object] | None = None,
) -> dict[str, float | str]:
    """Predict win/draw/loss probabilities for a match using Elo only.

    If ``neutral`` is False, team A is treated as the home team and receives
    the configured home-advantage adjustment.
    """
    team_a_elo = float(ratings.get(team_a, INITIAL_ELO))
    team_b_elo = float(ratings.get(team_b, INITIAL_ELO))
    adjusted_team_a_elo = team_a_elo if neutral else team_a_elo + HOME_ADVANTAGE_ELO
    elo_diff = adjusted_team_a_elo - team_b_elo
    expected_a = expected_score(adjusted_team_a_elo, team_b_elo)

    draw_prob = empirical_draw_probability(elo_diff, draw_model)
    team_a_win_prob = (1.0 - draw_prob) * expected_a
    team_b_win_prob = (1.0 - draw_prob) * (1.0 - expected_a)
    team_a_win_prob, draw_prob, team_b_win_prob = normalize_probabilities(
        [team_a_win_prob, draw_prob, team_b_win_prob]
    )

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_elo": team_a_elo,
        "team_b_elo": team_b_elo,
        "elo_diff": elo_diff,
        "team_a_win_prob": team_a_win_prob,
        "draw_prob": draw_prob,
        "team_b_win_prob": team_b_win_prob,
    }


def predict_knockout_advancement(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    neutral: bool = True,
    draw_model: dict[str, object] | None = None,
) -> dict[str, float | str]:
    """Predict knockout advancement probabilities.

    Draw probability is split evenly to represent extra time and penalties.
    """
    prediction = predict_match_elo(team_a, team_b, ratings, neutral=neutral, draw_model=draw_model)
    team_a_advancement = float(prediction["team_a_win_prob"]) + float(prediction["draw_prob"]) / 2.0
    team_b_advancement = float(prediction["team_b_win_prob"]) + float(prediction["draw_prob"]) / 2.0
    team_a_advancement, team_b_advancement = normalize_probabilities([team_a_advancement, team_b_advancement])

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_advancement_prob": team_a_advancement,
        "team_b_advancement_prob": team_b_advancement,
    }

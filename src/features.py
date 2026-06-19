"""Leakage-safe feature engineering for supervised match-outcome models."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import INITIAL_ELO
from src.draw_model import empirical_draw_probability
from src.elo import classify_tournament
from src.injury_features import build_pair_injury_features, neutral_pair_injury_features
from src.market_value import canonical_team_name, market_value_adjustment, median_market_value
from src.statsbomb_features import (
    build_statsbomb_pair_features,
    statsbomb_metric_defaults,
    update_statsbomb_histories_until,
)
from src.utils import normalize_probabilities


RECENT_DEFAULTS = {
    "win_rate": 0.33,
    "goals_for": 1.0,
    "goals_against": 1.0,
    "points": 1.0,
}

OPTIONAL_PROBABILITY_COLUMNS = {
    "poisson_team_a_win_prob": (
        "poisson_team_a_win_prob",
        "poisson_home_win_prob",
        "basic_poisson_home_win_prob",
    ),
    "poisson_draw_prob": (
        "poisson_draw_prob",
        "basic_poisson_draw_prob",
    ),
    "poisson_team_b_win_prob": (
        "poisson_team_b_win_prob",
        "poisson_away_win_prob",
        "basic_poisson_away_win_prob",
    ),
    "dixon_coles_team_a_win_prob": (
        "dixon_coles_team_a_win_prob",
        "dixon_coles_home_win_prob",
    ),
    "dixon_coles_draw_prob": (
        "dixon_coles_draw_prob",
    ),
    "dixon_coles_team_b_win_prob": (
        "dixon_coles_team_b_win_prob",
        "dixon_coles_away_win_prob",
    ),
}

GROUP_STAGE_MATCHES_32_TEAM_WORLD_CUP = 48
INJURED_PLAYER_REPLACEMENT_VALUE_DISCOUNT = 0.20
WORLD_CUP_PRIOR_SHRINKAGE_MATCHES = 10.0


def _neutral_statsbomb_pair_features() -> dict[str, float]:
    """Return neutral StatsBomb feature values when no StatsBomb table is provided."""
    features: dict[str, float] = {
        "team_a_statsbomb_matches_before": 0.0,
        "team_b_statsbomb_matches_before": 0.0,
        "team_a_has_statsbomb_features": 0.0,
        "team_b_has_statsbomb_features": 0.0,
        "both_teams_have_statsbomb_features": 0.0,
    }
    for window in (3, 5):
        for metric in ("xg_for", "xg_against", "xg_diff"):
            features[f"team_a_statsbomb_{metric}_last_{window}"] = 0.0
            features[f"team_b_statsbomb_{metric}_last_{window}"] = 0.0
        for metric in (
            "xg_diff",
            "non_penalty_xg_diff",
            "shot_diff",
            "box_entry_diff",
            "pressure_diff",
        ):
            features[f"statsbomb_{metric}_delta_last_{window}"] = 0.0
    return features


def classify_tournament_type(tournament: str) -> str:
    """Classify tournament text into the project's simple competition groups."""
    return classify_tournament(tournament)


def _target_from_scores(team_a_score: int, team_b_score: int) -> int:
    """Return 0 for team A win, 1 for draw, and 2 for team B win."""
    if team_a_score > team_b_score:
        return 0
    if team_a_score == team_b_score:
        return 1
    return 2


def _points_for(goals_for: int, goals_against: int) -> float:
    """Return soccer points from one team's perspective."""
    if goals_for > goals_against:
        return 3.0
    if goals_for == goals_against:
        return 1.0
    return 0.0


def get_recent_team_stats(
    team_history: Iterable[dict[str, float]],
    window: int,
) -> dict[str, float]:
    """Return rolling team stats using only prior matches in ``team_history``.

    If fewer than ``window`` matches are available, all available previous
    matches are used. If no previous matches exist, neutral defaults are used.
    """
    recent_matches = list(team_history)[-window:]
    if not recent_matches:
        return dict(RECENT_DEFAULTS)

    match_count = float(len(recent_matches))
    wins = sum(1.0 for match in recent_matches if float(match["points"]) == 3.0)
    goals_for = sum(float(match["goals_for"]) for match in recent_matches)
    goals_against = sum(float(match["goals_against"]) for match in recent_matches)
    points = sum(float(match["points"]) for match in recent_matches)
    return {
        "win_rate": wins / match_count,
        "goals_for": goals_for / match_count,
        "goals_against": goals_against / match_count,
        "points": points / match_count,
    }


def _first_existing_value(match: pd.Series, candidates: Iterable[str], default: Any = None) -> Any:
    """Return the first non-null value from candidate columns in a match row."""
    for column in candidates:
        if column in match and pd.notna(match[column]):
            return match[column]
    return default


def _add_elo_probability_features(row: dict[str, Any], match: pd.Series) -> None:
    """Add Elo W/D/L probabilities from pre-match Elo fields."""
    if all(column in match for column in ("elo_team_a_win_prob", "elo_draw_prob", "elo_team_b_win_prob")):
        row["elo_team_a_win_prob"] = float(match["elo_team_a_win_prob"])
        row["elo_draw_prob"] = float(match["elo_draw_prob"])
        row["elo_team_b_win_prob"] = float(match["elo_team_b_win_prob"])
        return

    expected_home = _first_existing_value(match, ("expected_home",), default=None)
    adjusted_elo_diff = _first_existing_value(match, ("adjusted_elo_diff", "elo_diff"), default=row["elo_diff"])
    if expected_home is None:
        expected_home = 1.0 / (1.0 + 10.0 ** (-float(adjusted_elo_diff) / 300.0))

    draw_prob = empirical_draw_probability(float(adjusted_elo_diff), draw_model=None)
    team_a_win_prob = (1.0 - draw_prob) * float(expected_home)
    team_b_win_prob = (1.0 - draw_prob) * (1.0 - float(expected_home))
    team_a_win_prob, draw_prob, team_b_win_prob = normalize_probabilities(
        [team_a_win_prob, draw_prob, team_b_win_prob]
    )
    row["elo_team_a_win_prob"] = team_a_win_prob
    row["elo_draw_prob"] = draw_prob
    row["elo_team_b_win_prob"] = team_b_win_prob


def _add_optional_probability_features(row: dict[str, Any], match: pd.Series) -> None:
    """Copy optional model probability features when source columns exist."""
    for output_column, candidates in OPTIONAL_PROBABILITY_COLUMNS.items():
        value = _first_existing_value(match, candidates, default=None)
        if value is not None:
            row[output_column] = float(value)


def _history_entry(goals_for: int, goals_against: int) -> dict[str, float]:
    """Create one completed-match entry for rolling team history."""
    return {
        "goals_for": float(goals_for),
        "goals_against": float(goals_against),
        "points": _points_for(goals_for, goals_against),
    }


def _market_value_features(
    team_a: str,
    team_b: str,
    match_date: pd.Timestamp,
    tournament: str,
    market_values_by_year: dict[int, dict[str, float]] | None,
) -> dict[str, float]:
    """Return tournament-year market value features for a match."""
    neutral_features = {
        "has_market_values": 0.0,
        "team_a_market_value_eur": 0.0,
        "team_b_market_value_eur": 0.0,
        "market_value_baseline_eur": 0.0,
        "market_value_log_ratio": 0.0,
        "market_value_adjustment_diff": 0.0,
    }
    if classify_tournament_type(tournament) != "world_cup" or not market_values_by_year:
        return neutral_features

    year_values = market_values_by_year.get(int(match_date.year), {})
    if not year_values:
        return neutral_features

    team_a_value = year_values.get(canonical_team_name(team_a))
    team_b_value = year_values.get(canonical_team_name(team_b))
    if team_a_value is None or team_b_value is None or team_a_value <= 0 or team_b_value <= 0:
        return neutral_features

    baseline = median_market_value(list(year_values.values()))
    team_a_adjustment = market_value_adjustment(team_a_value, baseline)
    team_b_adjustment = market_value_adjustment(team_b_value, baseline)
    return {
        "has_market_values": 1.0,
        "team_a_market_value_eur": float(team_a_value),
        "team_b_market_value_eur": float(team_b_value),
        "market_value_baseline_eur": float(baseline),
        "market_value_log_ratio": float(math.log(team_a_value / team_b_value)),
        "market_value_adjustment_diff": float(team_a_adjustment - team_b_adjustment),
    }


def _effective_market_value_features(
    market_features: dict[str, float],
    injury_features: dict[str, float],
) -> dict[str, float]:
    """Return market features after a conservative replacement-value injury discount."""
    neutral_features = {
        "team_a_effective_market_value_eur": 0.0,
        "team_b_effective_market_value_eur": 0.0,
        "effective_market_value_log_ratio": 0.0,
        "effective_market_value_adjustment_diff": 0.0,
        "effective_market_value_loss_diff": 0.0,
        "effective_market_value_loss_share_diff": 0.0,
    }
    if market_features.get("has_market_values", 0.0) < 0.5:
        return neutral_features

    team_a_value = float(market_features["team_a_market_value_eur"])
    team_b_value = float(market_features["team_b_market_value_eur"])
    baseline = float(market_features.get("market_value_baseline_eur", 0.0))
    if team_a_value <= 0 or team_b_value <= 0 or baseline <= 0:
        return neutral_features

    team_a_loss = INJURED_PLAYER_REPLACEMENT_VALUE_DISCOUNT * float(
        injury_features.get("team_a_injured_market_value_eur", 0.0)
    )
    team_b_loss = INJURED_PLAYER_REPLACEMENT_VALUE_DISCOUNT * float(
        injury_features.get("team_b_injured_market_value_eur", 0.0)
    )
    team_a_effective_value = max(team_a_value - team_a_loss, 1.0)
    team_b_effective_value = max(team_b_value - team_b_loss, 1.0)
    team_a_adjustment = market_value_adjustment(team_a_effective_value, baseline)
    team_b_adjustment = market_value_adjustment(team_b_effective_value, baseline)

    return {
        "team_a_effective_market_value_eur": float(team_a_effective_value),
        "team_b_effective_market_value_eur": float(team_b_effective_value),
        "effective_market_value_log_ratio": float(math.log(team_a_effective_value / team_b_effective_value)),
        "effective_market_value_adjustment_diff": float(team_a_adjustment - team_b_adjustment),
        "effective_market_value_loss_diff": float(team_a_loss - team_b_loss),
        "effective_market_value_loss_share_diff": float(
            (team_a_loss / team_a_value) - (team_b_loss / team_b_value)
        ),
    }


def _empty_group_state() -> dict[str, float]:
    """Return empty in-tournament group standings for one team."""
    return {
        "matches_played": 0.0,
        "points": 0.0,
        "goals_for": 0.0,
        "goals_against": 0.0,
        "goal_diff": 0.0,
    }


def _world_cup_group_stage_keys(matches: pd.DataFrame) -> set[int]:
    """Infer group-stage row indexes for completed 32-team World Cups."""
    keys: set[int] = set()
    world_cup_matches = matches[
        matches["tournament"].astype(str).eq("FIFA World Cup")
    ].copy()
    if world_cup_matches.empty:
        return keys

    for _, year_matches in world_cup_matches.groupby(world_cup_matches["date"].dt.year, sort=True):
        if len(year_matches) < GROUP_STAGE_MATCHES_32_TEAM_WORLD_CUP:
            continue
        group_stage = year_matches.sort_values("date", kind="mergesort").head(
            GROUP_STAGE_MATCHES_32_TEAM_WORLD_CUP
        )
        teams = set(group_stage["home_team"].astype(str)) | set(group_stage["away_team"].astype(str))
        if len(teams) != 32:
            continue
        keys.update(int(index) for index in group_stage.index)
    return keys


def _world_cup_group_state_features(
    team_a: str,
    team_b: str,
    group_states: dict[tuple[int, str], dict[str, float]],
    year: int,
    is_group_stage: bool,
) -> dict[str, float]:
    """Return pre-match World Cup group-state motivation features."""
    neutral_features = {
        "is_world_cup_group_stage": 0.0,
        "team_a_group_matches_played": 0.0,
        "team_b_group_matches_played": 0.0,
        "team_a_group_points_before": 0.0,
        "team_b_group_points_before": 0.0,
        "team_a_group_goal_diff_before": 0.0,
        "team_b_group_goal_diff_before": 0.0,
        "group_points_diff_before": 0.0,
        "group_goal_diff_diff_before": 0.0,
        "team_a_likely_qualified": 0.0,
        "team_b_likely_qualified": 0.0,
        "team_a_must_win": 0.0,
        "team_b_must_win": 0.0,
        "team_a_rotation_risk": 0.0,
        "team_b_rotation_risk": 0.0,
        "team_a_opponent_rotation_opportunity": 0.0,
        "team_b_opponent_rotation_opportunity": 0.0,
        "is_final_group_match": 0.0,
        "team_a_qualified_vs_team_b_must_win": 0.0,
        "team_b_qualified_vs_team_a_must_win": 0.0,
        "both_teams_need_result": 0.0,
        "both_teams_likely_qualified": 0.0,
        "motivation_diff": 0.0,
        "favorite_rotation_risk": 0.0,
        "underdog_motivation_boost": 0.0,
    }
    if not is_group_stage:
        return neutral_features

    team_a_state = group_states.get((year, team_a), _empty_group_state())
    team_b_state = group_states.get((year, team_b), _empty_group_state())
    team_a_played = float(team_a_state["matches_played"])
    team_b_played = float(team_b_state["matches_played"])
    team_a_points = float(team_a_state["points"])
    team_b_points = float(team_b_state["points"])
    team_a_goal_diff = float(team_a_state["goal_diff"])
    team_b_goal_diff = float(team_b_state["goal_diff"])
    team_a_likely_qualified = float(team_a_played >= 2.0 and team_a_points >= 6.0)
    team_b_likely_qualified = float(team_b_played >= 2.0 and team_b_points >= 6.0)
    team_a_must_win = float(team_a_played >= 2.0 and team_a_points <= 3.0)
    team_b_must_win = float(team_b_played >= 2.0 and team_b_points <= 3.0)
    team_a_rotation_risk = float(team_a_likely_qualified and team_b_must_win)
    team_b_rotation_risk = float(team_b_likely_qualified and team_a_must_win)
    team_a_opponent_rotation_opportunity = team_b_rotation_risk
    team_b_opponent_rotation_opportunity = team_a_rotation_risk
    is_final_group_match = float(team_a_played >= 2.0 and team_b_played >= 2.0)
    both_teams_need_result = float(is_final_group_match and team_a_must_win and team_b_must_win)
    both_teams_likely_qualified = float(
        is_final_group_match and team_a_likely_qualified and team_b_likely_qualified
    )
    team_a_motivation = team_a_must_win - team_a_likely_qualified
    team_b_motivation = team_b_must_win - team_b_likely_qualified

    return {
        "is_world_cup_group_stage": 1.0,
        "team_a_group_matches_played": team_a_played,
        "team_b_group_matches_played": team_b_played,
        "team_a_group_points_before": team_a_points,
        "team_b_group_points_before": team_b_points,
        "team_a_group_goal_diff_before": team_a_goal_diff,
        "team_b_group_goal_diff_before": team_b_goal_diff,
        "group_points_diff_before": team_a_points - team_b_points,
        "group_goal_diff_diff_before": team_a_goal_diff - team_b_goal_diff,
        "team_a_likely_qualified": team_a_likely_qualified,
        "team_b_likely_qualified": team_b_likely_qualified,
        "team_a_must_win": team_a_must_win,
        "team_b_must_win": team_b_must_win,
        "team_a_rotation_risk": team_a_rotation_risk,
        "team_b_rotation_risk": team_b_rotation_risk,
        "team_a_opponent_rotation_opportunity": team_a_opponent_rotation_opportunity,
        "team_b_opponent_rotation_opportunity": team_b_opponent_rotation_opportunity,
        "is_final_group_match": is_final_group_match,
        "team_a_qualified_vs_team_b_must_win": team_a_rotation_risk,
        "team_b_qualified_vs_team_a_must_win": team_b_rotation_risk,
        "both_teams_need_result": both_teams_need_result,
        "both_teams_likely_qualified": both_teams_likely_qualified,
        "motivation_diff": team_a_motivation - team_b_motivation,
        "favorite_rotation_risk": float(
            team_a_likely_qualified > team_b_likely_qualified
            or team_b_likely_qualified > team_a_likely_qualified
        ),
        "underdog_motivation_boost": float(
            (team_a_likely_qualified and team_b_must_win)
            or (team_b_likely_qualified and team_a_must_win)
        ),
    }


def _group_state_update(goals_for: int, goals_against: int) -> dict[str, float]:
    """Create one completed World Cup group-stage state update."""
    return {
        "matches_played": 1.0,
        "points": _points_for(goals_for, goals_against),
        "goals_for": float(goals_for),
        "goals_against": float(goals_against),
        "goal_diff": float(goals_for - goals_against),
    }


def _empty_world_cup_prior_state() -> dict[str, float]:
    """Return empty previous-World-Cup history for one team."""
    return {
        "matches": 0.0,
        "wins": 0.0,
        "draws": 0.0,
        "points": 0.0,
        "goal_diff": 0.0,
        "knockout_matches": 0.0,
    }


def _world_cup_prior_state_update(
    goals_for: int,
    goals_against: int,
    is_knockout: bool,
) -> dict[str, float]:
    """Create one historical World Cup prior update."""
    return {
        "matches": 1.0,
        "wins": float(goals_for > goals_against),
        "draws": float(goals_for == goals_against),
        "points": _points_for(goals_for, goals_against),
        "goal_diff": float(goals_for - goals_against),
        "knockout_matches": float(is_knockout),
    }


def _apply_world_cup_prior_update(
    prior_states: dict[str, dict[str, float]],
    team: str,
    update: dict[str, float],
) -> None:
    """Add one completed World Cup match to a team's prior history."""
    state = prior_states.setdefault(team, _empty_world_cup_prior_state())
    for column, value in update.items():
        state[column] = float(state[column]) + float(value)


def _world_cup_prior_team_features(state: dict[str, float] | None) -> dict[str, float]:
    """Return shrinkage-safe prior features from previous World Cups only."""
    if not state:
        state = _empty_world_cup_prior_state()

    matches = float(state["matches"])
    shrinkage_weight = matches / (matches + WORLD_CUP_PRIOR_SHRINKAGE_MATCHES) if matches > 0 else 0.0
    if matches <= 0:
        return {
            "wc_prior_matches": 0.0,
            "wc_prior_weight": 0.0,
            "wc_prior_points_per_match": 0.0,
            "wc_prior_goal_diff_per_match": 0.0,
            "wc_prior_win_rate": 0.0,
            "wc_prior_draw_rate": 0.0,
            "wc_prior_knockout_matches": 0.0,
        }

    return {
        "wc_prior_matches": matches,
        "wc_prior_weight": shrinkage_weight,
        "wc_prior_points_per_match": shrinkage_weight * (float(state["points"]) / matches),
        "wc_prior_goal_diff_per_match": shrinkage_weight * (float(state["goal_diff"]) / matches),
        "wc_prior_win_rate": shrinkage_weight * (float(state["wins"]) / matches),
        "wc_prior_draw_rate": shrinkage_weight * (float(state["draws"]) / matches),
        "wc_prior_knockout_matches": shrinkage_weight * float(state["knockout_matches"]),
    }


def _build_world_cup_prior_feature_index(
    matches: pd.DataFrame,
    group_stage_keys: set[int],
) -> dict[tuple[int, str], dict[str, float]]:
    """Build team priors by World Cup year using only previous World Cups."""
    world_cup_matches = matches[matches["tournament"].astype(str).eq("FIFA World Cup")].copy()
    if world_cup_matches.empty:
        return {}

    prior_states: dict[str, dict[str, float]] = {}
    prior_features_by_key: dict[tuple[int, str], dict[str, float]] = {}
    for year, year_matches in world_cup_matches.groupby(world_cup_matches["date"].dt.year, sort=True):
        year_int = int(year)
        teams = sorted(
            set(year_matches["home_team"].astype(str))
            | set(year_matches["away_team"].astype(str))
        )
        for team in teams:
            prior_features_by_key[(year_int, team)] = _world_cup_prior_team_features(
                prior_states.get(team)
            )

        for match_index, match in year_matches.sort_values("date", kind="mergesort").iterrows():
            team_a = str(match["home_team"])
            team_b = str(match["away_team"])
            team_a_score = int(match["home_score"])
            team_b_score = int(match["away_score"])
            is_knockout = int(match_index) not in group_stage_keys
            _apply_world_cup_prior_update(
                prior_states,
                team_a,
                _world_cup_prior_state_update(team_a_score, team_b_score, is_knockout),
            )
            _apply_world_cup_prior_update(
                prior_states,
                team_b,
                _world_cup_prior_state_update(team_b_score, team_a_score, is_knockout),
            )

    return prior_features_by_key


def _world_cup_pair_prior_features(
    team_a: str,
    team_b: str,
    year: int,
    prior_feature_index: dict[tuple[int, str], dict[str, float]],
) -> dict[str, float]:
    """Return previous-World-Cup priors for a match pair."""
    neutral = _world_cup_prior_team_features(None)
    team_a_features = prior_feature_index.get((year, team_a), neutral)
    team_b_features = prior_feature_index.get((year, team_b), neutral)
    features: dict[str, float] = {}
    for column, value in team_a_features.items():
        features[f"team_a_{column}"] = float(value)
    for column, value in team_b_features.items():
        features[f"team_b_{column}"] = float(value)

    for column in [
        "wc_prior_matches",
        "wc_prior_weight",
        "wc_prior_points_per_match",
        "wc_prior_goal_diff_per_match",
        "wc_prior_win_rate",
        "wc_prior_draw_rate",
        "wc_prior_knockout_matches",
    ]:
        features[f"{column}_diff"] = features[f"team_a_{column}"] - features[f"team_b_{column}"]
    return features


def _apply_group_state_update(
    group_states: dict[tuple[int, str], dict[str, float]],
    key: tuple[int, str],
    update: dict[str, float],
) -> None:
    """Add one completed-match update to a team's group-state table."""
    state = group_states.setdefault(key, _empty_group_state())
    for column, value in update.items():
        state[column] = float(state[column]) + float(value)


def build_ml_feature_dataset(
    matches: pd.DataFrame,
    market_values_by_year: dict[int, dict[str, float]] | None = None,
    statsbomb_team_match_features: pd.DataFrame | None = None,
    injury_feature_index: dict[tuple[int, str], dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Build one pre-match feature row per historical international match.

    The input dataframe should be cleaned and chronological. If Elo history
    columns are present they are reused; otherwise neutral Elo defaults are
    used. Rolling features are built from matches with dates strictly before
    the current match date.
    """
    required_columns = ["date", "home_team", "away_team", "home_score", "away_score"]
    missing = [column for column in required_columns if column not in matches.columns]
    if missing:
        raise ValueError(f"matches data missing required columns: {', '.join(missing)}")

    working = matches.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=required_columns).sort_values("date").reset_index(drop=True)

    team_histories: dict[str, deque[dict[str, float]]] = defaultdict(deque)
    group_stage_keys = _world_cup_group_stage_keys(working)
    world_cup_prior_feature_index = _build_world_cup_prior_feature_index(working, group_stage_keys)
    group_states: dict[tuple[int, str], dict[str, float]] = {}
    statsbomb_histories: dict[str, deque[dict[str, float]]] = defaultdict(deque)
    statsbomb_index = 0
    statsbomb_rows = pd.DataFrame()
    statsbomb_defaults: dict[str, float] = {}
    if statsbomb_team_match_features is not None and not statsbomb_team_match_features.empty:
        statsbomb_rows = statsbomb_team_match_features.copy()
        statsbomb_rows["match_date"] = pd.to_datetime(
            statsbomb_rows["match_date"],
            errors="coerce",
        )
        statsbomb_rows = statsbomb_rows.dropna(subset=["match_date", "team"]).sort_values(
            ["match_date", "match_id", "team"],
        ).reset_index(drop=True)
        statsbomb_defaults = statsbomb_metric_defaults(statsbomb_rows)
    rows: list[dict[str, Any]] = []

    for _, date_matches in working.groupby("date", sort=True):
        match_date_for_group = pd.Timestamp(date_matches["date"].iloc[0])
        if not statsbomb_rows.empty:
            statsbomb_index = update_statsbomb_histories_until(
                statsbomb_rows,
                statsbomb_histories,
                statsbomb_index,
                match_date_for_group,
            )

        pending_updates: list[tuple[str, dict[str, float]]] = []
        pending_group_updates: list[tuple[tuple[int, str], dict[str, float]]] = []

        for match_index, match in date_matches.iterrows():
            team_a = str(match["home_team"])
            team_b = str(match["away_team"])
            team_a_score = int(match["home_score"])
            team_b_score = int(match["away_score"])
            tournament = str(match.get("tournament", ""))
            neutral = bool(match.get("neutral", False))
            match_date = pd.Timestamp(match["date"])
            match_year = int(match_date.year)
            is_world_cup_group_stage = int(match_index) in group_stage_keys
            is_world_cup = classify_tournament_type(tournament) == "world_cup"
            is_world_cup_knockout = is_world_cup and not is_world_cup_group_stage

            team_a_stats_5 = get_recent_team_stats(team_histories[team_a], 5)
            team_b_stats_5 = get_recent_team_stats(team_histories[team_b], 5)
            team_a_stats_10 = get_recent_team_stats(team_histories[team_a], 10)
            team_b_stats_10 = get_recent_team_stats(team_histories[team_b], 10)

            team_a_pre_elo = float(
                _first_existing_value(
                    match,
                    ("home_pre_elo", "team_a_pre_elo", "team_a_elo"),
                    default=INITIAL_ELO,
                )
            )
            team_b_pre_elo = float(
                _first_existing_value(
                    match,
                    ("away_pre_elo", "team_b_pre_elo", "team_b_elo"),
                    default=INITIAL_ELO,
                )
            )
            elo_diff = float(_first_existing_value(match, ("elo_diff",), default=team_a_pre_elo - team_b_pre_elo))
            team_a_prediction_elo = float(
                _first_existing_value(
                    match,
                    ("home_prediction_elo", "team_a_prediction_elo"),
                    default=team_a_pre_elo,
                )
            )
            team_b_prediction_elo = float(
                _first_existing_value(
                    match,
                    ("away_prediction_elo", "team_b_prediction_elo"),
                    default=team_b_pre_elo,
                )
            )
            prediction_elo_diff = team_a_prediction_elo - team_b_prediction_elo
            adjusted_elo_diff = float(
                _first_existing_value(
                    match,
                    ("adjusted_elo_diff",),
                    default=prediction_elo_diff,
                )
            )
            abs_adjusted_elo_diff = abs(adjusted_elo_diff)
            team_a_home_advantage = 0.0 if neutral else 1.0
            team_b_home_advantage = 0.0
            team_a_attack_vs_team_b_defense_5 = (
                team_a_stats_5["goals_for"] + team_b_stats_5["goals_against"]
            )
            team_b_attack_vs_team_a_defense_5 = (
                team_b_stats_5["goals_for"] + team_a_stats_5["goals_against"]
            )
            team_a_attack_vs_team_b_defense_10 = (
                team_a_stats_10["goals_for"] + team_b_stats_10["goals_against"]
            )
            team_b_attack_vs_team_a_defense_10 = (
                team_b_stats_10["goals_for"] + team_a_stats_10["goals_against"]
            )
            market_features = _market_value_features(
                team_a,
                team_b,
                match_date,
                tournament,
                market_values_by_year,
            )
            group_state_features = _world_cup_group_state_features(
                team_a,
                team_b,
                group_states,
                match_year,
                is_world_cup_group_stage,
            )
            world_cup_prior_features = (
                _world_cup_pair_prior_features(
                    team_a,
                    team_b,
                    match_year,
                    world_cup_prior_feature_index,
                )
                if is_world_cup
                else _world_cup_pair_prior_features(team_a, team_b, match_year, {})
            )
            if statsbomb_defaults:
                statsbomb_features = build_statsbomb_pair_features(
                    team_a,
                    team_b,
                    statsbomb_histories,
                    statsbomb_defaults,
                )
            else:
                statsbomb_features = _neutral_statsbomb_pair_features()
            if classify_tournament_type(tournament) == "world_cup":
                injury_features = build_pair_injury_features(
                    team_a,
                    team_b,
                    match_year,
                    injury_feature_index,
                )
            else:
                injury_features = neutral_pair_injury_features()
            effective_market_features = _effective_market_value_features(
                market_features,
                injury_features,
            )

            row: dict[str, Any] = {
                "date": match_date,
                "team_a": team_a,
                "team_b": team_b,
                "tournament": tournament,
                "neutral": neutral,
                "team_a_score": team_a_score,
                "team_b_score": team_b_score,
                "target": _target_from_scores(team_a_score, team_b_score),
                "team_a_pre_elo": team_a_pre_elo,
                "team_b_pre_elo": team_b_pre_elo,
                "elo_diff": elo_diff,
                "team_a_prediction_elo": team_a_prediction_elo,
                "team_b_prediction_elo": team_b_prediction_elo,
                "prediction_elo_diff": prediction_elo_diff,
                "adjusted_elo_diff": adjusted_elo_diff,
                "abs_adjusted_elo_diff": abs_adjusted_elo_diff,
                "close_elo_gap_50": float(abs_adjusted_elo_diff < 50.0),
                "close_elo_gap_100": float(abs_adjusted_elo_diff < 100.0),
                "close_elo_gap_150": float(abs_adjusted_elo_diff < 150.0),
                "team_a_home_advantage": team_a_home_advantage,
                "team_b_home_advantage": team_b_home_advantage,
                "team_a_recent_win_rate_5": team_a_stats_5["win_rate"],
                "team_b_recent_win_rate_5": team_b_stats_5["win_rate"],
                "team_a_recent_win_rate_10": team_a_stats_10["win_rate"],
                "team_b_recent_win_rate_10": team_b_stats_10["win_rate"],
                "recent_win_rate_diff_5": team_a_stats_5["win_rate"] - team_b_stats_5["win_rate"],
                "recent_win_rate_diff_10": team_a_stats_10["win_rate"] - team_b_stats_10["win_rate"],
                "team_a_avg_goals_for_last_5": team_a_stats_5["goals_for"],
                "team_b_avg_goals_for_last_5": team_b_stats_5["goals_for"],
                "team_a_avg_goals_against_last_5": team_a_stats_5["goals_against"],
                "team_b_avg_goals_against_last_5": team_b_stats_5["goals_against"],
                "team_a_avg_goals_for_last_10": team_a_stats_10["goals_for"],
                "team_b_avg_goals_for_last_10": team_b_stats_10["goals_for"],
                "team_a_avg_goals_against_last_10": team_a_stats_10["goals_against"],
                "team_b_avg_goals_against_last_10": team_b_stats_10["goals_against"],
                "avg_goals_for_diff_5": team_a_stats_5["goals_for"] - team_b_stats_5["goals_for"],
                "avg_goals_for_diff_10": team_a_stats_10["goals_for"] - team_b_stats_10["goals_for"],
                "avg_goals_against_diff_5": team_a_stats_5["goals_against"] - team_b_stats_5["goals_against"],
                "avg_goals_against_diff_10": team_a_stats_10["goals_against"] - team_b_stats_10["goals_against"],
                "team_a_attack_vs_team_b_defense_last_5": team_a_attack_vs_team_b_defense_5,
                "team_b_attack_vs_team_a_defense_last_5": team_b_attack_vs_team_a_defense_5,
                "attack_defense_pressure_diff_5": (
                    team_a_attack_vs_team_b_defense_5 - team_b_attack_vs_team_a_defense_5
                ),
                "team_a_attack_vs_team_b_defense_last_10": team_a_attack_vs_team_b_defense_10,
                "team_b_attack_vs_team_a_defense_last_10": team_b_attack_vs_team_a_defense_10,
                "attack_defense_pressure_diff_10": (
                    team_a_attack_vs_team_b_defense_10 - team_b_attack_vs_team_a_defense_10
                ),
                "combined_avg_goals_for_last_5": (
                    team_a_stats_5["goals_for"] + team_b_stats_5["goals_for"]
                )
                / 2.0,
                "combined_avg_goals_for_last_10": (
                    team_a_stats_10["goals_for"] + team_b_stats_10["goals_for"]
                )
                / 2.0,
                "combined_avg_goals_against_last_5": (
                    team_a_stats_5["goals_against"] + team_b_stats_5["goals_against"]
                )
                / 2.0,
                "combined_avg_goals_against_last_10": (
                    team_a_stats_10["goals_against"] + team_b_stats_10["goals_against"]
                )
                / 2.0,
                "combined_recent_total_goals_last_5": (
                    team_a_stats_5["goals_for"]
                    + team_a_stats_5["goals_against"]
                    + team_b_stats_5["goals_for"]
                    + team_b_stats_5["goals_against"]
                )
                / 2.0,
                "combined_recent_total_goals_last_10": (
                    team_a_stats_10["goals_for"]
                    + team_a_stats_10["goals_against"]
                    + team_b_stats_10["goals_for"]
                    + team_b_stats_10["goals_against"]
                )
                / 2.0,
                "team_a_avg_points_last_5": team_a_stats_5["points"],
                "team_b_avg_points_last_5": team_b_stats_5["points"],
                "team_a_avg_points_last_10": team_a_stats_10["points"],
                "team_b_avg_points_last_10": team_b_stats_10["points"],
                "avg_points_diff_5": team_a_stats_5["points"] - team_b_stats_5["points"],
                "avg_points_diff_10": team_a_stats_10["points"] - team_b_stats_10["points"],
                "elo_diff_neutral_interaction": adjusted_elo_diff * float(neutral),
                "elo_diff_home_interaction": adjusted_elo_diff * team_a_home_advantage,
                "tournament_type": classify_tournament_type(tournament),
                "is_world_cup": float(is_world_cup),
                "is_world_cup_knockout": float(is_world_cup_knockout),
                "world_cup_knockout_abs_adjusted_elo_diff": (
                    abs_adjusted_elo_diff * float(is_world_cup_knockout)
                ),
                "world_cup_knockout_close_elo_gap_100": float(
                    is_world_cup_knockout and abs_adjusted_elo_diff < 100.0
                ),
                "world_cup_group_abs_adjusted_elo_diff": (
                    abs_adjusted_elo_diff * float(is_world_cup_group_stage)
                ),
                **market_features,
                **effective_market_features,
                **group_state_features,
                **world_cup_prior_features,
                **statsbomb_features,
                **injury_features,
            }
            _add_elo_probability_features(row, match)
            _add_optional_probability_features(row, match)
            rows.append(row)

            pending_updates.append((team_a, _history_entry(team_a_score, team_b_score)))
            pending_updates.append((team_b, _history_entry(team_b_score, team_a_score)))
            if is_world_cup_group_stage:
                pending_group_updates.append(
                    ((match_year, team_a), _group_state_update(team_a_score, team_b_score))
                )
                pending_group_updates.append(
                    ((match_year, team_b), _group_state_update(team_b_score, team_a_score))
                )

        for team, update in pending_updates:
            team_histories[team].append(update)
        for key, update in pending_group_updates:
            _apply_group_state_update(group_states, key, update)

    return pd.DataFrame(rows)


def save_ml_feature_dataset(
    features_df: pd.DataFrame,
    path: str = "data/processed/ml_match_features.csv",
) -> None:
    """Save the ML feature dataset to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

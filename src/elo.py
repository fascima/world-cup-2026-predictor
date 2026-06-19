"""Elo rating construction for international soccer."""

from __future__ import annotations

import math
import unicodedata

import pandas as pd

from src.config import (
    ELO_SCALE,
    ELO_UPDATE_MULTIPLIER,
    FAVORITE_MISMATCH_SCALE,
    FAVORITE_MISMATCH_START_ELO_GAP,
    HOME_ADVANTAGE_ELO,
    INITIAL_ELO,
    ITERATIVE_ELO_PASSES,
    K_FACTORS,
    MARGIN_OF_VICTORY_METHOD,
    MAX_OPPONENT_STRENGTH_MULTIPLIER,
    MAX_SCHEDULE_STRENGTH_ADJUSTMENT,
    MIN_FAVORITE_MISMATCH_DAMPENER,
    MIN_OPPONENT_STRENGTH_MULTIPLIER,
    OPPONENT_STRENGTH_BASELINE_ELO,
    OPPONENT_STRENGTH_SCALE,
    RECENT_FORM_K_MULTIPLIER,
    RECENT_FORM_WEIGHT,
    SCHEDULE_STRENGTH_BASELINE_ELO,
    SCHEDULE_STRENGTH_WEIGHT,
    SCHEDULE_STRENGTH_WINDOW,
    TIME_DECAY_HALF_LIFE_YEARS,
    USE_FAVORITE_MISMATCH_DAMPENER,
    USE_ITERATIVE_ELO,
    USE_MARGIN_OF_VICTORY,
    USE_OPPONENT_STRENGTH_MULTIPLIER,
    USE_RECENT_FORM_RATING,
    USE_SCHEDULE_STRENGTH_ADJUSTMENT,
    USE_TIME_DECAY,
    USE_ZERO_SUM_ELO_UPDATES,
)


QUALIFIER_TERMS = ("qualification", "qualifier", "qualifiers", "qualifying", "preliminary")
CONTINENTAL_TERMS = (
    "uefa euro",
    "uefa european championship",
    "copa america",
    "africa cup of nations",
    "concacaf gold cup",
    "afc asian cup",
    "ofc nations cup",
    "uefa nations league",
    "conmebol copa america",
)


def normalize_text(value: object) -> str:
    """Normalize tournament text for readable matching."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.lower().strip().split())


def classify_tournament(tournament: str) -> str:
    """Classify a tournament string for Elo K-factor selection."""
    text = normalize_text(tournament)
    if not text:
        return "default"

    if any(term in text for term in QUALIFIER_TERMS):
        return "qualifier"
    if text == "friendly" or "friendly" in text:
        return "friendly"
    if "fifa world cup" in text or text == "world cup":
        return "world_cup"
    if any(term in text for term in CONTINENTAL_TERMS):
        return "continental"
    return "default"


def get_k_factor(tournament: str) -> float:
    """Return the configured K-factor for a tournament."""
    tournament_class = classify_tournament(tournament)
    return float(K_FACTORS.get(tournament_class, K_FACTORS["default"]))


def get_margin_multiplier(margin: int) -> float:
    """Return the Elo multiplier for a goal margin."""
    margin = abs(int(margin))
    if not USE_MARGIN_OF_VICTORY or margin == 0:
        return 1.0
    if MARGIN_OF_VICTORY_METHOD == "one_plus_log":
        return 1.0 + math.log(margin)
    raise ValueError(f"Unsupported margin of victory method: {MARGIN_OF_VICTORY_METHOD}")


def get_time_decay_weight(match_date: pd.Timestamp, latest_match_date: pd.Timestamp) -> float:
    """Return a half-life decay weight for older matches."""
    if not USE_TIME_DECAY:
        return 1.0
    age_years = max(0.0, (latest_match_date - match_date).days / 365.25)
    return float(0.5 ** (age_years / TIME_DECAY_HALF_LIFE_YEARS))


def get_opponent_strength_multiplier(opponent_elo: float) -> float:
    """Return a positive-result multiplier based on opponent pre-match rating."""
    if not USE_OPPONENT_STRENGTH_MULTIPLIER:
        return 1.0

    raw_multiplier = 1.0 + (opponent_elo - OPPONENT_STRENGTH_BASELINE_ELO) / OPPONENT_STRENGTH_SCALE
    return min(
        MAX_OPPONENT_STRENGTH_MULTIPLIER,
        max(MIN_OPPONENT_STRENGTH_MULTIPLIER, raw_multiplier),
    )


def get_result_adjusted_opponent_strength_multiplier(opponent_elo: float, score_delta: float) -> float:
    """Return opponent-strength scaling for this team's rating change.

    Positive rating changes get more credit against stronger opponents. Negative
    rating changes are reversed: losing or underperforming against a strong
    opponent is penalized less, while underperforming against a weak opponent is
    penalized more.
    """
    if not USE_OPPONENT_STRENGTH_MULTIPLIER or score_delta == 0:
        return 1.0
    if score_delta > 0:
        return get_opponent_strength_multiplier(opponent_elo)

    raw_multiplier = 1.0 + (OPPONENT_STRENGTH_BASELINE_ELO - opponent_elo) / OPPONENT_STRENGTH_SCALE
    return min(
        MAX_OPPONENT_STRENGTH_MULTIPLIER,
        max(MIN_OPPONENT_STRENGTH_MULTIPLIER, raw_multiplier),
    )


def get_shared_opponent_strength_multiplier(home_elo: float, away_elo: float, home_score_delta: float) -> float:
    """Return the opponent-strength multiplier for a zero-sum match update."""
    if not USE_OPPONENT_STRENGTH_MULTIPLIER or home_score_delta == 0:
        return 1.0
    if home_score_delta > 0:
        return get_opponent_strength_multiplier(away_elo)
    return get_opponent_strength_multiplier(home_elo)


def get_favorite_mismatch_dampener(adjusted_elo_diff: float, actual_home: float, margin: int) -> float:
    """Reduce margin-of-victory credit for expected multi-goal favorite wins."""
    if not USE_FAVORITE_MISMATCH_DAMPENER or margin <= 1 or actual_home == 0.5:
        return 1.0

    home_won = actual_home == 1.0
    away_won = actual_home == 0.0
    favorite_won = (home_won and adjusted_elo_diff > 0) or (away_won and adjusted_elo_diff < 0)
    if not favorite_won:
        return 1.0

    favorite_gap = abs(adjusted_elo_diff)
    if favorite_gap <= FAVORITE_MISMATCH_START_ELO_GAP:
        return 1.0

    raw_dampener = 1.0 - (favorite_gap - FAVORITE_MISMATCH_START_ELO_GAP) / FAVORITE_MISMATCH_SCALE
    return max(MIN_FAVORITE_MISMATCH_DAMPENER, min(1.0, raw_dampener))


def get_schedule_strength(opponent_elos: list[float]) -> float:
    """Return average recent opponent Elo for a team."""
    if not opponent_elos:
        return float(SCHEDULE_STRENGTH_BASELINE_ELO)
    recent_opponents = opponent_elos[-SCHEDULE_STRENGTH_WINDOW:]
    return float(sum(recent_opponents) / len(recent_opponents))


def get_schedule_strength_adjustment(opponent_elos: list[float]) -> float:
    """Return a capped prediction-rating adjustment from schedule strength."""
    if not USE_SCHEDULE_STRENGTH_ADJUSTMENT:
        return 0.0
    raw_adjustment = (get_schedule_strength(opponent_elos) - SCHEDULE_STRENGTH_BASELINE_ELO) * SCHEDULE_STRENGTH_WEIGHT
    return max(
        -MAX_SCHEDULE_STRENGTH_ADJUSTMENT,
        min(MAX_SCHEDULE_STRENGTH_ADJUSTMENT, raw_adjustment),
    )


def get_prediction_elo(long_term_elo: float, recent_form_elo: float, schedule_adjustment: float) -> float:
    """Blend long-term Elo, recent-form Elo, and schedule adjustment."""
    recent_weight = RECENT_FORM_WEIGHT if USE_RECENT_FORM_RATING else 0.0
    blended_elo = (1.0 - recent_weight) * long_term_elo + recent_weight * recent_form_elo
    return blended_elo + schedule_adjustment


def _expected_home_score(adjusted_elo_diff: float) -> float:
    """Convert an adjusted Elo difference into expected home score."""
    return 1.0 / (1.0 + 10.0 ** (-adjusted_elo_diff / ELO_SCALE))


def initialize_elo_state() -> dict[str, dict]:
    """Create an empty Elo state object."""
    return {"ratings": {}, "recent_ratings": {}, "opponent_histories": {}}


def _final_prediction_ratings(state: dict[str, dict]) -> dict[str, float]:
    """Return current prediction ratings from an Elo state."""
    ratings = state["ratings"]
    recent_ratings = state["recent_ratings"]
    opponent_histories = state["opponent_histories"]
    return {
        team: get_prediction_elo(
            rating,
            recent_ratings.get(team, float(INITIAL_ELO)),
            get_schedule_strength_adjustment(opponent_histories.get(team, [])),
        )
        for team, rating in ratings.items()
    }


def _process_matches(
    matches: pd.DataFrame,
    state: dict[str, dict],
    latest_match_date: pd.Timestamp,
    collect_rows: bool,
    pass_number: int = 1,
) -> list[dict[str, object]]:
    """Process matches chronologically, mutating Elo state."""
    latest_match_date = pd.to_datetime(matches["date"]).max()
    ratings = state["ratings"]
    recent_ratings = state["recent_ratings"]
    opponent_histories = state["opponent_histories"]
    rows: list[dict[str, object]] = []

    for _, match in matches.sort_values("date").iterrows():
        home_team = str(match["home_team"])
        away_team = str(match["away_team"])
        home_elo = ratings.get(home_team, float(INITIAL_ELO))
        away_elo = ratings.get(away_team, float(INITIAL_ELO))
        home_recent_form_elo = recent_ratings.get(home_team, float(INITIAL_ELO))
        away_recent_form_elo = recent_ratings.get(away_team, float(INITIAL_ELO))
        home_opponent_history = opponent_histories.get(home_team, [])
        away_opponent_history = opponent_histories.get(away_team, [])
        home_schedule_strength = get_schedule_strength(home_opponent_history)
        away_schedule_strength = get_schedule_strength(away_opponent_history)
        home_schedule_adjustment = get_schedule_strength_adjustment(home_opponent_history)
        away_schedule_adjustment = get_schedule_strength_adjustment(away_opponent_history)
        home_prediction_elo = get_prediction_elo(home_elo, home_recent_form_elo, home_schedule_adjustment)
        away_prediction_elo = get_prediction_elo(away_elo, away_recent_form_elo, away_schedule_adjustment)

        elo_diff = home_prediction_elo - away_prediction_elo
        adjusted_elo_diff = elo_diff
        if not bool(match.get("neutral", False)):
            adjusted_elo_diff += HOME_ADVANTAGE_ELO

        expected_home = _expected_home_score(adjusted_elo_diff)
        expected_away = 1.0 - expected_home
        actual_home = float(match["result"])
        actual_away = 1.0 - actual_home
        base_k = get_k_factor(str(match.get("tournament", "")))
        decay_weight = get_time_decay_weight(pd.Timestamp(match["date"]), latest_match_date)
        effective_k = base_k * decay_weight * ELO_UPDATE_MULTIPLIER
        base_margin_multiplier = get_margin_multiplier(int(abs(match["goal_diff"])))
        favorite_mismatch_dampener = get_favorite_mismatch_dampener(
            adjusted_elo_diff,
            actual_home,
            int(abs(match["goal_diff"])),
        )
        margin_multiplier = base_margin_multiplier * favorite_mismatch_dampener

        # Project-specific modification to standard Elo. Standard Elo already
        # accounts for opponent strength through expected score, but this adds
        # an absolute opponent-quality adjustment for the team that outperforms
        # expectation. The update is still zero-sum: one team's gain is the
        # other team's loss.
        home_score_delta = actual_home - expected_home
        away_score_delta = actual_away - expected_away
        if USE_ZERO_SUM_ELO_UPDATES:
            shared_opponent_strength_multiplier = get_shared_opponent_strength_multiplier(
                home_elo,
                away_elo,
                home_score_delta,
            )
            home_opponent_strength_multiplier = shared_opponent_strength_multiplier
            away_opponent_strength_multiplier = shared_opponent_strength_multiplier
            home_elo_change = (
                effective_k
                * margin_multiplier
                * shared_opponent_strength_multiplier
                * home_score_delta
            )
            away_elo_change = -home_elo_change
        else:
            shared_opponent_strength_multiplier = 0.0
            home_opponent_strength_multiplier = get_result_adjusted_opponent_strength_multiplier(
                away_elo,
                home_score_delta,
            )
            away_opponent_strength_multiplier = get_result_adjusted_opponent_strength_multiplier(
                home_elo,
                away_score_delta,
            )
            home_elo_change = (
                effective_k
                * margin_multiplier
                * home_opponent_strength_multiplier
                * home_score_delta
            )
            away_elo_change = (
                effective_k
                * margin_multiplier
                * away_opponent_strength_multiplier
                * away_score_delta
            )
        home_recent_form_elo_change = home_elo_change * RECENT_FORM_K_MULTIPLIER
        away_recent_form_elo_change = away_elo_change * RECENT_FORM_K_MULTIPLIER

        new_home_elo = home_elo + home_elo_change
        new_away_elo = away_elo + away_elo_change
        new_home_recent_form_elo = home_recent_form_elo + home_recent_form_elo_change
        new_away_recent_form_elo = away_recent_form_elo + away_recent_form_elo_change

        if collect_rows:
            row = match.to_dict()
            row.update(
                {
                    "elo_pass": pass_number,
                    "home_pre_elo": home_elo,
                    "away_pre_elo": away_elo,
                    "home_recent_form_elo": home_recent_form_elo,
                    "away_recent_form_elo": away_recent_form_elo,
                    "home_schedule_strength": home_schedule_strength,
                    "away_schedule_strength": away_schedule_strength,
                    "home_schedule_adjustment": home_schedule_adjustment,
                    "away_schedule_adjustment": away_schedule_adjustment,
                    "home_prediction_elo": home_prediction_elo,
                    "away_prediction_elo": away_prediction_elo,
                    "elo_diff": elo_diff,
                    "adjusted_elo_diff": adjusted_elo_diff,
                    "expected_home": expected_home,
                    "expected_away": expected_away,
                    "base_k": base_k,
                    "decay_weight": decay_weight,
                    "elo_update_multiplier": ELO_UPDATE_MULTIPLIER,
                    "effective_k": effective_k,
                    "base_margin_multiplier": base_margin_multiplier,
                    "favorite_mismatch_dampener": favorite_mismatch_dampener,
                    "margin_multiplier": margin_multiplier,
                    "shared_opponent_strength_multiplier": shared_opponent_strength_multiplier,
                    "home_opponent_strength_multiplier": home_opponent_strength_multiplier,
                    "away_opponent_strength_multiplier": away_opponent_strength_multiplier,
                    "home_elo_change": home_elo_change,
                    "away_elo_change": away_elo_change,
                    "home_recent_form_elo_change": home_recent_form_elo_change,
                    "away_recent_form_elo_change": away_recent_form_elo_change,
                    "home_post_elo": new_home_elo,
                    "away_post_elo": new_away_elo,
                    "home_post_recent_form_elo": new_home_recent_form_elo,
                    "away_post_recent_form_elo": new_away_recent_form_elo,
                }
            )
            rows.append(row)

        ratings[home_team] = new_home_elo
        ratings[away_team] = new_away_elo
        recent_ratings[home_team] = new_home_recent_form_elo
        recent_ratings[away_team] = new_away_recent_form_elo
        opponent_histories.setdefault(home_team, []).append(away_prediction_elo)
        opponent_histories.setdefault(away_team, []).append(home_prediction_elo)

    return rows


def build_elo_history_from_state(
    matches: pd.DataFrame,
    state: dict[str, dict],
    collect_rows: bool = True,
    pass_number: int = 1,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build Elo history starting from an existing mutable state."""
    if matches.empty:
        return matches.copy(), _final_prediction_ratings(state)

    latest_match_date = pd.to_datetime(matches["date"]).max()
    rows = _process_matches(matches, state, latest_match_date, collect_rows=collect_rows, pass_number=pass_number)
    return pd.DataFrame(rows), _final_prediction_ratings(state)


def build_elo_history(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build Elo ratings by processing matches chronologically.

    Returns a copy of the match dataframe with pre-match Elo columns plus a
    dictionary of final prediction ratings.
    """
    state = initialize_elo_state()
    if matches.empty:
        return matches.copy(), {}

    if USE_ITERATIVE_ELO:
        latest_match_date = pd.to_datetime(matches["date"]).max()
        for pass_number in range(1, max(1, ITERATIVE_ELO_PASSES)):
            _process_matches(
                matches,
                state,
                latest_match_date,
                collect_rows=False,
                pass_number=pass_number,
            )
        return build_elo_history_from_state(
            matches,
            state,
            collect_rows=True,
            pass_number=max(1, ITERATIVE_ELO_PASSES),
        )

    return build_elo_history_from_state(matches, state, collect_rows=True, pass_number=1)


def save_elo_ratings(ratings: dict[str, float], path: str) -> None:
    """Save final Elo ratings sorted from highest to lowest."""
    ratings_df = pd.DataFrame([{"team": team, "elo": rating} for team, rating in ratings.items()])
    if not ratings_df.empty:
        ratings_df = ratings_df.sort_values("elo", ascending=False).reset_index(drop=True)
    else:
        ratings_df = pd.DataFrame(columns=["team", "elo"])
    ratings_df.to_csv(path, index=False)

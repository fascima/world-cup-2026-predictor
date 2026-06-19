"""Compare post-processing experiments for match-outcome probabilities."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.market_value import canonical_team_name, load_historical_world_cup_market_values


CLASS_LABELS = [0, 1, 2]
LOGISTIC_PATH = Path("results/logistic_backtest_predictions.csv")
GB_PATH = Path("results/gradient_boosting_backtest_predictions.csv")
POISSON_PATH = Path("results/poisson_backtest_predictions.csv")
ENHANCED_DC_PATH = Path("results/enhanced_dixon_coles_predictions.csv")
RAW_RESULTS_PATH = Path("data/raw/results.csv")
ML_FEATURE_PATH = Path("data/processed/ml_match_features.csv")
HISTORICAL_MARKET_VALUES_PATH = Path("data/fixtures/historical_world_cup_market_values.csv")
OUTPUT_PATH = Path("results/postprocess_experiment_comparison.csv")
PREDICTION_OUTPUT_PATH = Path("results/postprocess_experiment_predictions.csv")
GUARDRAIL_GRID_OUTPUT_PATH = Path("results/sanity_altitude_guardrail_grid.csv")
ABLATION_OUTPUT_PATH = Path("results/postprocess_ablation_summary.csv")

VENUE_ALTITUDE_BY_CITY_COUNTRY = {
    ("Bolivia", "La Paz"): 3640.0,
    ("Colombia", "Bogota"): 2640.0,
    ("Colombia", "Bogotá"): 2640.0,
    ("Ecuador", "Quito"): 2850.0,
    ("Mexico", "Guadalajara"): 1566.0,
    ("Mexico", "Mexico City"): 2240.0,
    ("Mexico", "Monterrey"): 540.0,
    ("Peru", "Cusco"): 3400.0,
    ("South Africa", "Johannesburg"): 1753.0,
    ("South Africa", "Pretoria"): 1339.0,
    ("United States", "Denver"): 1609.0,
}

ALTITUDE_FAMILIARITY = {
    "Bolivia": 1.0,
    "Ecuador": 0.85,
    "Mexico": 0.8,
    "Colombia": 0.65,
    "Peru": 0.55,
    "Guatemala": 0.45,
    "Costa Rica": 0.25,
    "United States": 0.2,
}


def _normalize(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    powered = np.clip(probs, 1e-12, 1.0) ** (1.0 / float(temperature))
    return _normalize(powered)


def _draw_multiplier(probs: np.ndarray, multiplier: float) -> np.ndarray:
    adjusted = np.asarray(probs, dtype=float).copy()
    adjusted[:, 1] *= float(multiplier)
    return _normalize(adjusted)


def _phase_draw_multiplier(
    rows: pd.DataFrame,
    probs: np.ndarray,
    group_multiplier: float,
    knockout_multiplier: float,
    non_world_cup_multiplier: float = 1.0,
) -> np.ndarray:
    adjusted = np.asarray(probs, dtype=float).copy()
    is_world_cup = rows["tournament"].astype(str).eq("FIFA World Cup").to_numpy()
    is_group = _numeric_column(rows, "is_world_cup_group_stage", 0.0) >= 0.5
    is_knockout = is_world_cup & ~is_group
    adjusted[~is_world_cup, 1] *= float(non_world_cup_multiplier)
    adjusted[is_world_cup & is_group, 1] *= float(group_multiplier)
    adjusted[is_knockout, 1] *= float(knockout_multiplier)
    return _normalize(adjusted)


def _shrink_to_prior(probs: np.ndarray, prior: np.ndarray, shrinkage: float) -> np.ndarray:
    return _normalize((1.0 - float(shrinkage)) * probs + float(shrinkage) * prior)


def _numeric_column(rows: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in rows.columns:
        return np.full(len(rows), default, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce").fillna(default).to_numpy(dtype=float)


def _draw_decision_predictions(
    probs: np.ndarray,
    threshold: float,
    max_win_gap: float,
) -> np.ndarray:
    pred = probs.argmax(axis=1)
    draw_mask = (probs[:, 1] >= threshold) & (np.abs(probs[:, 0] - probs[:, 2]) <= max_win_gap)
    pred[draw_mask] = 1
    return pred


def _merge_match_context(merged: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["date", "team_a", "team_b", "tournament", "neutral"]

    if RAW_RESULTS_PATH.exists():
        context = pd.read_csv(RAW_RESULTS_PATH, parse_dates=["date"])
        context = context.rename(columns={"home_team": "team_a", "away_team": "team_b"})
        context = context[key_columns + ["city", "country"]].drop_duplicates(key_columns)
        merged = merged.merge(context, on=key_columns, how="left", validate="many_to_one")

    if ML_FEATURE_PATH.exists():
        feature_columns = [
            "date",
            "team_a",
            "team_b",
            "tournament",
            "neutral",
            "team_a_pre_elo",
            "team_b_pre_elo",
            "prediction_elo_diff",
            "adjusted_elo_diff",
            "has_market_values",
            "team_a_market_value_eur",
            "team_b_market_value_eur",
            "team_a_effective_market_value_eur",
            "team_b_effective_market_value_eur",
            "market_value_log_ratio",
            "market_value_adjustment_diff",
            "effective_market_value_log_ratio",
            "effective_market_value_adjustment_diff",
            "effective_market_value_loss_diff",
            "effective_market_value_loss_share_diff",
            "team_a_home_advantage",
            "team_b_home_advantage",
            "is_world_cup_group_stage",
            "is_world_cup_knockout",
            "world_cup_knockout_abs_adjusted_elo_diff",
            "world_cup_knockout_close_elo_gap_100",
            "world_cup_group_abs_adjusted_elo_diff",
            "team_a_wc_prior_points_per_match",
            "team_b_wc_prior_points_per_match",
            "wc_prior_points_per_match_diff",
            "team_a_wc_prior_goal_diff_per_match",
            "team_b_wc_prior_goal_diff_per_match",
            "wc_prior_goal_diff_per_match_diff",
            "team_a_wc_prior_draw_rate",
            "team_b_wc_prior_draw_rate",
            "wc_prior_draw_rate_diff",
            "team_a_injured_market_value_share",
            "team_b_injured_market_value_share",
            "injured_market_value_share_diff",
        ]
        features = pd.read_csv(ML_FEATURE_PATH, parse_dates=["date"])
        available_columns = [column for column in feature_columns if column in features.columns]
        features = features[available_columns].drop_duplicates(key_columns)
        merged = merged.merge(features, on=key_columns, how="left", validate="many_to_one")

    return _add_context_features(merged)


def _historical_market_value(row: pd.Series, team_column: str, values_by_year: dict[int, dict[str, float]]) -> float:
    year_values = values_by_year.get(int(row["date"].year), {})
    return year_values.get(canonical_team_name(row[team_column]), np.nan)


def _add_context_features(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()

    for column in ["team_a_market_value_eur", "team_b_market_value_eur"]:
        if column not in rows.columns:
            rows[column] = np.nan

    if HISTORICAL_MARKET_VALUES_PATH.exists():
        values_by_year = load_historical_world_cup_market_values(HISTORICAL_MARKET_VALUES_PATH)
        for team_column, value_column in [
            ("team_a", "team_a_market_value_eur"),
            ("team_b", "team_b_market_value_eur"),
        ]:
            fallback_values = rows.apply(
                lambda row: _historical_market_value(row, team_column, values_by_year),
                axis=1,
            )
            rows[value_column] = rows[value_column].fillna(fallback_values)

    market_valid = (rows["team_a_market_value_eur"] > 0) & (rows["team_b_market_value_eur"] > 0)
    if "market_value_log_ratio" not in rows.columns:
        rows["market_value_log_ratio"] = np.nan
    rows.loc[market_valid, "market_value_log_ratio"] = rows.loc[market_valid, "market_value_log_ratio"].fillna(
        np.log(
            rows.loc[market_valid, "team_a_market_value_eur"]
            / rows.loc[market_valid, "team_b_market_value_eur"]
        )
    )
    rows["has_market_values"] = market_valid.astype(float)

    if "country" not in rows.columns:
        rows["country"] = np.nan
    rows["team_a_is_host"] = rows.apply(lambda row: _team_is_host(row, "team_a"), axis=1)
    rows["team_b_is_host"] = rows.apply(lambda row: _team_is_host(row, "team_b"), axis=1)
    rows["venue_altitude_m"] = rows.apply(_venue_altitude_m, axis=1)
    rows["team_a_altitude_familiarity"] = rows["team_a"].map(_altitude_familiarity)
    rows["team_b_altitude_familiarity"] = rows["team_b"].map(_altitude_familiarity)
    rows["altitude_familiarity_diff"] = (
        rows["team_a_altitude_familiarity"] - rows["team_b_altitude_familiarity"]
    )
    return rows


def _team_is_host(row: pd.Series, team_column: str) -> bool:
    if pd.isna(row.get("country")):
        return False
    return canonical_team_name(row[team_column]) == canonical_team_name(row["country"])


def _venue_altitude_m(row: pd.Series) -> float:
    if pd.isna(row.get("city")) or pd.isna(row.get("country")):
        return 0.0
    country = canonical_team_name(row["country"])
    city = str(row["city"])
    return VENUE_ALTITUDE_BY_CITY_COUNTRY.get((country, city), 0.0)


def _altitude_familiarity(team: object) -> float:
    return ALTITUDE_FAMILIARITY.get(canonical_team_name(team), 0.0)


def _market_favorite(row: pd.Series, log_gap_threshold: float) -> int | None:
    log_ratio = row.get("market_value_log_ratio")
    if pd.isna(log_ratio):
        return None
    if float(log_ratio) >= log_gap_threshold:
        return 0
    if float(log_ratio) <= -log_gap_threshold:
        return 2
    return None


def _effective_market_favorite(row: pd.Series, log_gap_threshold: float) -> int | None:
    log_ratio = row.get("effective_market_value_log_ratio")
    if pd.isna(log_ratio):
        return _market_favorite(row, log_gap_threshold)
    if float(log_ratio) >= log_gap_threshold:
        return 0
    if float(log_ratio) <= -log_gap_threshold:
        return 2
    return None


def _rating_favorite(row: pd.Series, elo_gap_threshold: float) -> int | None:
    elo_diff = row.get("adjusted_elo_diff")
    if pd.isna(elo_diff):
        elo_diff = row.get("prediction_elo_diff")
    if pd.isna(elo_diff):
        return None
    if float(elo_diff) >= elo_gap_threshold:
        return 0
    if float(elo_diff) <= -elo_gap_threshold:
        return 2
    return None


def _apply_sanity_guardrails(
    rows: pd.DataFrame,
    probs: np.ndarray,
    market_log_gap_threshold: float,
    base_transfer_fraction: float,
    rating_agreement_bonus: float,
    host_extra_transfer_fraction: float,
    force_weak_host_flip: bool,
    use_effective_market: bool = False,
    draw_share: float = 0.15,
) -> np.ndarray:
    adjusted = np.asarray(probs, dtype=float).copy()
    for index, row in rows.reset_index(drop=True).iterrows():
        top = int(adjusted[index].argmax())
        if top == 1:
            continue

        market_favorite = (
            _effective_market_favorite(row, market_log_gap_threshold)
            if use_effective_market
            else _market_favorite(row, market_log_gap_threshold)
        )
        if market_favorite is None or market_favorite == top:
            continue

        transfer_fraction = base_transfer_fraction
        if _rating_favorite(row, 75.0) == market_favorite:
            transfer_fraction += rating_agreement_bonus

        top_is_weak_host = (top == 0 and row.get("team_a_is_host", False)) or (
            top == 2 and row.get("team_b_is_host", False)
        )
        if top_is_weak_host:
            transfer_fraction += host_extra_transfer_fraction

        transfer_fraction = min(max(transfer_fraction, 0.0), 0.75)
        transfer = adjusted[index, top] * transfer_fraction
        adjusted[index, top] -= transfer
        adjusted[index, market_favorite] += transfer * (1.0 - draw_share)
        adjusted[index, 1] += transfer * draw_share

        if force_weak_host_flip and top_is_weak_host and adjusted[index, top] >= adjusted[index, market_favorite]:
            needed = adjusted[index, top] - adjusted[index, market_favorite] + 0.02
            extra_transfer = min(needed / 2.0, max(adjusted[index, top] - 0.05, 0.0))
            adjusted[index, top] -= extra_transfer
            adjusted[index, market_favorite] += extra_transfer

    return _normalize(adjusted)


def _two_stage_draw_features(rows: pd.DataFrame, base_probs: np.ndarray) -> pd.DataFrame:
    """Build compact binary draw-model features from calibrated base probabilities and context."""
    team_gap = np.abs(base_probs[:, 0] - base_probs[:, 2])
    features = pd.DataFrame(
        {
            "base_draw_prob": base_probs[:, 1],
            "base_win_gap": team_gap,
            "base_max_win_prob": np.maximum(base_probs[:, 0], base_probs[:, 2]),
            "is_world_cup_group_stage": _numeric_column(rows, "is_world_cup_group_stage", 0.0),
            "is_world_cup_knockout": _numeric_column(rows, "is_world_cup_knockout", 0.0),
            "knockout_close_gap": _numeric_column(rows, "world_cup_knockout_close_elo_gap_100", 0.0),
            "abs_adjusted_elo_diff": np.abs(_numeric_column(rows, "adjusted_elo_diff", 0.0)),
            "effective_market_gap_abs": np.abs(
                _numeric_column(rows, "effective_market_value_log_ratio", 0.0)
            ),
            "injury_share_gap_abs": np.abs(_numeric_column(rows, "injured_market_value_share_diff", 0.0)),
            "wc_prior_draw_rate_gap_abs": np.abs(_numeric_column(rows, "wc_prior_draw_rate_diff", 0.0)),
            "wc_prior_points_gap_abs": np.abs(_numeric_column(rows, "wc_prior_points_per_match_diff", 0.0)),
        }
    )
    features["knockout_base_win_gap"] = features["is_world_cup_knockout"] * features["base_win_gap"]
    features["group_base_win_gap"] = features["is_world_cup_group_stage"] * features["base_win_gap"]
    return features


def _two_stage_draw_probabilities(
    rows: pd.DataFrame,
    base_probs: np.ndarray,
    calibration_mask: pd.Series,
    c_value: float,
    shrinkage: float,
) -> np.ndarray:
    """Use a binary draw model, then split non-draw probability by base win ratio."""
    features = _two_stage_draw_features(rows, base_probs)
    y_draw = rows["target"].astype(int).eq(1).astype(int)
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    class_weight=None,
                    max_iter=2000,
                ),
            ),
        ]
    )
    model.fit(features.loc[calibration_mask], y_draw.loc[calibration_mask])
    draw_prob = model.predict_proba(features)[:, 1]
    draw_prob = (1.0 - float(shrinkage)) * draw_prob + float(shrinkage) * base_probs[:, 1]
    win_total = np.clip(base_probs[:, 0] + base_probs[:, 2], 1e-12, None)
    team_a_share = base_probs[:, 0] / win_total
    non_draw_prob = 1.0 - draw_prob
    adjusted = np.column_stack(
        [
            non_draw_prob * team_a_share,
            draw_prob,
            non_draw_prob * (1.0 - team_a_share),
        ]
    )
    return _normalize(adjusted)


def _apply_altitude_adjustment(
    rows: pd.DataFrame,
    probs: np.ndarray,
    min_altitude_m: float,
    max_transfer_fraction: float,
    draw_share: float = 0.1,
) -> np.ndarray:
    adjusted = np.asarray(probs, dtype=float).copy()
    for index, row in rows.reset_index(drop=True).iterrows():
        altitude = float(row.get("venue_altitude_m", 0.0) or 0.0)
        if altitude < min_altitude_m:
            continue

        familiarity_diff = float(row.get("altitude_familiarity_diff", 0.0) or 0.0)
        if abs(familiarity_diff) < 0.25:
            continue

        altitude_scale = min((altitude - min_altitude_m) / max(2500.0 - min_altitude_m, 1.0), 1.0)
        transfer_fraction = max_transfer_fraction * altitude_scale * min(abs(familiarity_diff), 1.0)
        if transfer_fraction <= 0:
            continue

        to_index, from_index = (0, 2) if familiarity_diff > 0 else (2, 0)
        transfer = adjusted[index, from_index] * transfer_fraction
        adjusted[index, from_index] -= transfer
        adjusted[index, to_index] += transfer * (1.0 - draw_share)
        adjusted[index, 1] += transfer * draw_share

    return _normalize(adjusted)


def _target_from_scores(rows: pd.DataFrame) -> pd.Series:
    return np.select(
        [rows["home_score"] > rows["away_score"], rows["home_score"].eq(rows["away_score"])],
        [0, 1],
        default=2,
    )


def _load_base_frame() -> pd.DataFrame:
    logistic = pd.read_csv(LOGISTIC_PATH, parse_dates=["date"])
    gb = pd.read_csv(GB_PATH, parse_dates=["date"])
    merged = logistic.merge(
        gb,
        on=["date", "team_a", "team_b", "tournament", "neutral", "target"],
        suffixes=("_logistic", "_gb"),
        validate="one_to_one",
    )

    if POISSON_PATH.exists():
        poisson = pd.read_csv(POISSON_PATH, parse_dates=["date"])
        poisson = poisson.rename(
            columns={
                "home_team": "team_a",
                "away_team": "team_b",
                "home_win_prob": "team_a_win_prob_poisson",
                "away_win_prob": "team_b_win_prob_poisson",
                "draw_prob": "draw_prob_poisson",
            }
        )
        poisson["target_poisson"] = _target_from_scores(poisson)
        merged = merged.merge(
            poisson[
                [
                    "date",
                    "team_a",
                    "team_b",
                    "tournament",
                    "neutral",
                    "team_a_win_prob_poisson",
                    "draw_prob_poisson",
                    "team_b_win_prob_poisson",
                ]
            ],
            on=["date", "team_a", "team_b", "tournament", "neutral"],
            how="left",
            validate="one_to_one",
        )

    if ENHANCED_DC_PATH.exists():
        enhanced = pd.read_csv(ENHANCED_DC_PATH, parse_dates=["date"])
        enhanced = enhanced.rename(
            columns={
                "home_team": "team_a",
                "away_team": "team_b",
                "enhanced_dc_home_win_prob": "team_a_win_prob_enhanced_dc",
                "enhanced_dc_draw_prob": "draw_prob_enhanced_dc",
                "enhanced_dc_away_win_prob": "team_b_win_prob_enhanced_dc",
            }
        )
        merged = merged.merge(
            enhanced[
                [
                    "date",
                    "team_a",
                    "team_b",
                    "tournament",
                    "neutral",
                    "team_a_win_prob_enhanced_dc",
                    "draw_prob_enhanced_dc",
                    "team_b_win_prob_enhanced_dc",
                ]
            ],
            on=["date", "team_a", "team_b", "tournament", "neutral"],
            how="left",
            validate="one_to_one",
        )

    probability_columns = [
        column
        for column in merged.columns
        if column.endswith("_poisson") or column.endswith("_enhanced_dc")
    ]
    for column in probability_columns:
        if merged[column].isna().any():
            fallback = {
                "team_a_win_prob_poisson": "team_a_win_prob_gb",
                "draw_prob_poisson": "draw_prob_gb",
                "team_b_win_prob_poisson": "team_b_win_prob_gb",
                "team_a_win_prob_enhanced_dc": "team_a_win_prob_gb",
                "draw_prob_enhanced_dc": "draw_prob_gb",
                "team_b_win_prob_enhanced_dc": "team_b_win_prob_gb",
            }[column]
            merged[column] = merged[column].fillna(merged[fallback])
    return _merge_match_context(merged)


def _score(
    name: str,
    rows: pd.DataFrame,
    probs: np.ndarray,
    predicted: np.ndarray | None = None,
    params: dict[str, float | str | bool] | None = None,
) -> dict[str, float | str | bool]:
    if predicted is None:
        predicted = probs.argmax(axis=1)
    wc_mask = rows["date"].dt.year.eq(2022) & rows["tournament"].eq("FIFA World Cup")
    wc_probs = probs[wc_mask.to_numpy()]
    wc_pred = predicted[wc_mask.to_numpy()]
    return {
        "experiment": name,
        **(params or {}),
        "log_loss_2022_plus": log_loss(rows["target"], probs, labels=CLASS_LABELS),
        "accuracy_2022_plus": accuracy_score(rows["target"], predicted),
        "draw_predictions_2022_plus": int((predicted == 1).sum()),
        "log_loss_2022_world_cup": log_loss(rows.loc[wc_mask, "target"], wc_probs, labels=CLASS_LABELS),
        "accuracy_2022_world_cup": accuracy_score(rows.loc[wc_mask, "target"], wc_pred),
        "draw_predictions_2022_world_cup": int((wc_pred == 1).sum()),
        "actual_draws_2022_world_cup": int((rows.loc[wc_mask, "target"] == 1).sum()),
    }


def _best_by_calibration(
    rows: pd.DataFrame,
    candidate_results: list[tuple[dict[str, float | str | bool], np.ndarray]],
    calibration_mask: pd.Series,
) -> tuple[dict[str, float | str | bool], np.ndarray]:
    y_true = rows.loc[calibration_mask, "target"]
    best_key: tuple[float, float] | None = None
    best: tuple[dict[str, float | str | bool], np.ndarray] | None = None
    for params, probs in candidate_results:
        cal_probs = probs[calibration_mask.to_numpy()]
        key = (
            log_loss(y_true, cal_probs, labels=CLASS_LABELS),
            -accuracy_score(y_true, cal_probs.argmax(axis=1)),
        )
        if best_key is None or key < best_key:
            best_key = key
            best = (params, probs)
    if best is None:
        raise ValueError("No candidate results were supplied.")
    return best


def _stacking_features(rows: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "logistic_home": rows["team_a_win_prob_logistic"],
            "logistic_draw": rows["draw_prob_logistic"],
            "logistic_away": rows["team_b_win_prob_logistic"],
            "gb_home": rows["team_a_win_prob_gb"],
            "gb_draw": rows["draw_prob_gb"],
            "gb_away": rows["team_b_win_prob_gb"],
            "poisson_home": rows.get("team_a_win_prob_poisson", rows["team_a_win_prob_gb"]),
            "poisson_draw": rows.get("draw_prob_poisson", rows["draw_prob_gb"]),
            "poisson_away": rows.get("team_b_win_prob_poisson", rows["team_b_win_prob_gb"]),
            "enhanced_home": rows.get("team_a_win_prob_enhanced_dc", rows["team_a_win_prob_gb"]),
            "enhanced_draw": rows.get("draw_prob_enhanced_dc", rows["draw_prob_gb"]),
            "enhanced_away": rows.get("team_b_win_prob_enhanced_dc", rows["team_b_win_prob_gb"]),
            "gb_win_loss_gap": (rows["team_a_win_prob_gb"] - rows["team_b_win_prob_gb"]).abs(),
            "logistic_win_loss_gap": (
                rows["team_a_win_prob_logistic"] - rows["team_b_win_prob_logistic"]
            ).abs(),
            "is_world_cup": rows["tournament"].eq("FIFA World Cup").astype(float),
            "is_neutral": rows["neutral"].astype(float),
        }
    )
    return features


def run_experiments() -> pd.DataFrame:
    rows = _load_base_frame()
    world_cup_mask = rows["date"].dt.year.eq(2022) & rows["tournament"].eq("FIFA World Cup")
    calibration_mask = ~world_cup_mask
    results: list[dict[str, float | str | bool]] = []
    prediction_outputs: list[pd.DataFrame] = []
    guardrail_grid_results: list[dict[str, float | str | bool]] = []

    gb_probs = rows[["team_a_win_prob_gb", "draw_prob_gb", "team_b_win_prob_gb"]].to_numpy()
    logistic_probs = rows[
        ["team_a_win_prob_logistic", "draw_prob_logistic", "team_b_win_prob_logistic"]
    ].to_numpy()

    def add_prediction_output(name: str, probs: np.ndarray, pred: np.ndarray | None = None) -> None:
        if pred is None:
            pred = probs.argmax(axis=1)
        output_columns = [
            "date",
            "team_a",
            "team_b",
            "tournament",
            "neutral",
            "target",
            "city",
            "country",
            "team_a_market_value_eur",
            "team_b_market_value_eur",
            "market_value_log_ratio",
            "effective_market_value_log_ratio",
            "effective_market_value_loss_share_diff",
            "adjusted_elo_diff",
            "is_world_cup_group_stage",
            "is_world_cup_knockout",
            "wc_prior_points_per_match_diff",
            "wc_prior_draw_rate_diff",
            "injured_market_value_share_diff",
            "team_a_is_host",
            "team_b_is_host",
            "venue_altitude_m",
            "altitude_familiarity_diff",
        ]
        output = rows[[column for column in output_columns if column in rows.columns]].copy()
        output["experiment"] = name
        output["team_a_win_prob"] = probs[:, 0]
        output["draw_prob"] = probs[:, 1]
        output["team_b_win_prob"] = probs[:, 2]
        output["predicted_target"] = pred
        prediction_outputs.append(output)

    results.append(_score("gb_current", rows, gb_probs))
    add_prediction_output("gb_current", gb_probs)

    results.append(_score("logistic_current", rows, logistic_probs))
    add_prediction_output("logistic_current", logistic_probs)

    # Idea 1: probability calibration by temperature scaling.
    temp_candidates = []
    for temperature in [0.75, 0.85, 0.95, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]:
        probs = _temperature_scale(gb_probs, temperature)
        temp_candidates.append(({"temperature": temperature}, probs))
    params, probs = _best_by_calibration(rows, temp_candidates, calibration_mask)
    results.append(_score("gb_temperature_calibrated", rows, probs, params=params))
    add_prediction_output("gb_temperature_calibrated", probs)
    temperature_probs = probs

    conservative_guardrail_params = {
        "market_log_gap_threshold": 1.0,
        "base_transfer_fraction": 0.2,
        "rating_agreement_bonus": 0.05,
        "host_extra_transfer_fraction": 0.15,
        "force_weak_host_flip": False,
    }
    conservative_guardrail_probs = _apply_sanity_guardrails(
        rows,
        temperature_probs,
        **conservative_guardrail_params,
    )
    results.append(
        _score(
            "gb_temperature_market_guardrail_conservative",
            rows,
            conservative_guardrail_probs,
            params=conservative_guardrail_params,
        )
    )
    add_prediction_output("gb_temperature_market_guardrail_conservative", conservative_guardrail_probs)

    guardrail_params = {
        "market_log_gap_threshold": 0.75,
        "base_transfer_fraction": 0.3,
        "rating_agreement_bonus": 0.05,
        "host_extra_transfer_fraction": 0.25,
        "force_weak_host_flip": True,
    }
    guardrail_probs = _apply_sanity_guardrails(rows, temperature_probs, **guardrail_params)
    results.append(_score("gb_temperature_market_guardrail", rows, guardrail_probs, params=guardrail_params))
    add_prediction_output("gb_temperature_market_guardrail", guardrail_probs)

    effective_guardrail_params = {**guardrail_params, "use_effective_market": True}
    effective_guardrail_probs = _apply_sanity_guardrails(
        rows,
        temperature_probs,
        **effective_guardrail_params,
    )
    results.append(
        _score(
            "gb_temperature_effective_market_guardrail",
            rows,
            effective_guardrail_probs,
            params=effective_guardrail_params,
        )
    )
    add_prediction_output("gb_temperature_effective_market_guardrail", effective_guardrail_probs)

    phase_draw_candidates = []
    for group_multiplier, knockout_multiplier in product(
        [0.85, 0.95, 1.0, 1.1, 1.2, 1.35],
        [0.85, 1.0, 1.15, 1.35, 1.55, 1.75],
    ):
        probs = _phase_draw_multiplier(
            rows,
            temperature_probs,
            group_multiplier=group_multiplier,
            knockout_multiplier=knockout_multiplier,
        )
        phase_draw_candidates.append(
            (
                {
                    "group_draw_multiplier": group_multiplier,
                    "knockout_draw_multiplier": knockout_multiplier,
                },
                probs,
            )
        )
    params, phase_draw_probs = _best_by_calibration(rows, phase_draw_candidates, calibration_mask)
    results.append(_score("gb_temperature_phase_draw_calibrated", rows, phase_draw_probs, params=params))
    add_prediction_output("gb_temperature_phase_draw_calibrated", phase_draw_probs)

    two_stage_candidates = []
    for c_value, shrinkage in product([0.02, 0.05, 0.1, 0.25], [0.25, 0.5, 0.75]):
        probs = _two_stage_draw_probabilities(
            rows,
            temperature_probs,
            calibration_mask=calibration_mask,
            c_value=c_value,
            shrinkage=shrinkage,
        )
        two_stage_candidates.append(({"C": c_value, "draw_shrinkage": shrinkage}, probs))
    params, two_stage_probs = _best_by_calibration(rows, two_stage_candidates, calibration_mask)
    results.append(_score("gb_temperature_two_stage_draw", rows, two_stage_probs, params=params))
    add_prediction_output("gb_temperature_two_stage_draw", two_stage_probs)

    two_stage_guardrail_probs = _apply_sanity_guardrails(
        rows,
        two_stage_probs,
        **effective_guardrail_params,
    )
    results.append(
        _score(
            "gb_temperature_two_stage_draw_effective_guardrail",
            rows,
            two_stage_guardrail_probs,
            params={**params, **effective_guardrail_params},
        )
    )
    add_prediction_output("gb_temperature_two_stage_draw_effective_guardrail", two_stage_guardrail_probs)

    altitude_params = {
        "min_altitude_m": 1500.0,
        "max_transfer_fraction": 0.04,
    }
    altitude_probs = _apply_altitude_adjustment(rows, temperature_probs, **altitude_params)
    results.append(_score("gb_temperature_altitude_adjusted", rows, altitude_probs, params=altitude_params))
    add_prediction_output("gb_temperature_altitude_adjusted", altitude_probs)

    guardrail_altitude_probs = _apply_altitude_adjustment(rows, guardrail_probs, **altitude_params)
    results.append(
        _score(
            "gb_temperature_market_guardrail_plus_altitude",
            rows,
            guardrail_altitude_probs,
            params={**guardrail_params, **altitude_params},
        )
    )
    add_prediction_output("gb_temperature_market_guardrail_plus_altitude", guardrail_altitude_probs)

    best_guardrail_key: tuple[float, float] | None = None
    best_guardrail_grid: tuple[dict[str, float | bool], np.ndarray] | None = None
    for market_log_gap_threshold, base_transfer_fraction, host_extra_transfer_fraction, force_weak_host_flip in product(
        [0.5, 0.75, 1.0, 1.25],
        [0.15, 0.25, 0.35, 0.45],
        [0.0, 0.15, 0.3],
        [False, True],
    ):
        params = {
            "market_log_gap_threshold": market_log_gap_threshold,
            "base_transfer_fraction": base_transfer_fraction,
            "rating_agreement_bonus": 0.05,
            "host_extra_transfer_fraction": host_extra_transfer_fraction,
            "force_weak_host_flip": force_weak_host_flip,
            "use_effective_market": True,
        }
        candidate_probs = _apply_sanity_guardrails(rows, temperature_probs, **params)
        score = _score("guardrail_grid", rows, candidate_probs, params=params)
        guardrail_grid_results.append(score)
        key = (
            float(score["log_loss_2022_world_cup"]),
            -float(score["accuracy_2022_world_cup"]),
        )
        if best_guardrail_key is None or key < best_guardrail_key:
            best_guardrail_key = key
            best_guardrail_grid = (params, candidate_probs)

    if best_guardrail_grid is not None:
        params, diagnostic_probs = best_guardrail_grid
        results.append(
            _score(
                "gb_temperature_market_guardrail_wc_diagnostic_best",
                rows,
                diagnostic_probs,
                params={**params, "selection": "2022_wc_diagnostic"},
            )
        )
        add_prediction_output("gb_temperature_market_guardrail_wc_diagnostic_best", diagnostic_probs)

    # Idea 4: draw-probability calibration for log loss.
    draw_candidates = []
    for temperature, draw_multiplier in product(
        [0.85, 1.0, 1.15, 1.35, 1.6],
        [0.8, 0.95, 1.0, 1.1, 1.25, 1.4, 1.6],
    ):
        probs = _draw_multiplier(_temperature_scale(gb_probs, temperature), draw_multiplier)
        draw_candidates.append(
            (
                {
                    "temperature": temperature,
                    "draw_multiplier": draw_multiplier,
                },
                probs,
            )
        )
    params, probs = _best_by_calibration(rows, draw_candidates, calibration_mask)
    results.append(_score("gb_draw_calibrated", rows, probs, params=params))
    add_prediction_output("gb_draw_calibrated", probs)

    # Idea 4 secondary: hard draw decision rule for accuracy only.
    best_key: tuple[float, float] | None = None
    best_decision: tuple[dict[str, float], np.ndarray] | None = None
    for threshold, gap in product([0.18, 0.20, 0.22, 0.24, 0.26, 0.28], [0.02, 0.04, 0.06, 0.08, 0.10]):
        pred = _draw_decision_predictions(gb_probs.copy(), threshold, gap)
        key = (
            accuracy_score(rows.loc[calibration_mask, "target"], pred[calibration_mask.to_numpy()]),
            -float((pred[calibration_mask.to_numpy()] == 1).sum()),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_decision = ({"decision_threshold": threshold, "decision_max_win_gap": gap}, pred)
    if best_decision is not None:
        params, pred = best_decision
        results.append(_score("gb_draw_decision_rule", rows, gb_probs, predicted=pred, params=params))
        add_prediction_output("gb_draw_decision_rule", gb_probs, pred)

    # Idea 5: conservative shrink to non-WC calibration prior.
    prior_counts = np.bincount(rows.loc[calibration_mask, "target"].astype(int), minlength=3).astype(float)
    prior = prior_counts / prior_counts.sum()
    shrink_candidates = []
    for shrinkage in [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25]:
        probs = _shrink_to_prior(gb_probs, prior, shrinkage)
        shrink_candidates.append(({"shrinkage": shrinkage}, probs))
    params, probs = _best_by_calibration(rows, shrink_candidates, calibration_mask)
    results.append(_score("gb_shrink_to_prior", rows, probs, params=params))
    add_prediction_output("gb_shrink_to_prior", probs)

    # Idea 2: stacking model over existing model probabilities and context.
    features = _stacking_features(rows)
    stacker = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.05,
                    class_weight=None,
                    max_iter=2000,
                ),
            ),
        ]
    )
    stacker.fit(features.loc[calibration_mask], rows.loc[calibration_mask, "target"].astype(int))
    stack_probs = stacker.predict_proba(features)
    class_to_index = {int(label): index for index, label in enumerate(stacker.named_steps["model"].classes_)}
    aligned_stack_probs = np.zeros((len(rows), 3), dtype=float)
    for class_label in CLASS_LABELS:
        aligned_stack_probs[:, class_label] = stack_probs[:, class_to_index[class_label]]
    aligned_stack_probs = _normalize(aligned_stack_probs)
    results.append(_score("stacking_meta_model", rows, aligned_stack_probs, params={"C": 0.05}))
    add_prediction_output("stacking_meta_model", aligned_stack_probs)

    # Combination: stacker plus draw calibration.
    stack_draw_candidates = []
    for temperature, draw_multiplier in product([0.85, 1.0, 1.15, 1.35], [0.9, 1.0, 1.1, 1.25, 1.4]):
        probs = _draw_multiplier(_temperature_scale(aligned_stack_probs, temperature), draw_multiplier)
        stack_draw_candidates.append(
            (
                {"temperature": temperature, "draw_multiplier": draw_multiplier},
                probs,
            )
        )
    params, probs = _best_by_calibration(rows, stack_draw_candidates, calibration_mask)
    results.append(_score("stacking_plus_draw_calibration", rows, probs, params=params))
    add_prediction_output("stacking_plus_draw_calibration", probs)

    comparison = pd.DataFrame(results).sort_values(
        ["log_loss_2022_world_cup", "accuracy_2022_world_cup"],
        ascending=[True, False],
    )
    ablation = comparison.copy()
    baseline_rows = ablation[ablation["experiment"].eq("gb_temperature_calibrated")]
    if not baseline_rows.empty:
        baseline = baseline_rows.iloc[0]
        ablation["delta_log_loss_2022_plus_vs_temp"] = (
            ablation["log_loss_2022_plus"] - float(baseline["log_loss_2022_plus"])
        )
        ablation["delta_accuracy_2022_plus_vs_temp"] = (
            ablation["accuracy_2022_plus"] - float(baseline["accuracy_2022_plus"])
        )
        ablation["delta_log_loss_2022_wc_vs_temp"] = (
            ablation["log_loss_2022_world_cup"] - float(baseline["log_loss_2022_world_cup"])
        )
        ablation["delta_accuracy_2022_wc_vs_temp"] = (
            ablation["accuracy_2022_world_cup"] - float(baseline["accuracy_2022_world_cup"])
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)
    ablation.to_csv(ABLATION_OUTPUT_PATH, index=False)
    pd.DataFrame(guardrail_grid_results).to_csv(GUARDRAIL_GRID_OUTPUT_PATH, index=False)
    pd.concat(prediction_outputs, ignore_index=True).to_csv(PREDICTION_OUTPUT_PATH, index=False)
    return comparison


def main() -> int:
    comparison = run_experiments()
    display_columns = [
        "experiment",
        "log_loss_2022_plus",
        "accuracy_2022_plus",
        "draw_predictions_2022_plus",
        "log_loss_2022_world_cup",
        "accuracy_2022_world_cup",
        "draw_predictions_2022_world_cup",
        "actual_draws_2022_world_cup",
    ]
    print(comparison[display_columns].to_string(index=False))
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {ABLATION_OUTPUT_PATH}")
    print(f"Wrote {PREDICTION_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

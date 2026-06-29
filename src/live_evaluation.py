"""Live World Cup prediction ledger and evaluation helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.live_world_cup import FINISHED_STATUSES, load_cached_matches, normalize_team_name


PREDICTION_LEDGER_PATH = Path("results/world_cup_2026_model_prediction_ledger.csv")
BACKFILL_PREDICTIONS_PATH = Path("results/world_cup_2026_model_cached_backfill_predictions.csv")
LIVE_METRICS_PATH = Path("results/world_cup_2026_model_live_metrics.csv")
MATCH_EVALUATION_PATH = Path("results/world_cup_2026_model_match_evaluation.csv")

LEDGER_KEY_COLUMNS = ["match_id", "model_key"]
PROBABILITY_COLUMNS = ["team_a_win_prob", "draw_prob", "team_b_win_prob"]
LEDGER_COLUMNS = [
    "prediction_generated_at",
    "prediction_date",
    "match_id",
    "kickoff_utc",
    "local_date",
    "status",
    "stage",
    "group",
    "team_a",
    "team_b",
    "model_key",
    "model",
    "model_version",
    "team_a_win_prob",
    "draw_prob",
    "team_b_win_prob",
    "team_a_advancement_prob",
    "team_b_advancement_prob",
]


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string for generated prediction rows."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def load_prediction_ledger(path: Path = PREDICTION_LEDGER_PATH) -> pd.DataFrame:
    """Load the immutable prediction ledger if present."""
    if not path.exists():
        return empty_ledger()
    try:
        return pd.read_csv(path, dtype={"match_id": str})
    except pd.errors.EmptyDataError:
        return empty_ledger()


def _normalize_prediction_frame(predictions: pd.DataFrame, generated_at: str | None = None) -> pd.DataFrame:
    if predictions.empty:
        return empty_ledger()

    normalized = predictions.copy()
    normalized["match_id"] = normalized["match_id"].astype(str)
    if "prediction_generated_at" not in normalized.columns:
        if "prediction_date" in normalized.columns:
            normalized["prediction_generated_at"] = (
                pd.to_datetime(normalized["prediction_date"], errors="coerce")
                .dt.strftime("%Y-%m-%dT00:00:00Z")
                .fillna(generated_at or utc_now_iso())
            )
        else:
            normalized["prediction_generated_at"] = generated_at or utc_now_iso()
    if "model_version" not in normalized.columns:
        normalized["model_version"] = "live"
    if "prediction_date" not in normalized.columns:
        normalized["prediction_date"] = pd.to_datetime(
            normalized["prediction_generated_at"], utc=True, errors="coerce"
        ).dt.date.astype(str)

    for column in PROBABILITY_COLUMNS + ["team_a_advancement_prob", "team_b_advancement_prob"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    for column in LEDGER_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
    return normalized[LEDGER_COLUMNS].copy()


def append_predictions_to_ledger(
    predictions: pd.DataFrame,
    ledger_path: Path = PREDICTION_LEDGER_PATH,
    generated_at: str | None = None,
) -> pd.DataFrame:
    """Append only new match/model predictions, preserving all existing rows."""
    incoming = _normalize_prediction_frame(predictions, generated_at=generated_at)
    existing = load_prediction_ledger(ledger_path)
    if incoming.empty:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        existing.to_csv(ledger_path, index=False)
        return existing

    generated_times = pd.to_datetime(incoming["prediction_generated_at"], utc=True, errors="coerce")
    kickoff_times = pd.to_datetime(incoming["kickoff_utc"], utc=True, errors="coerce")
    valid_timing = generated_times.notna() & kickoff_times.notna() & generated_times.le(kickoff_times)
    incoming = incoming[valid_timing].copy()
    if incoming.empty:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        existing.to_csv(ledger_path, index=False)
        return existing

    if existing.empty:
        updated = incoming.drop_duplicates(subset=LEDGER_KEY_COLUMNS, keep="first")
    else:
        existing = _normalize_prediction_frame(existing)
        keys = set(map(tuple, existing[LEDGER_KEY_COLUMNS].astype(str).to_numpy()))
        incoming = incoming[
            ~incoming[LEDGER_KEY_COLUMNS].astype(str).apply(tuple, axis=1).isin(keys)
        ].copy()
        updated = pd.concat([existing, incoming], ignore_index=True)

    updated = updated.drop_duplicates(subset=LEDGER_KEY_COLUMNS, keep="first")
    updated = updated.sort_values(["kickoff_utc", "match_id", "model_key"], kind="mergesort")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(ledger_path, index=False)
    return updated


def _actual_class(row: pd.Series) -> int | None:
    winner = str(row.get("winner", "") or "")
    if winner == "HOME_TEAM":
        return 0
    if winner == "DRAW":
        return 1
    if winner == "AWAY_TEAM":
        return 2

    home = pd.to_numeric(row.get("home_score"), errors="coerce")
    away = pd.to_numeric(row.get("away_score"), errors="coerce")
    if pd.isna(home) or pd.isna(away):
        return None
    if float(home) > float(away):
        return 0
    if float(home) < float(away):
        return 2
    return 1


def _probability_for_actual(row: pd.Series, actual_class: int) -> float:
    return float(row[PROBABILITY_COLUMNS[actual_class]])


def build_match_evaluation(
    ledger: pd.DataFrame | None = None,
    matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one evaluated row per finished match/model prediction."""
    prediction_ledger = _normalize_prediction_frame(
        ledger if ledger is not None else load_prediction_ledger()
    )
    cached_matches = matches if matches is not None else load_cached_matches()
    if prediction_ledger.empty or cached_matches.empty:
        return pd.DataFrame()

    finished = cached_matches[cached_matches["status"].astype(str).isin(FINISHED_STATUSES)].copy()
    if finished.empty:
        return pd.DataFrame()

    finished["match_id"] = finished["match_id"].astype(str)
    finished["actual_class"] = finished.apply(_actual_class, axis=1)
    finished = finished.dropna(subset=["actual_class"])
    if finished.empty:
        return pd.DataFrame()

    actual_columns = [
        "match_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "winner",
        "actual_class",
    ]
    evaluation = prediction_ledger.merge(finished[actual_columns], on="match_id", how="inner")
    if evaluation.empty:
        return pd.DataFrame()

    for column in PROBABILITY_COLUMNS:
        evaluation[column] = pd.to_numeric(evaluation[column], errors="coerce").clip(1e-12, 1.0)
    total = evaluation[PROBABILITY_COLUMNS].sum(axis=1)
    evaluation = evaluation[total.gt(0)].copy()
    evaluation[PROBABILITY_COLUMNS] = evaluation[PROBABILITY_COLUMNS].div(total[total.gt(0)], axis=0)

    evaluation["actual_class"] = evaluation["actual_class"].astype(int)
    evaluation["actual_result"] = evaluation["actual_class"].map(
        {0: "team_a_win", 1: "draw", 2: "team_b_win"}
    )
    evaluation["predicted_class"] = np.argmax(evaluation[PROBABILITY_COLUMNS].to_numpy(), axis=1)
    evaluation["predicted_result"] = evaluation["predicted_class"].map(
        {0: "team_a_win", 1: "draw", 2: "team_b_win"}
    )
    evaluation["correct"] = evaluation["predicted_class"].eq(evaluation["actual_class"])
    evaluation["actual_probability"] = [
        _probability_for_actual(row, int(row["actual_class"])) for _, row in evaluation.iterrows()
    ]
    evaluation["log_loss"] = -np.log(evaluation["actual_probability"].clip(1e-12, 1.0))

    one_hot = np.zeros((len(evaluation), 3), dtype=float)
    one_hot[np.arange(len(evaluation)), evaluation["actual_class"].to_numpy()] = 1.0
    evaluation["brier"] = np.mean((one_hot - evaluation[PROBABILITY_COLUMNS].to_numpy()) ** 2, axis=1)
    evaluation["home_team"] = evaluation["home_team"].map(normalize_team_name)
    evaluation["away_team"] = evaluation["away_team"].map(normalize_team_name)
    return evaluation


def combine_ledger_with_backfill(
    ledger: pd.DataFrame,
    backfill: pd.DataFrame,
) -> pd.DataFrame:
    """Prefer locked ledger rows, filling missing match/model keys from cached backfill rows."""
    normalized_ledger = _normalize_prediction_frame(ledger)
    normalized_backfill = _normalize_prediction_frame(backfill)
    if normalized_ledger.empty:
        return normalized_backfill
    if normalized_backfill.empty:
        return normalized_ledger

    locked_keys = set(map(tuple, normalized_ledger[LEDGER_KEY_COLUMNS].astype(str).to_numpy()))
    backfill_missing = normalized_backfill[
        ~normalized_backfill[LEDGER_KEY_COLUMNS].astype(str).apply(tuple, axis=1).isin(locked_keys)
    ].copy()
    return pd.concat([normalized_ledger, backfill_missing], ignore_index=True)


def summarize_live_metrics(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Aggregate evaluated prediction rows into model-level live metrics."""
    if evaluation.empty:
        return pd.DataFrame(
            columns=[
                "model_key",
                "model",
                "matches_evaluated",
                "correct_picks",
                "accuracy",
                "log_loss",
                "brier",
                "last_updated",
            ]
        )

    grouped = (
        evaluation.groupby(["model_key", "model"], as_index=False)
        .agg(
            matches_evaluated=("match_id", "nunique"),
            correct_picks=("correct", "sum"),
            accuracy=("correct", "mean"),
            log_loss=("log_loss", "mean"),
            brier=("brier", "mean"),
        )
        .sort_values(["log_loss", "accuracy"], ascending=[True, False], kind="mergesort")
    )
    if "model_version" in evaluation.columns:
        locked_counts = (
            evaluation[evaluation["model_version"].astype(str).ne("cached_fixture_backfill")]
            .groupby(["model_key", "model"])["match_id"]
            .nunique()
            .rename("locked_matches")
            .reset_index()
        )
        backfill_counts = (
            evaluation[evaluation["model_version"].astype(str).eq("cached_fixture_backfill")]
            .groupby(["model_key", "model"])["match_id"]
            .nunique()
            .rename("backfill_matches")
            .reset_index()
        )
        grouped = grouped.merge(locked_counts, on=["model_key", "model"], how="left")
        grouped = grouped.merge(backfill_counts, on=["model_key", "model"], how="left")
        grouped[["locked_matches", "backfill_matches"]] = grouped[
            ["locked_matches", "backfill_matches"]
        ].fillna(0).astype(int)
    else:
        grouped["locked_matches"] = grouped["matches_evaluated"]
        grouped["backfill_matches"] = 0
    grouped["last_updated"] = utc_now_iso()
    return grouped


def refresh_live_metrics(
    ledger_path: Path = PREDICTION_LEDGER_PATH,
    backfill_path: Path = BACKFILL_PREDICTIONS_PATH,
    metrics_path: Path = LIVE_METRICS_PATH,
    evaluation_path: Path = MATCH_EVALUATION_PATH,
    matches: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Evaluate all preserved predictions against completed World Cup matches."""
    ledger = load_prediction_ledger(ledger_path)
    backfill = load_prediction_ledger(backfill_path)
    prediction_rows = combine_ledger_with_backfill(ledger, backfill)
    evaluation = build_match_evaluation(ledger=prediction_rows, matches=matches)
    metrics = summarize_live_metrics(evaluation)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(evaluation_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    return {
        "ledger_rows": len(ledger),
        "backfill_rows": len(backfill),
        "evaluated_rows": len(evaluation),
        "metric_rows": len(metrics),
    }

"""Command-line entry point for the World Cup predictor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.bracket import MissingThirdPlaceMappingError
from src.data_loader import clean_results, load_results
from src.draw_model import build_empirical_draw_model
from src.elo import build_elo_history, save_elo_ratings
from src.market_value import (
    apply_market_value_adjustments,
    load_historical_world_cup_market_values,
    load_market_values,
)
from src.simulate import load_groups, run_monte_carlo, simulate_tournament
from src.utils import ensure_directories


RESULTS_PATH = Path("data/raw/results.csv")
GROUPS_PATH = Path("data/fixtures/world_cup_2026_groups.csv")
MARKET_VALUES_PATH = Path("data/fixtures/team_market_values.csv")
HISTORICAL_MARKET_VALUES_PATH = Path("data/fixtures/historical_world_cup_market_values.csv")
INJURED_PLAYERS_MARKET_VALUES_PATH = Path("data/fixtures/injured_players_market_value_filled.csv")
ELO_RATINGS_PATH = Path("results/current_elo_ratings.csv")
MATCH_PREDICTIONS_PATH = Path("results/elo_match_predictions.csv")
STAGE_PROBABILITIES_PATH = Path("results/team_stage_probabilities.csv")
SAMPLE_BRACKET_PATH = Path("results/sample_bracket.json")
MODEL_RATINGS_PATH = Path("results/world_cup_team_model_ratings.csv")
ML_FEATURES_PATH = Path("data/processed/ml_match_features.csv")
STATSBOMB_TEAM_MATCH_FEATURES_PATH = Path("data/processed/statsbomb_team_match_features.csv")
STATSBOMB_ROLLING_FEATURES_PATH = Path("data/processed/statsbomb_team_rolling_features.csv")
LOGISTIC_MODEL_PATH = Path("models/logistic_match_outcome.joblib")
LOGISTIC_PREDICTIONS_PATH = Path("results/logistic_backtest_predictions.csv")
GRADIENT_BOOSTING_MODEL_PATH = Path("models/gradient_boosting_match_outcome.joblib")
GRADIENT_BOOSTING_PREDICTIONS_PATH = Path("results/gradient_boosting_backtest_predictions.csv")
BLEND_PREDICTIONS_PATH = Path("results/blend_backtest_predictions.csv")
BLEND_LOGISTIC_WEIGHT = 0.65

TRAIN_LOGISTIC_MODEL = True
TRAIN_GRADIENT_BOOSTING_MODEL = True


def _save_match_predictions(elo_history: pd.DataFrame, path: Path) -> None:
    """Save pre-match Elo predictions from the historical rating pass."""
    columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
        "home_pre_elo",
        "away_pre_elo",
        "home_recent_form_elo",
        "away_recent_form_elo",
        "home_schedule_strength",
        "away_schedule_strength",
        "home_schedule_adjustment",
        "away_schedule_adjustment",
        "home_prediction_elo",
        "away_prediction_elo",
        "elo_diff",
        "adjusted_elo_diff",
        "expected_home",
        "expected_away",
        "base_k",
        "decay_weight",
        "elo_update_multiplier",
        "effective_k",
        "base_margin_multiplier",
        "favorite_mismatch_dampener",
        "margin_multiplier",
        "shared_opponent_strength_multiplier",
        "home_opponent_strength_multiplier",
        "away_opponent_strength_multiplier",
        "home_elo_change",
        "away_elo_change",
        "home_recent_form_elo_change",
        "away_recent_form_elo_change",
    ]
    available_columns = [column for column in columns if column in elo_history.columns]
    elo_history[available_columns].to_csv(path, index=False)


def _json_default(value: object) -> object:
    """Convert numpy/pandas scalar values for JSON output."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _groups_are_complete(groups: dict[str, list[str]]) -> bool:
    """Return True when fixture data has 12 complete four-team groups."""
    expected_groups = set("ABCDEFGHIJKL")
    return set(groups) == expected_groups and all(len(teams) == 4 for teams in groups.values())


def _print_world_cup_team_elos(ratings_df: pd.DataFrame, groups_path: Path) -> dict[str, list[str]] | None:
    """Print Elo ratings only for teams listed in the World Cup fixture file."""
    if not groups_path.exists():
        print("No World Cup fixture file found; skipping World Cup team Elo printout.")
        return None

    groups = load_groups(str(groups_path))
    world_cup_teams = {team for teams in groups.values() for team in teams}
    if not world_cup_teams:
        print("No non-placeholder World Cup teams found in the fixture file.")
        return groups

    world_cup_ratings = ratings_df[ratings_df["team"].isin(world_cup_teams)].copy()
    missing_teams = sorted(world_cup_teams - set(world_cup_ratings["team"]))
    if missing_teams:
        missing_rows = pd.DataFrame({"team": missing_teams, "elo": [float("nan")] * len(missing_teams)})
        world_cup_ratings = pd.concat([world_cup_ratings, missing_rows], ignore_index=True)

    world_cup_ratings = world_cup_ratings.sort_values("elo", ascending=False, na_position="last")
    print("World Cup teams by Elo:")
    print(world_cup_ratings.to_string(index=False))
    if missing_teams:
        print("Teams missing from historical Elo ratings will use INITIAL_ELO in predictions:")
        print(", ".join(missing_teams))

    return groups


def _prepare_simulation_ratings(
    ratings: dict[str, float],
    groups: dict[str, list[str]],
) -> dict[str, float]:
    """Apply optional simulation-only rating adjustments for World Cup teams."""
    world_cup_teams = {team for teams in groups.values() for team in teams}
    market_values = load_market_values(MARKET_VALUES_PATH)
    simulation_ratings, model_ratings = apply_market_value_adjustments(
        ratings,
        market_values,
        teams=world_cup_teams,
    )
    model_ratings.to_csv(MODEL_RATINGS_PATH, index=False)

    if market_values:
        print("World Cup teams by model rating (Elo + market value adjustment):")
        print(
            model_ratings[
                [
                    "team",
                    "elo",
                    "market_value_eur",
                    "market_value_adjustment",
                    "model_rating",
                ]
            ].to_string(index=False)
        )
    else:
        print(
            f"No market values found at {MARKET_VALUES_PATH}; World Cup simulation uses Elo ratings only."
        )

    return simulation_ratings


def _load_statsbomb_features_for_ml() -> pd.DataFrame | None:
    """Load parsed StatsBomb features and save rolling pre-match diagnostics if available."""
    if not STATSBOMB_TEAM_MATCH_FEATURES_PATH.exists():
        return None

    from src.statsbomb_features import (
        build_statsbomb_rolling_feature_dataset,
        load_statsbomb_team_match_features,
        save_statsbomb_rolling_feature_dataset,
    )

    statsbomb_team_match_features = load_statsbomb_team_match_features(
        STATSBOMB_TEAM_MATCH_FEATURES_PATH
    )
    rolling_features = build_statsbomb_rolling_feature_dataset(statsbomb_team_match_features)
    save_statsbomb_rolling_feature_dataset(rolling_features, STATSBOMB_ROLLING_FEATURES_PATH)
    print(
        "StatsBomb rolling features prepared: "
        f"{len(rolling_features)} team-match rows from "
        f"{rolling_features['match_id'].nunique()} matches."
    )
    return statsbomb_team_match_features


def _run_logistic_model_training(elo_history: pd.DataFrame) -> None:
    """Build ML features, train logistic regression, and save evaluation outputs."""
    from src.evaluate import evaluate_multiclass_predictions, save_feature_importance_or_coefficients
    from src.features import build_ml_feature_dataset, save_ml_feature_dataset
    from src.injury_features import load_injury_feature_index
    from src.ml_logistic import train_logistic_regression_model

    market_values_by_year = load_historical_world_cup_market_values(HISTORICAL_MARKET_VALUES_PATH)
    injury_feature_index = load_injury_feature_index(
        INJURED_PLAYERS_MARKET_VALUES_PATH,
        market_values_by_year=market_values_by_year,
    )
    statsbomb_team_match_features = _load_statsbomb_features_for_ml()
    features_df = build_ml_feature_dataset(
        elo_history,
        market_values_by_year=market_values_by_year,
        statsbomb_team_match_features=statsbomb_team_match_features,
        injury_feature_index=injury_feature_index,
    )
    save_ml_feature_dataset(features_df, str(ML_FEATURES_PATH))

    model, test_predictions = train_logistic_regression_model(
        features_df,
        model_path=str(LOGISTIC_MODEL_PATH),
    )
    y_proba = test_predictions[["team_a_win_prob", "draw_prob", "team_b_win_prob"]].to_numpy()
    metrics = evaluate_multiclass_predictions(
        test_predictions["target"],
        y_proba,
        test_predictions["predicted_target"],
        top_probability_pred=test_predictions["top_probability_target"],
    )
    save_feature_importance_or_coefficients(model)
    test_predictions.to_csv(LOGISTIC_PREDICTIONS_PATH, index=False)

    print("Logistic regression model performance on matches from 2022-01-01 onward:")
    print(
        pd.DataFrame(
            [
                {
                    "test_matches": len(test_predictions),
                    "best_c": test_predictions.attrs.get("best_c"),
                    "best_class_weight": test_predictions.attrs.get("best_class_weight"),
                    "draw_probability_multiplier": test_predictions.attrs.get(
                        "draw_probability_multiplier"
                    ),
                    "draw_decision_threshold": test_predictions.attrs.get(
                        "draw_decision_threshold"
                    ),
                    "draw_decision_max_win_prob_gap": test_predictions.attrs.get(
                        "draw_decision_max_win_prob_gap"
                    ),
                    "flipped_training_rows": test_predictions.attrs.get("use_flipped_training_rows"),
                    "flipped_decisive_target_as_draw": test_predictions.attrs.get(
                        "flipped_decisive_target_as_draw"
                    ),
                    **metrics,
                }
            ]
        ).to_string(index=False)
    )


def _run_gradient_boosting_model_training(features_df: pd.DataFrame) -> None:
    """Train gradient boosting on prepared features and save evaluation outputs."""
    from src.evaluate import evaluate_multiclass_predictions
    from src.ml_gradient_boosting import train_gradient_boosting_model

    model, test_predictions = train_gradient_boosting_model(
        features_df,
        model_path=str(GRADIENT_BOOSTING_MODEL_PATH),
    )
    y_proba = test_predictions[["team_a_win_prob", "draw_prob", "team_b_win_prob"]].to_numpy()
    metrics = evaluate_multiclass_predictions(
        test_predictions["target"],
        y_proba,
        test_predictions["predicted_target"],
        top_probability_pred=test_predictions["top_probability_target"],
        metrics_output_path="results/gradient_boosting_model_metrics.csv",
        confusion_matrix_output_path="results/gradient_boosting_confusion_matrix.csv",
    )
    test_predictions.to_csv(GRADIENT_BOOSTING_PREDICTIONS_PATH, index=False)
    _ = model

    print("Gradient boosting model performance on matches from 2022-01-01 onward:")
    print(
        pd.DataFrame(
            [
                {
                    "test_matches": len(test_predictions),
                    "feature_profile": test_predictions.attrs.get("feature_profile"),
                    "max_iter": test_predictions.attrs.get("max_iter"),
                    "learning_rate": test_predictions.attrs.get("learning_rate"),
                    "max_leaf_nodes": test_predictions.attrs.get("max_leaf_nodes"),
                    "l2_regularization": test_predictions.attrs.get("l2_regularization"),
                    "class_weight": test_predictions.attrs.get("class_weight"),
                    "draw_probability_multiplier": test_predictions.attrs.get(
                        "draw_probability_multiplier"
                    ),
                    "calibration_temperature": test_predictions.attrs.get(
                        "calibration_temperature"
                    ),
                    "world_cup_group_shrinkage": test_predictions.attrs.get(
                        "world_cup_group_shrinkage"
                    ),
                    "world_cup_knockout_shrinkage": test_predictions.attrs.get(
                        "world_cup_knockout_shrinkage"
                    ),
                    "world_cup_rotation_strength": test_predictions.attrs.get(
                        "world_cup_rotation_strength"
                    ),
                    "draw_decision_threshold": test_predictions.attrs.get(
                        "draw_decision_threshold"
                    ),
                    "draw_decision_max_win_prob_gap": test_predictions.attrs.get(
                        "draw_decision_max_win_prob_gap"
                    ),
                    **metrics,
                }
            ]
        ).to_string(index=False)
    )


def _run_blend_evaluation(features_df: pd.DataFrame | None = None) -> None:
    """Blend logistic and gradient boosting probabilities and save evaluation outputs."""
    from src.evaluate import evaluate_multiclass_predictions

    if not LOGISTIC_PREDICTIONS_PATH.exists() or not GRADIENT_BOOSTING_PREDICTIONS_PATH.exists():
        return

    logistic = pd.read_csv(LOGISTIC_PREDICTIONS_PATH, parse_dates=["date"])
    gradient_boosting = pd.read_csv(GRADIENT_BOOSTING_PREDICTIONS_PATH, parse_dates=["date"])
    merge_columns = ["date", "team_a", "team_b", "tournament", "neutral", "target"]
    merged = logistic.merge(
        gradient_boosting,
        on=merge_columns,
        suffixes=("_logistic", "_gradient_boosting"),
        validate="one_to_one",
    )
    logistic_weight = BLEND_LOGISTIC_WEIGHT
    blend_tuning_results = pd.DataFrame()
    if features_df is not None:
        from src.blend import tune_blend_weight

        logistic_weight, blend_tuning_results = tune_blend_weight(features_df)
    gradient_boosting_weight = 1.0 - logistic_weight
    prediction_rows = merged[merge_columns].copy()
    prediction_rows["team_a_win_prob"] = (
        logistic_weight * merged["team_a_win_prob_logistic"]
        + gradient_boosting_weight * merged["team_a_win_prob_gradient_boosting"]
    )
    prediction_rows["draw_prob"] = (
        logistic_weight * merged["draw_prob_logistic"]
        + gradient_boosting_weight * merged["draw_prob_gradient_boosting"]
    )
    prediction_rows["team_b_win_prob"] = (
        logistic_weight * merged["team_b_win_prob_logistic"]
        + gradient_boosting_weight * merged["team_b_win_prob_gradient_boosting"]
    )
    y_proba = prediction_rows[["team_a_win_prob", "draw_prob", "team_b_win_prob"]].to_numpy()
    prediction_rows["top_probability_target"] = np.argmax(y_proba, axis=1)
    prediction_rows["predicted_target"] = prediction_rows["top_probability_target"]
    metrics = evaluate_multiclass_predictions(
        prediction_rows["target"],
        y_proba,
        prediction_rows["predicted_target"],
        top_probability_pred=prediction_rows["top_probability_target"],
        metrics_output_path="results/blend_model_metrics.csv",
        confusion_matrix_output_path="results/blend_confusion_matrix.csv",
    )
    prediction_rows.to_csv(BLEND_PREDICTIONS_PATH, index=False)
    print("Blend model performance on matches from 2022-01-01 onward:")
    print(
        pd.DataFrame(
            [
                {
                    "test_matches": len(prediction_rows),
                    "logistic_weight": logistic_weight,
                    "gradient_boosting_weight": gradient_boosting_weight,
                    "blend_weight_tuned": not blend_tuning_results.empty,
                    **metrics,
                }
            ]
        ).to_string(index=False)
    )


def main() -> int:
    """Run the Elo build and optional World Cup simulation."""
    ensure_directories()

    if not RESULTS_PATH.exists():
        print(f"Missing {RESULTS_PATH}. Add historical results before running the model.")
        return 1

    raw_results = load_results(str(RESULTS_PATH))
    matches = clean_results(raw_results)
    if matches.empty:
        print("No usable matches found after cleaning and applying MIN_MATCH_YEAR.")
        return 1
    elo_history, ratings = build_elo_history(matches)

    save_elo_ratings(ratings, str(ELO_RATINGS_PATH))
    _save_match_predictions(elo_history, MATCH_PREDICTIONS_PATH)

    features_df = None
    if TRAIN_LOGISTIC_MODEL or TRAIN_GRADIENT_BOOSTING_MODEL:
        from src.features import build_ml_feature_dataset, save_ml_feature_dataset
        from src.injury_features import load_injury_feature_index

        market_values_by_year = load_historical_world_cup_market_values(HISTORICAL_MARKET_VALUES_PATH)
        injury_feature_index = load_injury_feature_index(
            INJURED_PLAYERS_MARKET_VALUES_PATH,
            market_values_by_year=market_values_by_year,
        )
        if injury_feature_index:
            print(f"Injury features prepared for {len(injury_feature_index)} World Cup team-year rows.")
        else:
            print(f"No injury market value file found at {INJURED_PLAYERS_MARKET_VALUES_PATH}; using neutral injury features.")
        statsbomb_team_match_features = _load_statsbomb_features_for_ml()
        features_df = build_ml_feature_dataset(
            elo_history,
            market_values_by_year=market_values_by_year,
            statsbomb_team_match_features=statsbomb_team_match_features,
            injury_feature_index=injury_feature_index,
        )
        save_ml_feature_dataset(features_df, str(ML_FEATURES_PATH))

    if TRAIN_LOGISTIC_MODEL:
        try:
            if features_df is None:
                raise ValueError("No feature dataset was prepared for logistic regression.")
            from src.evaluate import evaluate_multiclass_predictions, save_feature_importance_or_coefficients
            from src.ml_logistic import train_logistic_regression_model

            model, test_predictions = train_logistic_regression_model(
                features_df,
                model_path=str(LOGISTIC_MODEL_PATH),
            )
            y_proba = test_predictions[["team_a_win_prob", "draw_prob", "team_b_win_prob"]].to_numpy()
            metrics = evaluate_multiclass_predictions(
                test_predictions["target"],
                y_proba,
                test_predictions["predicted_target"],
                top_probability_pred=test_predictions["top_probability_target"],
            )
            save_feature_importance_or_coefficients(model)
            test_predictions.to_csv(LOGISTIC_PREDICTIONS_PATH, index=False)

            print("Logistic regression model performance on matches from 2022-01-01 onward:")
            print(
                pd.DataFrame(
                    [
                        {
                            "test_matches": len(test_predictions),
                            "best_c": test_predictions.attrs.get("best_c"),
                            "best_class_weight": test_predictions.attrs.get("best_class_weight"),
                            "draw_probability_multiplier": test_predictions.attrs.get(
                                "draw_probability_multiplier"
                            ),
                            "world_cup_group_shrinkage": test_predictions.attrs.get(
                                "world_cup_group_shrinkage"
                            ),
                            "world_cup_knockout_shrinkage": test_predictions.attrs.get(
                                "world_cup_knockout_shrinkage"
                            ),
                            "world_cup_rotation_strength": test_predictions.attrs.get(
                                "world_cup_rotation_strength"
                            ),
                            "draw_decision_threshold": test_predictions.attrs.get(
                                "draw_decision_threshold"
                            ),
                            "draw_decision_max_win_prob_gap": test_predictions.attrs.get(
                                "draw_decision_max_win_prob_gap"
                            ),
                            "flipped_training_rows": test_predictions.attrs.get("use_flipped_training_rows"),
                            "flipped_decisive_target_as_draw": test_predictions.attrs.get(
                                "flipped_decisive_target_as_draw"
                            ),
                            **metrics,
                        }
                    ]
                ).to_string(index=False)
            )
        except (ImportError, ValueError) as exc:
            print(f"Skipping logistic regression model training: {exc}")
    if TRAIN_GRADIENT_BOOSTING_MODEL:
        try:
            if features_df is None:
                raise ValueError("No feature dataset was prepared for gradient boosting.")
            _run_gradient_boosting_model_training(features_df)
        except (ImportError, ValueError) as exc:
            print(f"Skipping gradient boosting model training: {exc}")
    if TRAIN_LOGISTIC_MODEL and TRAIN_GRADIENT_BOOSTING_MODEL:
        try:
            _run_blend_evaluation(features_df)
        except (ImportError, ValueError) as exc:
            print(f"Skipping blend model evaluation: {exc}")

    ratings_df = pd.read_csv(ELO_RATINGS_PATH)
    groups = _print_world_cup_team_elos(ratings_df, GROUPS_PATH)

    if not GROUPS_PATH.exists():
        print(f"No fixture file found at {GROUPS_PATH}; skipping World Cup simulation.")
        return 0

    if groups is None:
        groups = load_groups(str(GROUPS_PATH))
    if not _groups_are_complete(groups):
        print(
            "Elo ratings were created successfully, but the World Cup simulation was skipped because "
            f"{GROUPS_PATH} does not contain 12 complete groups of 4 non-placeholder teams."
        )
        return 0

    simulation_ratings = _prepare_simulation_ratings(ratings, groups)

    try:
        draw_model = build_empirical_draw_model(elo_history)
        probabilities = run_monte_carlo(
            groups,
            simulation_ratings,
            n_simulations=1000,
            seed=config.MONTE_CARLO_SEED,
            draw_model=draw_model,
        )
        probabilities.to_csv(STAGE_PROBABILITIES_PATH, index=False)

        sample_rng = np.random.default_rng(config.SAMPLE_BRACKET_SEED)
        sample_bracket = simulate_tournament(groups, simulation_ratings, sample_rng, draw_model=draw_model)
        with SAMPLE_BRACKET_PATH.open("w", encoding="utf-8") as f:
            json.dump(sample_bracket, f, indent=2, default=_json_default)

        print("Top 10 teams by World Cup champion probability:")
        print(probabilities.head(10)[["team", "champion_prob"]].to_string(index=False))
    except MissingThirdPlaceMappingError as exc:
        print(
            "Elo ratings were created successfully, but the World Cup simulation needs the missing "
            f"third-place bracket mapping to be added first: {exc}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

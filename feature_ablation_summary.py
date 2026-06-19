"""Summarize model feature-profile and post-processing ablation results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


GB_TUNING_PATH = Path("results/gradient_boosting_tuning_results.csv")
GB_FOLD_PATH = Path("results/gradient_boosting_rolling_validation_results.csv")
POSTPROCESS_PATH = Path("results/postprocess_ablation_summary.csv")
OUTPUT_PATH = Path("results/feature_ablation_summary.csv")


def _summarize_gradient_boosting_profiles() -> pd.DataFrame:
    if not GB_TUNING_PATH.exists():
        return pd.DataFrame()

    tuning = pd.read_csv(GB_TUNING_PATH)
    required = {"feature_profile", "validation_log_loss", "validation_accuracy"}
    if not required.issubset(tuning.columns):
        return pd.DataFrame()

    rows = []
    for profile, group in tuning.groupby("feature_profile"):
        best = group.sort_values(["validation_log_loss", "validation_accuracy"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "ablation_type": "gradient_boosting_feature_profile",
                "experiment": profile,
                "validation_log_loss": float(best["validation_log_loss"]),
                "validation_accuracy": float(best["validation_accuracy"]),
                "selected": bool(best.get("selected", False)),
                "notes": "Rolling validation best row for this feature profile.",
            }
        )
    return pd.DataFrame(rows)


def _summarize_postprocess() -> pd.DataFrame:
    if not POSTPROCESS_PATH.exists():
        return pd.DataFrame()

    postprocess = pd.read_csv(POSTPROCESS_PATH)
    required = {
        "experiment",
        "log_loss_2022_plus",
        "accuracy_2022_plus",
        "log_loss_2022_world_cup",
        "accuracy_2022_world_cup",
    }
    if not required.issubset(postprocess.columns):
        return pd.DataFrame()

    rows = []
    for _, row in postprocess.iterrows():
        rows.append(
            {
                "ablation_type": "postprocess_experiment",
                "experiment": row["experiment"],
                "log_loss_2022_plus": row["log_loss_2022_plus"],
                "accuracy_2022_plus": row["accuracy_2022_plus"],
                "log_loss_2022_world_cup": row["log_loss_2022_world_cup"],
                "accuracy_2022_world_cup": row["accuracy_2022_world_cup"],
                "delta_log_loss_2022_wc_vs_temp": row.get("delta_log_loss_2022_wc_vs_temp"),
                "delta_accuracy_2022_wc_vs_temp": row.get("delta_accuracy_2022_wc_vs_temp"),
                "selected": False,
                "notes": "Post-processing or calibration experiment.",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    summaries = [
        summary
        for summary in [
            _summarize_gradient_boosting_profiles(),
            _summarize_postprocess(),
        ]
        if not summary.empty
    ]
    if not summaries:
        print("No ablation inputs found. Run main.py and postprocess_experiments.py first.")
        return 1

    output = pd.concat(summaries, ignore_index=True, sort=False)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

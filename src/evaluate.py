"""Evaluation helpers for supervised match-outcome models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

from src.ml_logistic import CLASS_LABELS, CLASS_NAMES


def evaluate_multiclass_predictions(
    y_true,
    y_proba,
    y_pred,
    top_probability_pred=None,
    metrics_output_path: str = "results/logistic_model_metrics.csv",
    confusion_matrix_output_path: str = "results/logistic_confusion_matrix.csv",
) -> dict[str, float]:
    """Evaluate multiclass match predictions and save summary CSV outputs."""
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    y_proba_array = np.asarray(y_proba, dtype=float)
    top_probability_pred_array = (
        np.asarray(top_probability_pred, dtype=int)
        if top_probability_pred is not None
        else np.array(CLASS_LABELS, dtype=int)[np.argmax(y_proba_array, axis=1)]
    )
    if y_proba_array.ndim != 2 or y_proba_array.shape[1] != len(CLASS_LABELS):
        raise ValueError("y_proba must have shape (n_samples, 3) for classes [0, 1, 2].")

    one_hot = np.zeros_like(y_proba_array, dtype=float)
    for row_index, class_label in enumerate(y_true_array):
        if int(class_label) in CLASS_LABELS:
            one_hot[row_index, CLASS_LABELS.index(int(class_label))] = 1.0

    actual_draws = y_true_array == 1
    predicted_draws = y_pred_array == 1
    true_predicted_draws = actual_draws & predicted_draws
    non_draw_actual = ~actual_draws
    metrics = {
        "top_probability_accuracy": float(accuracy_score(y_true_array, top_probability_pred_array)),
        "decision_accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "log_loss": float(log_loss(y_true_array, y_proba_array, labels=CLASS_LABELS)),
        "multiclass_brier_score": float(np.mean((one_hot - y_proba_array) ** 2)),
        "top_probability_predicted_draws": float((top_probability_pred_array == 1).sum()),
        "predicted_draws": float(predicted_draws.sum()),
        "actual_draws": float(actual_draws.sum()),
        "draw_recall": (
            float(true_predicted_draws.sum() / actual_draws.sum())
            if actual_draws.sum() > 0
            else 0.0
        ),
        "draw_precision": (
            float(true_predicted_draws.sum() / predicted_draws.sum())
            if predicted_draws.sum() > 0
            else 0.0
        ),
        "non_draw_accuracy": (
            float((y_pred_array[non_draw_actual] == y_true_array[non_draw_actual]).sum() / non_draw_actual.sum())
            if non_draw_actual.sum() > 0
            else 0.0
        ),
    }
    metrics["accuracy"] = metrics["decision_accuracy"]

    metrics_output = Path(metrics_output_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_output, index=False)

    matrix = confusion_matrix(y_true_array, y_pred_array, labels=CLASS_LABELS)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{CLASS_NAMES[label]}" for label in CLASS_LABELS],
        columns=[f"predicted_{CLASS_NAMES[label]}" for label in CLASS_LABELS],
    )
    matrix_output = Path(confusion_matrix_output_path)
    matrix_output.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(matrix_output)
    return metrics


def save_feature_importance_or_coefficients(
    model,
    output_path: str = "results/logistic_coefficients.csv",
) -> None:
    """Save logistic regression coefficients with transformed feature names."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"feature_{index}" for index in range(classifier.coef_.shape[1])]

    rows = []
    for class_index, class_label in enumerate(classifier.classes_):
        class_name = CLASS_NAMES.get(int(class_label), str(class_label))
        for feature_name, coefficient in zip(feature_names, classifier.coef_[class_index], strict=False):
            rows.append(
                {
                    "class_label": int(class_label),
                    "class_name": class_name,
                    "feature": str(feature_name).replace("numeric__", "").replace("categorical__", ""),
                    "coefficient": float(coefficient),
                }
            )

    pd.DataFrame(rows).to_csv(output, index=False)

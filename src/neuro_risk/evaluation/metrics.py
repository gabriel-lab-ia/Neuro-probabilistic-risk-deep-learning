from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score


def entropy_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    safe_probabilities = np.clip(probabilities, 1.0e-8, 1.0)
    return -np.sum(safe_probabilities * np.log(safe_probabilities), axis=1)


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_bins: int = 15,
) -> tuple[float, list[dict[str, float]]]:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == labels).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    ece = 0.0
    bins: list[dict[str, float]] = []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "accuracy": 0.0,
                    "confidence": 0.0,
                    "count": 0.0,
                }
            )
            continue
        bin_accuracy = float(correctness[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        proportion = float(mask.mean())
        ece += abs(bin_accuracy - bin_confidence) * proportion
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "accuracy": bin_accuracy,
                "confidence": bin_confidence,
                "count": float(mask.sum()),
            }
        )
    return float(ece), bins


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    label_names: tuple[str, ...],
    calibration_bins: int = 15,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    ece, bin_stats = expected_calibration_error(labels, probabilities, num_bins=calibration_bins)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
        "nll": float(log_loss(labels, probabilities, labels=list(range(len(label_names))))),
        "ece": float(ece),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(len(label_names)))).tolist(),
        "reliability_bins": bin_stats,
    }

    try:
        metrics["auroc_ovr"] = float(
            roc_auc_score(labels, probabilities, multi_class="ovr", labels=list(range(len(label_names))))
        )
    except ValueError:
        metrics["auroc_ovr"] = None

    metrics["mean_confidence"] = float(probabilities.max(axis=1).mean())
    metrics["mean_entropy"] = float(entropy_from_probabilities(probabilities).mean())
    return metrics

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from neuro_risk.inference import MCDropoutResult
from neuro_risk.utils.io import write_json

sns.set_theme(style="whitegrid")


def save_confusion_matrix(confusion: np.ndarray, label_names: tuple[str, ...], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=axis,
    )
    axis.set_title("Confusion matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_logit_distribution(logits: np.ndarray, labels: np.ndarray, label_names: tuple[str, ...], output_path: Path) -> None:
    records: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(label_names):
        for value, label in zip(logits[:, class_index], labels, strict=True):
            records.append(
                {
                    "logit": float(value),
                    "true_label": label_names[int(label)],
                    "logit_class": class_name,
                }
            )
    frame = pd.DataFrame.from_records(records)
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=frame, x="logit_class", y="logit", hue="true_label", ax=axis)
    axis.set_title("Logit distribution by output neuron")
    axis.set_xlabel("Logit dimension")
    axis.set_ylabel("Logit value")
    axis.legend(title="True label", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_probability_heatmap(probabilities: np.ndarray, label_names: tuple[str, ...], output_path: Path, title: str) -> None:
    max_rows = min(40, probabilities.shape[0])
    ordered = np.argsort(probabilities.max(axis=1))
    panel = probabilities[ordered[:max_rows]]
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(panel, cmap="mako", xticklabels=label_names, yticklabels=False, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Class")
    axis.set_ylabel("Low-confidence samples")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_uncertainty_panels(result: MCDropoutResult, label_names: tuple[str, ...], output_path: Path) -> None:
    ordered = np.argsort(result.predictive_entropy)[-40:]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    sns.heatmap(result.mean_probabilities[ordered], cmap="crest", xticklabels=label_names, yticklabels=False, ax=axes[0, 0])
    axes[0, 0].set_title("MC mean predictive probability")

    sns.heatmap(result.class_variances[ordered], cmap="flare", xticklabels=label_names, yticklabels=False, ax=axes[0, 1])
    axes[0, 1].set_title("MC predictive variance")

    sns.histplot(result.predictive_entropy, bins=20, color="#145374", ax=axes[1, 0])
    axes[1, 0].set_title("Predictive entropy")
    axes[1, 0].set_xlabel("Entropy")

    sns.histplot(result.mutual_information, bins=20, color="#b85c38", ax=axes[1, 1])
    axes[1, 1].set_title("Mutual information")
    axes[1, 1].set_xlabel("Mutual information")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_confidence_distribution(confidence: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.histplot(confidence, bins=20, color="#386641", ax=axis)
    axis.set_title("Confidence distribution")
    axis.set_xlabel("Max calibrated probability")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_reliability_diagram(
    before_bins: list[dict[str, float]],
    after_bins: list[dict[str, float]],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    centers_before = [0.5 * (item["lower"] + item["upper"]) for item in before_bins]
    centers_after = [0.5 * (item["lower"] + item["upper"]) for item in after_bins]
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    axis.plot(centers_before, [item["accuracy"] for item in before_bins], marker="o", label="Before")
    axis.plot(centers_after, [item["accuracy"] for item in after_bins], marker="o", label="After")
    axis.set_title("Reliability diagram")
    axis.set_xlabel("Confidence bin center")
    axis.set_ylabel("Empirical accuracy")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_interactive_uncertainty_scatter(
    mean_probabilities: np.ndarray,
    entropy: np.ndarray,
    mutual_information: np.ndarray,
    labels: np.ndarray,
    predicted_labels: np.ndarray,
    label_names: tuple[str, ...],
    output_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "confidence": mean_probabilities.max(axis=1),
            "entropy": entropy,
            "mutual_information": mutual_information,
            "true_label": [label_names[int(label)] for label in labels],
            "predicted_label": [label_names[int(label)] for label in predicted_labels],
        }
    )
    figure = px.scatter(
        frame,
        x="confidence",
        y="entropy",
        size="mutual_information",
        color="predicted_label",
        hover_data=["true_label", "predicted_label"],
        title="Calibrated confidence vs predictive entropy",
    )
    figure.write_html(output_path, include_plotlyjs="cdn")


def build_jsviz_payload(
    report: dict[str, Any],
    mc_result: MCDropoutResult,
    label_names: tuple[str, ...],
    output_path: Path,
    architecture: dict[str, Any] | None = None,
) -> None:
    most_uncertain = np.argsort(mc_result.predictive_entropy)[-8:][::-1]
    calibrated_metrics = report["metrics"].get("calibrated_test", report["metrics"].get("mc_dropout_test", {}))
    raw_metrics = report["metrics"].get("raw_test", calibrated_metrics)
    mc_metrics = report["metrics"].get("mc_dropout_test", calibrated_metrics)
    payload = {
        "project": report["project"],
        "temperature": report["temperature"],
        "metrics": report["metrics"],
        "history": report.get("history", {}),
        "best_epoch": report.get("best_epoch"),
        "device": report.get("device", "unknown"),
        "cautions": report.get("cautions", []),
        "label_names": list(label_names),
        "class_mean_probabilities": mc_result.mean_probabilities.mean(axis=0).tolist(),
        "class_mean_variances": mc_result.class_variances.mean(axis=0).tolist(),
        "class_confidence_profile": {
            label_names[class_index]: {
                "mean_probability": float(mc_result.mean_probabilities[:, class_index].mean()),
                "mean_variance": float(mc_result.class_variances[:, class_index].mean()),
            }
            for class_index in range(len(label_names))
        },
        "chart_data": {
            "comparison_metrics": {
                "labels": ["accuracy", "f1_macro", "auroc_ovr", "mean_confidence"],
                "raw": [
                    float(raw_metrics.get("accuracy", 0.0)),
                    float(raw_metrics.get("f1_macro", 0.0)),
                    float(raw_metrics.get("auroc_ovr", 0.0) or 0.0),
                    float(raw_metrics.get("mean_confidence", 0.0)),
                ],
                "calibrated": [
                    float(calibrated_metrics.get("accuracy", 0.0)),
                    float(calibrated_metrics.get("f1_macro", 0.0)),
                    float(calibrated_metrics.get("auroc_ovr", 0.0) or 0.0),
                    float(calibrated_metrics.get("mean_confidence", 0.0)),
                ],
                "mc_dropout": [
                    float(mc_metrics.get("accuracy", 0.0)),
                    float(mc_metrics.get("f1_macro", 0.0)),
                    float(mc_metrics.get("auroc_ovr", 0.0) or 0.0),
                    float(mc_metrics.get("mean_confidence", 0.0)),
                ],
            },
            "calibration_curves": {
                "raw": raw_metrics.get("reliability_bins", []),
                "calibrated": calibrated_metrics.get("reliability_bins", []),
                "mc_dropout": mc_metrics.get("reliability_bins", []),
            },
            "confusion_matrices": {
                "raw": raw_metrics.get("confusion_matrix", []),
                "calibrated": calibrated_metrics.get("confusion_matrix", []),
                "mc_dropout": mc_metrics.get("confusion_matrix", []),
            },
            "uncertainty_scatter": [
                {
                    "sample_index": int(index),
                    "confidence": float(mc_result.confidence[index]),
                    "entropy": float(mc_result.predictive_entropy[index]),
                    "mutual_information": float(mc_result.mutual_information[index]),
                    "predicted_label": label_names[int(mc_result.predicted_labels[index])],
                    "true_label": label_names[int(mc_result.labels[index])],
                }
                for index in range(mc_result.mean_probabilities.shape[0])
            ],
        },
        "architecture": architecture or {},
        "uncertain_examples": [
            {
                "sample_index": int(index),
                "predicted_label": label_names[int(mc_result.predicted_labels[index])],
                "true_label": label_names[int(mc_result.labels[index])],
                "confidence": float(mc_result.confidence[index]),
                "predictive_entropy": float(mc_result.predictive_entropy[index]),
                "mutual_information": float(mc_result.mutual_information[index]),
                "mean_probabilities": mc_result.mean_probabilities[index].tolist(),
            }
            for index in most_uncertain
        ],
    }
    write_json(output_path, payload)

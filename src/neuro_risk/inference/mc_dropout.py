from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from neuro_risk.evaluation.metrics import entropy_from_probabilities
from neuro_risk.training.calibration import apply_temperature


@dataclass(slots=True)
class MCDropoutResult:
    logits_samples: np.ndarray
    probability_samples: np.ndarray
    mean_probabilities: np.ndarray
    class_variances: np.ndarray
    predictive_entropy: np.ndarray
    expected_entropy: np.ndarray
    mutual_information: np.ndarray
    confidence: np.ndarray
    predicted_labels: np.ndarray
    labels: np.ndarray


def _enable_dropout_layers(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def _collect_single_pass(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    temperature: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits_list: list[np.ndarray] = []
    probabilities_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.no_grad():
        for tabular, temporal, labels in loader:
            tabular = tabular.to(device)
            temporal = temporal.to(device)
            labels = labels.to(device)
            logits = model(tabular, temporal)
            if temperature is not None:
                logits = apply_temperature(logits, temperature)
            probabilities = torch.softmax(logits, dim=1)
            logits_list.append(logits.cpu().numpy())
            probabilities_list.append(probabilities.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return (
        np.concatenate(logits_list, axis=0),
        np.concatenate(probabilities_list, axis=0),
        np.concatenate(labels_list, axis=0),
    )


def mc_dropout_predict(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    num_samples: int,
    temperature: float | None = None,
) -> MCDropoutResult:
    """Keep dropout active at inference time to sample stochastic subnetworks.

    This approximates a posterior predictive distribution: each forward pass uses
    a different dropout mask, so averaging the resulting probabilities yields a
    calibrated uncertainty estimate that a single deterministic pass cannot expose.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive for MC Dropout, got {num_samples}.")

    was_training = model.training
    _enable_dropout_layers(model)
    logits_samples: list[np.ndarray] = []
    probability_samples: list[np.ndarray] = []
    labels = np.empty((0,), dtype=np.int64)

    for sample_index in range(num_samples):
        logits, probabilities, current_labels = _collect_single_pass(model, loader, device, temperature=temperature)
        logits_samples.append(logits)
        probability_samples.append(probabilities)
        if sample_index == 0:
            labels = current_labels

    logits_stack = np.stack(logits_samples, axis=0)
    probability_stack = np.stack(probability_samples, axis=0)
    mean_probabilities = probability_stack.mean(axis=0)
    class_variances = probability_stack.var(axis=0)
    predictive_entropy = entropy_from_probabilities(mean_probabilities)
    expected_entropy = entropy_from_probabilities(probability_stack.reshape(-1, probability_stack.shape[-1])).reshape(
        num_samples,
        -1,
    ).mean(axis=0)
    mutual_information = predictive_entropy - expected_entropy
    confidence = mean_probabilities.max(axis=1)
    predicted_labels = mean_probabilities.argmax(axis=1)

    model.train(was_training)
    return MCDropoutResult(
        logits_samples=logits_stack,
        probability_samples=probability_stack,
        mean_probabilities=mean_probabilities,
        class_variances=class_variances,
        predictive_entropy=predictive_entropy,
        expected_entropy=expected_entropy,
        mutual_information=mutual_information,
        confidence=confidence,
        predicted_labels=predicted_labels,
        labels=labels,
    )

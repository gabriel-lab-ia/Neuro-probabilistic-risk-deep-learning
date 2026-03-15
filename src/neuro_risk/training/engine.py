from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from neuro_risk.config import TrainConfig
from neuro_risk.evaluation.metrics import classification_metrics


@dataclass(slots=True)
class TrainingArtifacts:
    history: dict[str, list[float]]
    best_epoch: int
    best_state_dict: dict[str, Any]


def _move_batch_to_device(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tabular, temporal, labels = batch
    return tabular.to(device), temporal.to(device), labels.to(device)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    logits_list: list[np.ndarray] = []
    probabilities_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            tabular, temporal, labels = _move_batch_to_device(batch, device)
            logits = model(tabular, temporal)
            probabilities = torch.softmax(logits, dim=1)
            logits_list.append(logits.cpu().numpy())
            probabilities_list.append(probabilities.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return {
        "logits": np.concatenate(logits_list, axis=0),
        "probabilities": np.concatenate(probabilities_list, axis=0),
        "labels": np.concatenate(labels_list, axis=0),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    device: torch.device,
    train_config: TrainConfig,
    label_names: tuple[str, ...],
) -> TrainingArtifacts:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "validation_f1_macro": [],
    }

    best_validation_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(train_config.epochs):
        model.train()
        batch_losses: list[float] = []
        for batch in train_loader:
            tabular, temporal, labels = _move_batch_to_device(batch, device)
            optimizer.zero_grad()
            logits = model(tabular, temporal)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        validation_outputs = collect_predictions(model, validation_loader, device)
        validation_logits = torch.from_numpy(validation_outputs["logits"])
        validation_labels = torch.from_numpy(validation_outputs["labels"])
        validation_loss = float(criterion(validation_logits, validation_labels).item())
        validation_metrics = classification_metrics(
            validation_outputs["labels"],
            validation_outputs["probabilities"],
            label_names=label_names,
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(float(validation_metrics["accuracy"]))
        history["validation_f1_macro"].append(float(validation_metrics["f1_macro"]))

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= train_config.patience:
            break

    model.load_state_dict(best_state_dict)
    return TrainingArtifacts(
        history=history,
        best_epoch=best_epoch,
        best_state_dict=best_state_dict,
    )

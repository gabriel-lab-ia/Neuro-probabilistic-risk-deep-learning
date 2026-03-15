from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuro_risk.config import ProjectConfig
from neuro_risk.data import build_data_loaders, generate_synthetic_dataset, save_dataset_bundle
from neuro_risk.evaluation import classification_metrics
from neuro_risk.inference import mc_dropout_predict
from neuro_risk.models import NeuroRiskClassifier
from neuro_risk.training import TemperatureScaler, apply_temperature, collect_predictions, train_model
from neuro_risk.utils import ensure_directory, resolve_device, seed_everything, write_json, write_runtime_manifest
from neuro_risk.viz import (
    build_jsviz_payload,
    save_confidence_distribution,
    save_confusion_matrix,
    save_interactive_uncertainty_scatter,
    save_logit_distribution,
    save_probability_heatmap,
    save_reliability_diagram,
    save_uncertainty_panels,
)


@dataclass(slots=True)
class WorkflowArtifacts:
    checkpoint_path: Path
    report_path: Path
    figures_dir: Path
    interactive_plot_path: Path
    jsviz_payload_path: Path


def _tensor_from_numpy(array: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device)


def run_research_prototype(config: ProjectConfig) -> WorkflowArtifacts:
    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)

    dataset_bundle = generate_synthetic_dataset(config)
    save_dataset_bundle(dataset_bundle, config.dataset_dir)
    data_loaders, scaler = build_data_loaders(
        dataset_bundle,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    model = NeuroRiskClassifier(
        num_tabular_features=config.data.num_tabular_features,
        temporal_channels=config.data.temporal_channels,
        num_classes=config.data.num_classes,
        config=config.model,
    ).to(device)

    training_artifacts = train_model(
        model,
        train_loader=data_loaders["train"],
        validation_loader=data_loaders["validation"],
        device=device,
        train_config=config.training,
        label_names=config.label_names,
    )

    validation_outputs = collect_predictions(model, data_loaders["validation"], device)
    test_outputs = collect_predictions(model, data_loaders["test"], device)
    temperature_scaler = TemperatureScaler().to(device)
    calibration = temperature_scaler.fit(
        _tensor_from_numpy(validation_outputs["logits"], device),
        _tensor_from_numpy(validation_outputs["labels"], device, dtype=torch.long),
        max_iter=config.training.temperature_max_iter,
    )
    learned_temperature = calibration.temperature

    calibrated_probabilities = torch.softmax(
        apply_temperature(torch.from_numpy(test_outputs["logits"]), learned_temperature),
        dim=1,
    ).numpy()

    raw_metrics = classification_metrics(
        test_outputs["labels"],
        test_outputs["probabilities"],
        label_names=config.label_names,
    )
    calibrated_metrics = classification_metrics(
        test_outputs["labels"],
        calibrated_probabilities,
        label_names=config.label_names,
    )

    mc_result = mc_dropout_predict(
        model=model,
        loader=data_loaders["test"],
        device=device,
        num_samples=config.training.mc_samples,
        temperature=learned_temperature,
    )
    mc_metrics = classification_metrics(
        mc_result.labels,
        mc_result.mean_probabilities,
        label_names=config.label_names,
    )

    figures_dir = ensure_directory(config.output_dir / "figures")
    save_confusion_matrix(
        np.asarray(calibrated_metrics["confusion_matrix"], dtype=np.int64),
        config.label_names,
        figures_dir / "confusion_matrix.png",
    )
    save_logit_distribution(
        logits=test_outputs["logits"],
        labels=test_outputs["labels"],
        label_names=config.label_names,
        output_path=figures_dir / "logit_distribution.png",
    )
    save_probability_heatmap(
        probabilities=calibrated_probabilities,
        label_names=config.label_names,
        output_path=figures_dir / "softmax_probability_heatmap.png",
        title="Calibrated softmax probabilities",
    )
    save_uncertainty_panels(
        result=mc_result,
        label_names=config.label_names,
        output_path=figures_dir / "mc_uncertainty_panels.png",
    )
    save_confidence_distribution(
        confidence=mc_result.confidence,
        output_path=figures_dir / "confidence_distribution.png",
    )
    save_reliability_diagram(
        before_bins=raw_metrics["reliability_bins"],
        after_bins=calibrated_metrics["reliability_bins"],
        output_path=figures_dir / "reliability_before_after.png",
    )
    interactive_plot_path = config.output_dir / "interactive_uncertainty.html"
    save_interactive_uncertainty_scatter(
        mean_probabilities=mc_result.mean_probabilities,
        entropy=mc_result.predictive_entropy,
        mutual_information=mc_result.mutual_information,
        labels=mc_result.labels,
        predicted_labels=mc_result.predicted_labels,
        label_names=config.label_names,
        output_path=interactive_plot_path,
    )

    ensure_directory(config.model_dir)
    checkpoint_path = config.model_dir / "neuro_risk_mvp.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "temperature": learned_temperature,
            "label_names": config.label_names,
            "scaler": scaler.to_dict(),
            "best_epoch": training_artifacts.best_epoch,
            "history": training_artifacts.history,
        },
        checkpoint_path,
    )

    metrics_payload = {
        "raw_test": raw_metrics,
        "calibrated_test": calibrated_metrics,
        "mc_dropout_test": mc_metrics,
    }
    report = {
        "project": config.project_name,
        "temperature": learned_temperature,
        "temperature_nll_before": calibration.nll_before,
        "temperature_nll_after": calibration.nll_after,
        "metrics": metrics_payload,
        "best_epoch": training_artifacts.best_epoch,
        "history": training_artifacts.history,
        "device": str(device),
        "cautions": [
            "Research prototype only; not a medical diagnosis system.",
            "Softmax confidence should not be interpreted as clinical certainty.",
            "Model uncertainty should always be shown alongside class probabilities.",
        ],
    }
    report_path = config.output_dir / "report.json"
    write_json(report_path, report)
    architecture = {
        "tabular_input_dim": config.data.num_tabular_features,
        "temporal_input_shape": [config.data.temporal_channels, config.data.temporal_length],
        "tabular_hidden_dims": list(config.model.tabular_hidden_dims),
        "temporal_channels": list(config.model.temporal_channels),
        "temporal_embedding_dim": config.model.temporal_embedding_dim,
        "fusion_hidden_dim": config.model.fusion_hidden_dim,
        "num_classes": config.data.num_classes,
        "dropout": config.model.dropout,
        "stages": [
            {"name": "tabular-input", "nodes": config.data.num_tabular_features, "group": "input"},
            {"name": "tabular-mlp", "nodes": config.model.tabular_hidden_dims[0], "group": "tabular"},
            {"name": "temporal-conv", "nodes": config.model.temporal_channels[-1], "group": "temporal"},
            {"name": "fusion", "nodes": config.model.fusion_hidden_dim, "group": "fusion"},
            {"name": "classifier", "nodes": config.data.num_classes, "group": "output"},
        ],
    }
    build_jsviz_payload(
        report,
        mc_result,
        config.label_names,
        config.jsviz_payload_path,
        architecture=architecture,
    )
    write_runtime_manifest(
        config.output_dir / "runtime_manifest.json",
        packages=[
            "torch",
            "numpy",
            "pandas",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "plotly",
            "jupyterlab",
        ],
    )

    return WorkflowArtifacts(
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        figures_dir=figures_dir,
        interactive_plot_path=interactive_plot_path,
        jsviz_payload_path=config.jsviz_payload_path,
    )

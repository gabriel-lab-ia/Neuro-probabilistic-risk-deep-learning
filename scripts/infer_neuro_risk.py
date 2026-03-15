#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _set_local_runtime_env() -> None:
    local_dirs = {
        "JUPYTER_CONFIG_DIR": PROJECT_ROOT / ".jupyter",
        "JUPYTER_DATA_DIR": PROJECT_ROOT / ".jupyter" / "data",
        "JUPYTER_RUNTIME_DIR": PROJECT_ROOT / ".jupyter" / "runtime",
        "IPYTHONDIR": PROJECT_ROOT / ".ipython",
        "MPLCONFIGDIR": PROJECT_ROOT / ".cache" / "matplotlib",
        "PIP_CACHE_DIR": PROJECT_ROOT / ".cache" / "pip",
        "PYTHONPYCACHEPREFIX": PROJECT_ROOT / ".cache" / "pycache",
    }
    for key, value in local_dirs.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run calibrated MC dropout inference from a saved checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "neuro_risk" / "neuro_risk_mvp.pt",
        help="Checkpoint path created by scripts/run_neuro_risk_mvp.py.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "neuro_risk_placeholder",
        help="Directory containing train/validation/test split .npz files.",
    )
    parser.add_argument("--mc-samples", type=int, default=32, help="Number of stochastic forward passes.")
    parser.add_argument("--device", type=str, default="auto", help="Device override, for example cpu.")
    return parser.parse_args()


def main() -> None:
    _set_local_runtime_env()
    args = parse_args()

    from neuro_risk.config import ModelConfig
    from neuro_risk.data import FeatureScaler, MultimodalRiskDataset, load_dataset_bundle
    from neuro_risk.evaluation import classification_metrics
    from neuro_risk.inference import mc_dropout_predict
    from neuro_risk.models import NeuroRiskClassifier
    from neuro_risk.utils import ensure_directory, resolve_device, write_json
    from neuro_risk.viz import (
        build_jsviz_payload,
        save_confidence_distribution,
        save_interactive_uncertainty_scatter,
        save_probability_heatmap,
        save_uncertainty_panels,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    label_names = tuple(checkpoint["label_names"])
    config_payload = checkpoint["config"]

    bundle = load_dataset_bundle(args.dataset_dir, label_names=label_names)
    scaler = FeatureScaler.from_dict(checkpoint["scaler"])
    test_dataset = MultimodalRiskDataset(scaler.transform(bundle.test))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model_config = ModelConfig(
        tabular_hidden_dims=tuple(config_payload["model"]["tabular_hidden_dims"]),
        temporal_channels=tuple(config_payload["model"]["temporal_channels"]),
        temporal_embedding_dim=int(config_payload["model"]["temporal_embedding_dim"]),
        fusion_hidden_dim=int(config_payload["model"]["fusion_hidden_dim"]),
        dropout=float(config_payload["model"]["dropout"]),
    )
    model = NeuroRiskClassifier(
        num_tabular_features=int(config_payload["data"]["num_tabular_features"]),
        temporal_channels=int(config_payload["data"]["temporal_channels"]),
        num_classes=int(config_payload["data"]["num_classes"]),
        config=model_config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = resolve_device(args.device)
    model = model.to(device)
    learned_temperature = float(checkpoint["temperature"])

    result = mc_dropout_predict(
        model=model,
        loader=test_loader,
        device=device,
        num_samples=args.mc_samples,
        temperature=learned_temperature,
    )
    metrics = classification_metrics(result.labels, result.mean_probabilities, label_names)

    output_dir = ensure_directory(PROJECT_ROOT / "outputs" / "neuro_risk" / "inference")
    save_probability_heatmap(
        result.mean_probabilities,
        label_names,
        output_dir / "mc_mean_probabilities.png",
        title="MC mean predictive probabilities",
    )
    save_uncertainty_panels(result, label_names, output_dir / "mc_uncertainty_summary.png")
    save_confidence_distribution(result.confidence, output_dir / "confidence_distribution.png")
    interactive_path = output_dir / "interactive_uncertainty.html"
    save_interactive_uncertainty_scatter(
        mean_probabilities=result.mean_probabilities,
        entropy=result.predictive_entropy,
        mutual_information=result.mutual_information,
        labels=result.labels,
        predicted_labels=result.predicted_labels,
        label_names=label_names,
        output_path=interactive_path,
    )

    report = {
        "project": checkpoint["config"]["project_name"],
        "temperature": learned_temperature,
        "metrics": {"mc_dropout_test": metrics},
        "note": "Inference keeps dropout active to sample a posterior-like predictive distribution.",
    }
    report_path = output_dir / "inference_report.json"
    write_json(report_path, report)
    architecture = {
        "tabular_input_dim": int(config_payload["data"]["num_tabular_features"]),
        "temporal_input_shape": [
            int(config_payload["data"]["temporal_channels"]),
            int(config_payload["data"]["temporal_length"]),
        ],
        "tabular_hidden_dims": list(config_payload["model"]["tabular_hidden_dims"]),
        "temporal_channels": list(config_payload["model"]["temporal_channels"]),
        "temporal_embedding_dim": int(config_payload["model"]["temporal_embedding_dim"]),
        "fusion_hidden_dim": int(config_payload["model"]["fusion_hidden_dim"]),
        "num_classes": int(config_payload["data"]["num_classes"]),
        "dropout": float(config_payload["model"]["dropout"]),
        "stages": [
            {
                "name": "tabular-input",
                "nodes": int(config_payload["data"]["num_tabular_features"]),
                "group": "input",
            },
            {
                "name": "tabular-mlp",
                "nodes": int(config_payload["model"]["tabular_hidden_dims"][0]),
                "group": "tabular",
            },
            {
                "name": "temporal-conv",
                "nodes": int(config_payload["model"]["temporal_channels"][-1]),
                "group": "temporal",
            },
            {
                "name": "fusion",
                "nodes": int(config_payload["model"]["fusion_hidden_dim"]),
                "group": "fusion",
            },
            {
                "name": "classifier",
                "nodes": int(config_payload["data"]["num_classes"]),
                "group": "output",
            },
        ],
    }
    build_jsviz_payload(
        report,
        result,
        label_names,
        PROJECT_ROOT / "jsviz" / "public" / "latest_inference.json",
        architecture=architecture,
    )

    print(f"inference_report={report_path}")
    print(f"interactive_plot={interactive_path}")


if __name__ == "__main__":
    main()

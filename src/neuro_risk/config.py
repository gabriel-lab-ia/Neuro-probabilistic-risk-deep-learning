from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    num_samples: int = 720
    num_tabular_features: int = 16
    temporal_channels: int = 3
    temporal_length: int = 96
    num_classes: int = 3
    train_fraction: float = 0.65
    val_fraction: float = 0.15
    class_probabilities: tuple[float, ...] = (0.34, 0.33, 0.33)


@dataclass(slots=True)
class ModelConfig:
    tabular_hidden_dims: tuple[int, ...] = (96, 128, 64)
    temporal_channels: tuple[int, ...] = (32, 48, 64)
    temporal_embedding_dim: int = 64
    fusion_hidden_dim: int = 96
    dropout: float = 0.25


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 64
    epochs: int = 16
    learning_rate: float = 2.5e-3
    weight_decay: float = 1.0e-4
    patience: int = 5
    mc_samples: int = 24
    temperature_max_iter: int = 60
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"


@dataclass(slots=True)
class ProjectConfig:
    project_name: str = "Neuro-inspired probabilistic deep learning framework for neurological risk classification with calibrated uncertainty"
    label_names: tuple[str, ...] = (
        "baseline-risk",
        "monitor-closely",
        "high-risk-flag",
    )
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    dataset_dir: Path = Path("data/processed/neuro_risk_placeholder")
    model_dir: Path = Path("models/neuro_risk")
    output_dir: Path = Path("outputs/neuro_risk")
    jsviz_payload_path: Path = Path("jsviz/public/latest_inference.json")

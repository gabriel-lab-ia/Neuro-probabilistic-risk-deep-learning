from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuro_risk.config import DataConfig, ProjectConfig
from neuro_risk.utils.io import ensure_directory


@dataclass(slots=True)
class SyntheticSplit:
    tabular: np.ndarray
    temporal: np.ndarray
    labels: np.ndarray


@dataclass(slots=True)
class DatasetBundle:
    train: SyntheticSplit
    validation: SyntheticSplit
    test: SyntheticSplit
    label_names: tuple[str, ...]


def _generate_latent_states(rng: np.random.Generator, num_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    latent_burden = rng.normal(0.0, 0.95, size=num_samples)
    network_instability = 0.70 * latent_burden + rng.normal(0.0, 0.70, size=num_samples)
    neurochemical_shift = 0.55 * latent_burden + rng.normal(0.0, 0.65, size=num_samples)
    risk_score = latent_burden + 0.35 * network_instability + 0.25 * neurochemical_shift + rng.normal(
        0.0,
        0.55,
        size=num_samples,
    )
    labels = np.digitize(risk_score, bins=np.asarray([-0.40, 0.70], dtype=np.float32)).astype(np.int64)
    return (
        latent_burden.astype(np.float32),
        network_instability.astype(np.float32),
        neurochemical_shift.astype(np.float32),
        labels,
    )


def _generate_tabular_features(
    rng: np.random.Generator,
    labels: np.ndarray,
    latent_burden: np.ndarray,
    network_instability: np.ndarray,
    neurochemical_shift: np.ndarray,
    config: DataConfig,
) -> np.ndarray:
    num_samples = labels.shape[0]
    tabular = rng.normal(0.0, 0.45, size=(num_samples, config.num_tabular_features)).astype(np.float32)

    class_offsets = np.asarray(
        [
            np.linspace(-0.45, 0.15, config.num_tabular_features),
            np.linspace(-0.15, 0.20, config.num_tabular_features),
            np.linspace(0.10, 0.48, config.num_tabular_features),
        ],
        dtype=np.float32,
    )
    tabular += 0.35 * class_offsets[labels]

    tabular[:, 0] += 0.85 * latent_burden + 0.20 * np.sin(latent_burden)
    tabular[:, 1] += 0.90 * network_instability
    tabular[:, 2] += 0.80 * neurochemical_shift
    tabular[:, 3] += 0.55 * latent_burden - 0.30 * network_instability
    tabular[:, 4] += 0.45 * network_instability + 0.25 * neurochemical_shift
    tabular[:, 5] += 0.30 * np.square(latent_burden)
    tabular[:, 6] += 0.35 * latent_burden * neurochemical_shift
    tabular[:, 7] += 0.18 * labels
    tabular[:, 8] += 0.15 * labels + rng.normal(0.0, 0.20, size=num_samples)
    tabular[:, 9] += 0.25 * np.tanh(network_instability)
    tabular[:, 10] += 0.30 * latent_burden + rng.normal(0.0, 0.25, size=num_samples)
    tabular[:, 11] += 0.28 * neurochemical_shift
    tabular[:, 12] += 0.20 * latent_burden - 0.18 * labels
    tabular[:, 13] += 0.35 * np.sin(network_instability)
    tabular[:, 14] += 0.22 * np.cos(neurochemical_shift)
    tabular[:, 15] += 0.40 * latent_burden + 0.16 * labels

    return tabular.astype(np.float32)


def _generate_temporal_features(
    rng: np.random.Generator,
    labels: np.ndarray,
    latent_burden: np.ndarray,
    network_instability: np.ndarray,
    config: DataConfig,
) -> np.ndarray:
    num_samples = labels.shape[0]
    time_axis = np.linspace(0.0, 1.0, config.temporal_length, dtype=np.float32)

    slow_freq = 5.2 + 0.9 * np.tanh(latent_burden)
    fast_freq = 12.8 + 0.8 * labels + 0.6 * np.tanh(network_instability)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=num_samples).astype(np.float32)
    burst_center = 0.42 + 0.12 * np.tanh(latent_burden)
    burst_width = 0.16 - 0.03 * np.tanh(np.abs(latent_burden))

    slow_wave = (0.78 + 0.18 * latent_burden)[:, None] * np.sin(
        2.0 * np.pi * slow_freq[:, None] * time_axis[None, :] + phase[:, None]
    )
    spindle_envelope = np.exp(
        -0.5 * np.square((time_axis[None, :] - burst_center[:, None]) / burst_width[:, None])
    )
    spindle = (0.45 + 0.10 * latent_burden + 0.06 * labels)[:, None] * spindle_envelope * np.sin(
        2.0 * np.pi * fast_freq[:, None] * time_axis[None, :]
    )
    drift = (0.30 * latent_burden + 0.12 * network_instability)[:, None] * (time_axis[None, :] - 0.5)
    channel_noise = rng.normal(0.0, 0.18, size=(num_samples, config.temporal_channels, config.temporal_length))

    temporal = np.zeros((num_samples, config.temporal_channels, config.temporal_length), dtype=np.float32)
    temporal[:, 0, :] = slow_wave + 0.20 * spindle
    temporal[:, 1, :] = 0.35 * slow_wave + spindle + drift
    temporal[:, 2, :] = 0.25 * slow_wave + 0.55 * spindle + 0.4 * np.cos(
        2.0 * np.pi * (2.2 + 0.18 * latent_burden[:, None] + 0.12 * labels[:, None]) * time_axis[None, :]
    )

    temporal += channel_noise.astype(np.float32)
    return temporal.astype(np.float32)


def _split_indices(rng: np.random.Generator, config: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(config.num_samples)
    rng.shuffle(indices)
    train_end = int(config.num_samples * config.train_fraction)
    validation_end = train_end + int(config.num_samples * config.val_fraction)
    train_idx = indices[:train_end]
    validation_idx = indices[train_end:validation_end]
    test_idx = indices[validation_end:]
    return train_idx, validation_idx, test_idx


def _make_split(indices: np.ndarray, tabular: np.ndarray, temporal: np.ndarray, labels: np.ndarray) -> SyntheticSplit:
    return SyntheticSplit(
        tabular=tabular[indices],
        temporal=temporal[indices],
        labels=labels[indices],
    )


def generate_synthetic_dataset(config: ProjectConfig) -> DatasetBundle:
    rng = np.random.default_rng(config.training.seed)
    latent_burden, network_instability, neurochemical_shift, labels = _generate_latent_states(
        rng,
        config.data.num_samples,
    )
    tabular = _generate_tabular_features(
        rng,
        labels,
        latent_burden,
        network_instability,
        neurochemical_shift,
        config.data,
    )
    temporal = _generate_temporal_features(rng, labels, latent_burden, network_instability, config.data)
    train_idx, validation_idx, test_idx = _split_indices(rng, config.data)

    return DatasetBundle(
        train=_make_split(train_idx, tabular, temporal, labels),
        validation=_make_split(validation_idx, tabular, temporal, labels),
        test=_make_split(test_idx, tabular, temporal, labels),
        label_names=config.label_names,
    )


def save_dataset_bundle(bundle: DatasetBundle, dataset_dir: Path) -> None:
    ensure_directory(dataset_dir)
    for split_name, split in (
        ("train", bundle.train),
        ("validation", bundle.validation),
        ("test", bundle.test),
    ):
        np.savez_compressed(
            dataset_dir / f"{split_name}.npz",
            tabular=split.tabular,
            temporal=split.temporal,
            labels=split.labels,
        )


def load_dataset_bundle(dataset_dir: Path, label_names: tuple[str, ...]) -> DatasetBundle:
    loaded: dict[str, SyntheticSplit] = {}
    for split_name in ("train", "validation", "test"):
        payload = np.load(dataset_dir / f"{split_name}.npz")
        loaded[split_name] = SyntheticSplit(
            tabular=payload["tabular"].astype(np.float32),
            temporal=payload["temporal"].astype(np.float32),
            labels=payload["labels"].astype(np.int64),
        )
    return DatasetBundle(
        train=loaded["train"],
        validation=loaded["validation"],
        test=loaded["test"],
        label_names=label_names,
    )

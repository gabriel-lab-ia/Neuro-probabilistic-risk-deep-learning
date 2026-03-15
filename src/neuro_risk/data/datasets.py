from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from neuro_risk.data.synthetic import DatasetBundle, SyntheticSplit


@dataclass(slots=True)
class FeatureScaler:
    tabular_mean: np.ndarray
    tabular_std: np.ndarray
    temporal_mean: np.ndarray
    temporal_std: np.ndarray

    def transform(self, split: SyntheticSplit) -> SyntheticSplit:
        return SyntheticSplit(
            tabular=((split.tabular - self.tabular_mean) / self.tabular_std).astype(np.float32),
            temporal=((split.temporal - self.temporal_mean) / self.temporal_std).astype(np.float32),
            labels=split.labels.astype(np.int64),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "tabular_mean": self.tabular_mean.tolist(),
            "tabular_std": self.tabular_std.tolist(),
            "temporal_mean": self.temporal_mean.tolist(),
            "temporal_std": self.temporal_std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureScaler":
        return cls(
            tabular_mean=np.asarray(payload["tabular_mean"], dtype=np.float32),
            tabular_std=np.asarray(payload["tabular_std"], dtype=np.float32),
            temporal_mean=np.asarray(payload["temporal_mean"], dtype=np.float32),
            temporal_std=np.asarray(payload["temporal_std"], dtype=np.float32),
        )


class MultimodalRiskDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, split: SyntheticSplit) -> None:
        self.tabular = torch.from_numpy(split.tabular)
        self.temporal = torch.from_numpy(split.temporal)
        self.labels = torch.from_numpy(split.labels)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.tabular[index], self.temporal[index], self.labels[index]


def fit_feature_scaler(train_split: SyntheticSplit) -> FeatureScaler:
    tabular_mean = train_split.tabular.mean(axis=0)
    tabular_std = train_split.tabular.std(axis=0) + 1.0e-6
    temporal_mean = train_split.temporal.mean(axis=(0, 2), keepdims=True)
    temporal_std = train_split.temporal.std(axis=(0, 2), keepdims=True) + 1.0e-6
    return FeatureScaler(
        tabular_mean=tabular_mean.astype(np.float32),
        tabular_std=tabular_std.astype(np.float32),
        temporal_mean=temporal_mean.astype(np.float32),
        temporal_std=temporal_std.astype(np.float32),
    )


def build_data_loaders(
    bundle: DatasetBundle,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[dict[str, DataLoader[Any]], FeatureScaler]:
    scaler = fit_feature_scaler(bundle.train)
    scaled_bundle = DatasetBundle(
        train=scaler.transform(bundle.train),
        validation=scaler.transform(bundle.validation),
        test=scaler.transform(bundle.test),
        label_names=bundle.label_names,
    )
    loaders = {
        "train": DataLoader(
            MultimodalRiskDataset(scaled_bundle.train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "validation": DataLoader(
            MultimodalRiskDataset(scaled_bundle.validation),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            MultimodalRiskDataset(scaled_bundle.test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return loaders, scaler

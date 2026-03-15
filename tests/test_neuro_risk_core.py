from __future__ import annotations

import os
import unittest
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"

MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neuro_risk.config import ProjectConfig
from neuro_risk.data import MultimodalRiskDataset, SyntheticSplit, generate_synthetic_dataset
from neuro_risk.evaluation import expected_calibration_error
from neuro_risk.inference import MCDropoutResult, mc_dropout_predict
from neuro_risk.models import NeuroRiskClassifier
from neuro_risk.viz import build_jsviz_payload
from neuro_risk.utils import read_json


class CalibrationMetricsTests(unittest.TestCase):
    def test_expected_calibration_error_is_near_zero_for_perfect_predictions(self) -> None:
        labels = np.asarray([0, 1, 2], dtype=np.int64)
        probabilities = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        ece, _ = expected_calibration_error(labels, probabilities, num_bins=5)
        self.assertLess(ece, 1.0e-6)


class MCDropoutTests(unittest.TestCase):
    def test_mc_dropout_outputs_expected_shapes(self) -> None:
        config = ProjectConfig()
        model = NeuroRiskClassifier(
            num_tabular_features=config.data.num_tabular_features,
            temporal_channels=config.data.temporal_channels,
            num_classes=config.data.num_classes,
            config=config.model,
        )
        split = SyntheticSplit(
            tabular=np.random.randn(8, config.data.num_tabular_features).astype(np.float32),
            temporal=np.random.randn(8, config.data.temporal_channels, config.data.temporal_length).astype(np.float32),
            labels=np.random.randint(0, config.data.num_classes, size=8, dtype=np.int64),
        )
        loader = DataLoader(MultimodalRiskDataset(split), batch_size=4, shuffle=False)
        result = mc_dropout_predict(model, loader, torch.device("cpu"), num_samples=4)
        self.assertEqual(result.mean_probabilities.shape, (8, config.data.num_classes))
        self.assertEqual(result.class_variances.shape, (8, config.data.num_classes))
        self.assertEqual(result.predictive_entropy.shape, (8,))

    def test_mc_dropout_rejects_non_positive_sample_count(self) -> None:
        config = ProjectConfig()
        model = NeuroRiskClassifier(
            num_tabular_features=config.data.num_tabular_features,
            temporal_channels=config.data.temporal_channels,
            num_classes=config.data.num_classes,
            config=config.model,
        )
        split = SyntheticSplit(
            tabular=np.random.randn(4, config.data.num_tabular_features).astype(np.float32),
            temporal=np.random.randn(4, config.data.temporal_channels, config.data.temporal_length).astype(np.float32),
            labels=np.random.randint(0, config.data.num_classes, size=4, dtype=np.int64),
        )
        loader = DataLoader(MultimodalRiskDataset(split), batch_size=2, shuffle=False)
        with self.assertRaises(ValueError):
            mc_dropout_predict(model, loader, torch.device("cpu"), num_samples=0)


class SyntheticDatasetTests(unittest.TestCase):
    def test_synthetic_dataset_split_sizes_match_configuration(self) -> None:
        config = ProjectConfig()
        bundle = generate_synthetic_dataset(config)
        total = (
            bundle.train.labels.shape[0]
            + bundle.validation.labels.shape[0]
            + bundle.test.labels.shape[0]
        )
        self.assertEqual(total, config.data.num_samples)
        self.assertEqual(bundle.train.tabular.shape[1], config.data.num_tabular_features)
        self.assertEqual(bundle.train.temporal.shape[1], config.data.temporal_channels)
        self.assertEqual(bundle.train.temporal.shape[2], config.data.temporal_length)


class JSVizPayloadTests(unittest.TestCase):
    def test_build_jsviz_payload_contains_frontend_sections(self) -> None:
        mc_result = MCDropoutResult(
            logits_samples=np.zeros((2, 4, 3), dtype=np.float32),
            probability_samples=np.full((2, 4, 3), 1.0 / 3.0, dtype=np.float32),
            mean_probabilities=np.full((4, 3), 1.0 / 3.0, dtype=np.float32),
            class_variances=np.full((4, 3), 0.01, dtype=np.float32),
            predictive_entropy=np.full((4,), 1.0, dtype=np.float32),
            expected_entropy=np.full((4,), 0.8, dtype=np.float32),
            mutual_information=np.full((4,), 0.2, dtype=np.float32),
            confidence=np.full((4,), 0.5, dtype=np.float32),
            predicted_labels=np.asarray([0, 1, 2, 1], dtype=np.int64),
            labels=np.asarray([0, 1, 2, 0], dtype=np.int64),
        )
        report = {
            "project": "demo-project",
            "temperature": 1.1,
            "best_epoch": 3,
            "device": "cpu",
            "history": {
                "train_loss": [1.0, 0.8],
                "validation_loss": [0.9, 0.85],
                "validation_accuracy": [0.6, 0.7],
                "validation_f1_macro": [0.58, 0.69],
            },
            "metrics": {
                "raw_test": {
                    "accuracy": 0.5,
                    "f1_macro": 0.48,
                    "auroc_ovr": 0.7,
                    "mean_confidence": 0.62,
                    "reliability_bins": [],
                    "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
                "calibrated_test": {
                    "accuracy": 0.6,
                    "f1_macro": 0.58,
                    "auroc_ovr": 0.73,
                    "mean_confidence": 0.59,
                    "reliability_bins": [],
                    "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
                "mc_dropout_test": {
                    "accuracy": 0.7,
                    "f1_macro": 0.68,
                    "auroc_ovr": 0.77,
                    "mean_confidence": 0.55,
                    "mean_entropy": 1.0,
                    "ece": 0.1,
                    "reliability_bins": [],
                    "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
            },
        }
        architecture = {
            "tabular_input_dim": 16,
            "temporal_input_shape": [3, 96],
            "fusion_hidden_dim": 64,
            "dropout": 0.25,
            "num_classes": 3,
            "stages": [{"name": "fusion", "nodes": 64, "group": "fusion"}],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "payload.json"
            build_jsviz_payload(report, mc_result, ("a", "b", "c"), out_path, architecture=architecture)
            payload = read_json(out_path)

        self.assertIn("chart_data", payload)
        self.assertIn("architecture", payload)
        self.assertIn("uncertainty_scatter", payload["chart_data"])
        self.assertEqual(payload["architecture"]["fusion_hidden_dim"], 64)
        self.assertEqual(len(payload["uncertain_examples"]), 4)


if __name__ == "__main__":
    unittest.main()

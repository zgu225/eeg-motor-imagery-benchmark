"""Regression tests for preprocessing, metrics, and model contracts."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bci_project.data.preprocessing import standardize_epochs, train_test_subject_split
from bci_project.evaluation.metrics import compute_metrics
from bci_project.models.deep_models import build_deep_model


class PreprocessingTests(unittest.TestCase):
    """Verify standardization and complete-subject holdout behavior."""

    def test_standardization(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(4, 3, 100)).astype(np.float32)
        standardized = standardize_epochs(X)

        self.assertEqual(standardized.shape, X.shape)
        np.testing.assert_allclose(standardized.mean(axis=-1), 0.0, atol=1e-5)
        np.testing.assert_allclose(standardized.std(axis=-1), 1.0, atol=1e-5)

    def test_subject_split(self) -> None:
        X = np.arange(24).reshape(4, 2, 3)
        y = np.array([0, 1, 0, 1])
        subjects = np.array([1, 1, 2, 2])
        X_train, X_test, y_train, y_test = train_test_subject_split(X, y, subjects, 2)

        self.assertEqual(len(X_train), 2)
        self.assertEqual(len(X_test), 2)
        np.testing.assert_array_equal(y_train, [0, 1])
        np.testing.assert_array_equal(y_test, [0, 1])


class MetricsTests(unittest.TestCase):
    """Verify metric behavior for a perfectly classified batch."""

    def test_perfect_predictions(self) -> None:
        labels = np.array([0, 1, 2, 3])
        metrics = compute_metrics(labels, labels)

        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.macro_f1, 1.0)
        np.testing.assert_array_equal(metrics.confusion_matrix, np.eye(4, dtype=int))


class ModelTests(unittest.TestCase):
    """Verify neural-network tensor contracts and EEGNet constraints."""

    def test_deep_model_output_shapes(self) -> None:
        batch = torch.randn(2, 22, 875)
        for name in ("eegnet", "lawhern_eegnet", "shallow_convnet"):
            with self.subTest(model=name):
                model = build_deep_model(name, 22, 875, 4, 0.5)
                self.assertEqual(tuple(model(batch).shape), (2, 4))

    def test_lawhern_max_norm(self) -> None:
        model = build_deep_model("lawhern_eegnet", 22, 875, 4, 0.5)
        with torch.no_grad():
            model.classifier.weight.fill_(10.0)
        model.apply_max_norm()

        row_norms = model.classifier.weight.norm(p=2, dim=1)
        self.assertTrue(torch.all(row_norms <= 0.25001))


if __name__ == "__main__":
    unittest.main()

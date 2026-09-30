"""Classification metrics used by all evaluation protocols."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


@dataclass(frozen=True)
class ClassificationMetrics:
    """Aggregate scalar metrics and the confusion matrix."""

    accuracy: float
    macro_f1: float
    confusion_matrix: np.ndarray


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """Compute accuracy, macro-F1, and the confusion matrix."""

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        confusion_matrix=confusion_matrix(y_true, y_pred),
    )

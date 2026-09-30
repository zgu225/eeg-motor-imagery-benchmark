"""EEG preprocessing and subject-level splitting utilities."""

from __future__ import annotations

import numpy as np


def standardize_epochs(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Standardize every channel independently within each trial.

    Neural networks are sensitive to input scale; per-trial z-scoring reduces
    absolute amplitude differences while preserving temporal structure.
    """

    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    # MNE returns volts: adding 1e-6 would materially distort EEG-scale signals.
    return (X - mean) / np.maximum(std, eps)


def train_test_subject_split(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    test_subject: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hold out one complete subject and return train/test arrays."""

    test_mask = subjects == test_subject
    train_mask = ~test_mask
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]

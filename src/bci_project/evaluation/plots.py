"""Visualization utilities for experiment diagnostics and results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    """Render and save a confusion-matrix heatmap."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_subject_barplot(results: pd.DataFrame, output_path: str | Path, title: str) -> None:
    """Render and save per-subject accuracy bars."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    subject_column = "subject" if "subject" in results.columns else "test_subject"
    sns.barplot(data=results, x=subject_column, y="accuracy", hue="model")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_training_history(
    train_loss: list[float],
    val_accuracy: list[float],
    output_path: str | Path,
    title: str,
) -> None:
    """Save loss and validation curves for convergence diagnostics."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_loss) + 1)

    figure, loss_axis = plt.subplots(figsize=(8, 5))
    accuracy_axis = loss_axis.twinx()
    loss_axis.plot(epochs, train_loss, color="#167d9a", label="Train loss")
    accuracy_axis.plot(epochs, val_accuracy, color="#d05a47", label="Validation accuracy")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Cross-entropy loss")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0.0, 1.0)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

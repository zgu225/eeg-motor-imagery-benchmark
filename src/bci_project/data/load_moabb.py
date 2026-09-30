"""Load the public motor imagery dataset through MOABB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery


@dataclass(frozen=True)
class EEGDataset:
    """Container for EEG arrays and metadata required by the experiments."""

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    channels: list[str]
    sfreq: float
    label_mapping: dict[str, int]
    runs: np.ndarray | None = None


def load_bnci2014_001(
    subjects: list[int] | None = None,
    tmin: float = 0.5,
    tmax: float = 4.0,
    fmin: float = 8.0,
    fmax: float = 30.0,
) -> EEGDataset:
    """Load and preprocess the BNCI2014_001 motor imagery dataset.

    MOABB downloads the dataset automatically. The returned ``X`` array has
    shape ``(n_trials, n_channels, n_samples)``.
    """

    dataset = BNCI2014_001()
    selected_subjects = subjects or dataset.subject_list

    # MotorImagery extracts events, crops epochs, and applies band-pass filtering.
    paradigm = MotorImagery(
        n_classes=4,
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
    )
    # Request MNE Epochs so channel names and sampling frequency are obtained
    # from the data files instead of version-dependent paradigm attributes.
    epochs, labels, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=selected_subjects,
        return_epochs=True,
    )
    X = epochs.get_data(copy=True)

    # Map string event labels to contiguous integers for sklearn and PyTorch.
    unique_labels = sorted(set(labels))
    label_mapping = {label: index for index, label in enumerate(unique_labels)}
    y = np.asarray([label_mapping[label] for label in labels], dtype=np.int64)

    return EEGDataset(
        X=X.astype(np.float32),
        y=y,
        subjects=metadata["subject"].to_numpy(dtype=np.int64),
        sessions=metadata["session"].astype(str).to_numpy(),
        channels=list(epochs.ch_names),
        sfreq=float(epochs.info["sfreq"]),
        label_mapping=label_mapping,
        runs=metadata["run"].astype(str).to_numpy(),
    )


def load_dataset_from_config(config: dict[str, Any]) -> EEGDataset:
    """Load the configured dataset and preprocessing parameters."""

    dataset_config = config["dataset"]
    if dataset_config["name"] != "BNCI2014_001":
        raise ValueError(f"Unsupported dataset: {dataset_config['name']}")

    return load_bnci2014_001(
        subjects=dataset_config.get("subjects"),
        tmin=float(dataset_config.get("tmin", 0.5)),
        tmax=float(dataset_config.get("tmax", 4.0)),
        fmin=float(dataset_config.get("fmin", 8.0)),
        fmax=float(dataset_config.get("fmax", 30.0)),
    )

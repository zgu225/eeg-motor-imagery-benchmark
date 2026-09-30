"""Portable inference bundles for already filtered EEG epochs in volts."""

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from bci_project.data.preprocessing import standardize_epochs
from bci_project.models.deep_models import build_deep_model
from bci_project.training import predict_model


def save_bundle(model, directory, metadata):
    """Persist model weights together with the complete input contract."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "input.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if metadata["model"].startswith("csp_"):
        joblib.dump(model, directory / "model.joblib")
    else:
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory / "model.pt")


def predict_bundle(directory, X, channels, sfreq):
    """Validate input metadata, load a trusted local bundle, and predict labels.

    Filtering/epoching must match input.json. joblib files must only be loaded
    from trusted sources because their deserialization can execute Python code.
    """
    directory = Path(directory)
    meta = json.loads((directory / "input.json").read_text(encoding="utf-8"))
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or list(X.shape[1:]) != [meta["n_channels"], meta["n_times"]]:
        raise ValueError("Expected (trials, channels, samples) matching input.json")
    if len(X) == 0 or not np.isfinite(X).all():
        raise ValueError("Input must contain finite, non-empty EEG epochs")
    if list(channels) != meta["channels"] or not np.isclose(sfreq, meta["sfreq"]):
        raise ValueError("Channel order and sampling rate must match the model")
    if meta["model"].startswith("csp_"):
        model = joblib.load(directory / "model.joblib")
        return model.predict(X)
    model = build_deep_model(meta["model"], meta["n_channels"], meta["n_times"],
                             len(meta["label_mapping"]), meta["dropout"])
    model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
    return predict_model(model, standardize_epochs(X))

"""Auditable benchmark: grouped validation, fold artifacts, and safe resume."""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import time

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import torch
import yaml

from bci_project.artifacts import predict_bundle, save_bundle
from bci_project.data.load_moabb import load_dataset_from_config
from bci_project.data.preprocessing import standardize_epochs
from bci_project.evaluation.metrics import compute_metrics
from bci_project.models.csp_baseline import build_csp_lda, build_csp_svm
from bci_project.models.deep_models import build_deep_model
from bci_project.training import predict_model, train_model
from bci_project.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["csp_lda", "csp_svm", "lawhern_eegnet", "eegnet", "shallow_convnet"]
PROTOCOLS = ["cross_session", "loso"]


def write_json(path, value):
    """Replace a JSON artifact atomically to avoid partial completion markers."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_indices(data, protocol, subject, seed):
    """Use disjoint recording runs or disjoint subjects for validation.

    All models use exactly the same fit/test indices for a fair comparison.
    Baselines do not use validation labels. No target-subject adaptation occurs.
    """
    if protocol == "cross_session":
        sessions = sorted(set(data.sessions[data.subjects == subject]))
        if len(sessions) != 2:
            raise ValueError("Cross-session evaluation requires exactly two sessions")
        pool = np.flatnonzero((data.subjects == subject) & (data.sessions == sessions[0]))
        test = np.flatnonzero((data.subjects == subject) & (data.sessions == sessions[1]))
        if data.runs is None:
            raise ValueError("Recording run metadata is required")
        groups = data.runs[pool]
    elif protocol == "loso":
        pool = np.flatnonzero(data.subjects != subject)
        test = np.flatnonzero(data.subjects == subject)
        groups = data.subjects[pool]
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    if len(set(groups)) < 2 or len(test) == 0:
        raise ValueError("Insufficient groups for independent fit/validation/test splits")
    # One complete run (cross-session) or participant (LOSO) is held out.
    for fit_rel, val_rel in GroupShuffleSplit(n_splits=100, test_size=1, random_state=seed).split(pool, groups=groups):
        fit, val = pool[fit_rel], pool[val_rel]
        if set(data.y[fit]) == set(data.y) == set(data.y[val]) == set(data.y[test]):
            return fit, val, test
    raise ValueError("Could not form grouped splits containing all classes")


def complete_fold(directory):
    """Only resume a fold when every artifact matches its recorded checksum."""
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    for name, digest in record["checksums"].items():
        if not (directory / name).is_file() or sha256(directory / name) != digest:
            raise ValueError(f"Corrupted fold artifact: {directory / name}")
    return True


def run(args):
    """Execute all requested folds; a completed fold is never retrained."""
    torch.set_num_threads(args.threads)
    mne.set_log_level("ERROR")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    source = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))}
    identity = dict(config=config, models=args.models, protocols=args.protocols,
                    seeds=args.seeds, threads=args.threads, source=source)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise ValueError("Configuration or source changed. Use a new output directory.")
    else:
        write_json(manifest, dict(signature=signature, **identity,
                   python=platform.python_version(), platform=platform.platform(),
                   packages={d.metadata["Name"]: d.version for d in distributions()}))
    data = load_dataset_from_config(config)
    # Hash the processed data as well as code/configuration before resuming.
    digest = hashlib.sha256(data.X.tobytes() + data.y.tobytes())
    for values in (data.subjects, data.sessions, data.runs):
        digest.update(json.dumps(values.tolist()).encode())
    description = dict(shape=list(data.X.shape), channels=data.channels, sfreq=data.sfreq,
                       label_mapping=data.label_mapping, units="V", sha256=digest.hexdigest())
    data_path = output / "dataset.json"
    if data_path.exists() and json.loads(data_path.read_text()) != description:
        raise ValueError("Dataset changed since this run was created")
    write_json(data_path, description)
    print(f"Dataset {data.X.shape}; output={output}", flush=True)
    for seed in args.seeds:
        for protocol in args.protocols:
            for subject in sorted(set(data.subjects)):
                fit, val, test = split_indices(data, protocol, subject, seed)
                for name in args.models:
                    directory = output / f"{protocol}/{name}/seed_{seed}/subject_{subject}"
                    directory.mkdir(parents=True, exist_ok=True)
                    if complete_fold(directory):
                        print(f"Resume: {directory.relative_to(output)}", flush=True)
                        continue
                    print(f"Fit: {directory.relative_to(output)}", flush=True)
                    start = time.monotonic()
                    set_seed(seed)
                    meta = dict(model=name, n_channels=data.X.shape[1], n_times=data.X.shape[2],
                                channels=data.channels, sfreq=data.sfreq, units="V",
                                label_mapping=data.label_mapping, dropout=config["model"]["dropout"],
                                preprocessing=config["dataset"])
                    if name.startswith("csp_"):
                        model = (build_csp_lda if name == "csp_lda" else build_csp_svm)()
                        model.fit(data.X[fit], data.y[fit])
                        prediction = model.predict(data.X[test])
                        history = {}
                    else:
                        model = build_deep_model(name, meta["n_channels"], meta["n_times"],
                                                 len(data.label_mapping), meta["dropout"])
                        settings = dict(config["training"], seed=seed)
                        history = asdict(train_model(model, standardize_epochs(data.X[fit]), data.y[fit],
                                validation_data=(standardize_epochs(data.X[val]), data.y[val]), **settings))
                        prediction = predict_model(model, standardize_epochs(data.X[test]))
                    save_bundle(model, directory, meta)
                    # Verify serialized weights reproduce inference before marking success.
                    reloaded = predict_bundle(directory, data.X[test], data.channels, data.sfreq)
                    np.testing.assert_array_equal(prediction, reloaded)
                    np.savez_compressed(directory / "split.npz", fit=fit, validation=val, test=test)
                    pd.DataFrame(dict(trial_index=test, subject=data.subjects[test],
                                      session=data.sessions[test], y_true=data.y[test],
                                      y_pred=prediction)).to_csv(directory / "predictions.csv", index=False)
                    metrics = compute_metrics(data.y[test], prediction)
                    row = dict(protocol=protocol, model=name, seed=seed, subject=int(subject),
                               n_fit=len(fit), n_validation=len(val), n_test=len(test),
                               accuracy=metrics.accuracy, macro_f1=metrics.macro_f1,
                               seconds=time.monotonic()-start, epochs=len(history.get("train_loss", [])))
                    write_json(directory / "metrics.json", row)
                    write_json(directory / "history.json", history)
                    write_json(directory / "confusion.json", metrics.confusion_matrix.tolist())
                    checksums = {p.name: sha256(p) for p in directory.iterdir()
                                 if p.is_file() and p.name != "complete.json"}
                    write_json(directory / "complete.json", dict(checksums=checksums))
                    print(f"Finished accuracy={row['accuracy']:.4f} seconds={row['seconds']:.1f}", flush=True)
    from bci_project.reporting import report
    report(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bnci2014_001.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/benchmark_v2")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=PROTOCOLS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, help="Override only for explicitly labeled smoke runs")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

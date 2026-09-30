"""Subject-specific cross-session evaluation protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bci_project.config import load_config
from bci_project.data.load_moabb import load_dataset_from_config
from bci_project.data.preprocessing import standardize_epochs
from bci_project.evaluation.metrics import compute_metrics
from bci_project.evaluation.plots import save_confusion_matrix, save_training_history
from bci_project.models.csp_baseline import build_csp_lda, build_csp_svm, fit_predict_baseline
from bci_project.models.deep_models import build_deep_model
from bci_project.training import predict_model, train_model
from bci_project.utils.seed import set_seed


def run_within_subject(config_path: str, model_name: str) -> pd.DataFrame:
    """Train and test an independent model for each subject."""

    config = load_config(config_path)
    set_seed(int(config["training"]["seed"]))

    dataset = load_dataset_from_config(config)
    n_classes = len(dataset.label_mapping)
    results = []
    aggregate_confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    label_names = [name for name, _ in sorted(dataset.label_mapping.items(), key=lambda item: item[1])]

    for subject in sorted(set(dataset.subjects)):
        # Restrict the fold to one participant, as in subject-specific calibration.
        mask = dataset.subjects == subject
        X_subject = dataset.X[mask]
        y_subject = dataset.y[mask]

        # Dataset 2a contains two independent sessions. Session-wise splitting
        # evaluates day-to-day robustness without randomly mixing trials.
        subject_sessions = dataset.sessions[mask]
        unique_sessions = sorted(set(subject_sessions))
        if len(unique_sessions) < 2:
            raise ValueError(f"Subject {subject} has fewer than two sessions")
        train_mask = subject_sessions == unique_sessions[0]
        test_mask = ~train_mask
        X_train, X_test = X_subject[train_mask], X_subject[test_mask]
        y_train, y_test = y_subject[train_mask], y_subject[test_mask]

        if model_name == "csp_lda":
            y_pred = fit_predict_baseline(build_csp_lda(), X_train, y_train, X_test)
        elif model_name == "csp_svm":
            y_pred = fit_predict_baseline(build_csp_svm(), X_train, y_train, X_test)
        else:
            # Neural networks receive per-trial, per-channel standardized inputs.
            X_train = standardize_epochs(X_train)
            X_test = standardize_epochs(X_test)
            model = build_deep_model(
                name=model_name,
                n_channels=X_train.shape[1],
                n_times=X_train.shape[2],
                n_classes=n_classes,
                dropout=float(config["model"]["dropout"]),
            )
            history = train_model(model, X_train, y_train, **config["training"])
            save_training_history(
                history.train_loss,
                history.val_accuracy,
                Path(config["output"]["figures_dir"]) / f"within_subject_{model_name}_subject_{subject}_training.png",
                f"Within-subject {model_name} - subject {subject}",
            )
            y_pred = predict_model(model, X_test)

        metrics = compute_metrics(y_test, y_pred)
        aggregate_confusion += metrics.confusion_matrix
        results.append(
            {
                "subject": int(subject),
                "model": model_name,
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
            }
        )

    results_df = pd.DataFrame(results)
    output_path = Path(config["output"]["results_dir"]) / f"within_subject_{model_name}.csv"
    results_df.to_csv(output_path, index=False)
    save_confusion_matrix(
        aggregate_confusion,
        label_names,
        Path(config["output"]["figures_dir"]) / f"within_subject_{model_name}_confusion_matrix.png",
        f"Within-subject {model_name}",
    )
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bnci2014_001.yaml")
    parser.add_argument(
        "--model",
        default="csp_lda",
        choices=["csp_lda", "csp_svm", "lawhern_eegnet", "eegnet", "shallow_convnet"],
    )
    args = parser.parse_args()

    results = run_within_subject(args.config, args.model)
    print(results)
    print("Mean accuracy:", results["accuracy"].mean())


if __name__ == "__main__":
    main()

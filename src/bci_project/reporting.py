"""Generate auditable tables and figures from complete benchmark folds."""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def report(directory):
    """Validate completeness and generate descriptive paired uncertainty estimates."""
    from bci_project.benchmark import complete_fold, write_json
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    dataset = json.loads((directory / "dataset.json").read_text())
    subjects = manifest["config"]["dataset"].get("subjects") or list(range(1, 10))
    rows, predictions = [], []
    for protocol, model, seed, subject in itertools.product(
            manifest["protocols"], manifest["models"], manifest["seeds"], subjects):
        fold = directory / f"{protocol}/{model}/seed_{seed}/subject_{subject}"
        if not complete_fold(fold):
            raise ValueError(f"Incomplete benchmark: {fold}")
        rows.append(json.loads((fold / "metrics.json").read_text()))
        frame = pd.read_csv(fold / "predictions.csv")
        predictions.append(frame.assign(protocol=protocol, model=model, seed=seed))
    results = pd.DataFrame(rows)
    results.to_csv(directory / "fold_metrics.csv", index=False)
    # Average seeds within subjects before computing between-subject statistics.
    per_subject = results.groupby(["protocol", "model", "subject"])[["accuracy", "macro_f1"]].mean().reset_index()
    summary = per_subject.groupby(["protocol", "model"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        subjects=("subject", "count")).reset_index()
    summary.to_csv(directory / "summary.csv", index=False)
    comparisons = []
    for protocol in manifest["protocols"]:
        pivot = per_subject[per_subject.protocol == protocol].pivot(index="subject", columns="model", values="accuracy")
        if {"csp_lda", "lawhern_eegnet"}.issubset(pivot.columns):
            differences = (pivot.lawhern_eegnet - pivot.csp_lda).to_numpy()
            rng = np.random.default_rng(42)
            boot = rng.choice(differences, (10000, len(differences)), replace=True).mean(axis=1)
            signs = np.asarray(list(itertools.product([-1, 1], repeat=len(differences))))
            permuted = (signs * differences).mean(axis=1)
            comparisons.append(dict(protocol=protocol, difference_pp=float(differences.mean()*100),
                               descriptive_bootstrap_95ci_pp=(np.quantile(boot, [0.025, 0.975])*100).tolist(),
                               exploratory_signflip_p=float(np.mean(abs(permuted) >= abs(differences.mean())-1e-12))))
    write_json(directory / "paired_comparison.json", comparisons)
    figure, axes = plt.subplots(1, len(manifest["protocols"]), figsize=(13, 5), squeeze=False)
    for ax, protocol in zip(axes[0], manifest["protocols"]):
        part = summary[summary.protocol == protocol].sort_values("accuracy_mean", ascending=False)
        ax.bar(part.model, part.accuracy_mean, yerr=part.accuracy_sd.fillna(0), color="#287f8e", capsize=3)
        ax.axhline(.25, color="gray", linestyle="--", label="Chance")
        ax.set(title=protocol, ylim=(0, 1), ylabel="Accuracy (subject mean +/- SD)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
    figure.tight_layout()
    figure.savefig(directory / "comparison.png", dpi=160)
    plt.close(figure)
    combined = pd.concat(predictions, ignore_index=True)
    labels = list(range(len(dataset["label_mapping"])))
    names = [k for k, v in sorted(dataset["label_mapping"].items(), key=lambda pair: pair[1])]
    for (protocol, model), frame in combined.groupby(["protocol", "model"]):
        matrix = confusion_matrix(frame.y_true, frame.y_pred, labels=labels)
        pd.DataFrame(matrix, index=names, columns=names).to_csv(directory / f"{protocol}_{model}_confusion.csv")
        from bci_project.evaluation.plots import save_confusion_matrix
        save_confusion_matrix(matrix, names, directory / f"{protocol}_{model}_confusion.png", f"{protocol}: {model}")
    lines = ["# Verified Benchmark Results", "", f"Completed folds: {len(results)}.", "",
             "| Protocol | Model | Accuracy +/- SD (%) | Macro-F1 +/- SD (%) |", "|---|---|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.protocol} | {row.model} | {row.accuracy_mean*100:.2f} +/- {row.accuracy_sd*100:.2f} | {row.macro_f1_mean*100:.2f} +/- {row.macro_f1_sd*100:.2f} |")
    lines += ["", "## Paired EEGNet versus CSP-LDA comparison", ""]
    for comparison in comparisons:
        lo, hi = comparison["descriptive_bootstrap_95ci_pp"]
        lines.append(f"- {comparison['protocol']}: {comparison['difference_pp']:.2f} percentage points; descriptive subject-bootstrap 95% interval [{lo:.2f}, {hi:.2f}]; exploratory sign-flip p={comparison['exploratory_signflip_p']:.4f}.")
    lines += ["", "These are exploratory, unadjusted comparisons on nine subjects. LOSO training sets overlap,",
              "so folds are not independent replications. Intervals are descriptive, not confirmatory evidence.",
              "A single seed does not measure training-seed uncertainty. No causal explanation follows",
              "from the model ranking alone. Model reload predictions matched in-memory predictions for every fold.",
              "", "![Model comparison](comparison.png)", ""]
    (directory / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)

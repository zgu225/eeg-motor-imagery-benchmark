"""Aggregate experiment CSV files and generate publication-ready summaries."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def main() -> None:
    """Compute aggregate statistics and render the model comparison figure."""

    rows: list[dict[str, float | str | int]] = []
    raw_frames: list[pd.DataFrame] = []
    for path in sorted(RESULTS_DIR.glob("*.csv")):
        if path.name == "summary.csv":
            continue
        frame = pd.read_csv(path)
        if not {"model", "accuracy", "macro_f1"}.issubset(frame.columns):
            continue
        protocol = "within-subject" if path.name.startswith("within_subject") else "cross-subject"
        frame = frame.assign(protocol=protocol)
        raw_frames.append(frame)
        rows.append(
            {
                "protocol": protocol,
                "model": str(frame["model"].iloc[0]),
                "n_folds": len(frame),
                "accuracy_mean": frame["accuracy"].mean(),
                "accuracy_std": frame["accuracy"].std(ddof=1),
                "macro_f1_mean": frame["macro_f1"].mean(),
                "macro_f1_std": frame["macro_f1"].std(ddof=1),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No experiment CSV files found in {RESULTS_DIR}")

    summary = pd.DataFrame(rows).sort_values(["protocol", "accuracy_mean"], ascending=[True, False])
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False))

    comparison = pd.concat(raw_frames, ignore_index=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # Protocols contain different model sets. Separate panels prevent missing
    # experiments from being misinterpreted as zero-valued observations.
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    protocol_titles = {
        "within-subject": "Within-subject (cross-session)",
        "cross-subject": "Cross-subject (LOSO)",
    }
    for axis, protocol in zip(axes, ("within-subject", "cross-subject"), strict=True):
        protocol_data = comparison[comparison["protocol"] == protocol]
        order = protocol_data.groupby("model")["accuracy"].mean().sort_values(ascending=False).index
        sns.barplot(
            data=protocol_data,
            x="model",
            y="accuracy",
            order=order,
            errorbar="sd",
            color="#287f8e" if protocol == "within-subject" else "#df7434",
            ax=axis,
        )
        axis.axhline(0.25, color="#555555", linestyle="--", linewidth=1, label="Chance (25%)")
        axis.set_title(protocol_titles[protocol])
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=25)
        axis.legend(loc="upper right")

    axes[0].set_ylabel("Accuracy")
    axes[1].set_ylabel("")
    axes[0].set_ylim(0.0, 1.0)
    figure.suptitle("EEG motor imagery model comparison")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "model_comparison.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()

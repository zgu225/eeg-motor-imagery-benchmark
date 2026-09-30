"""Predict filtered, epoched EEG from a trusted local model bundle."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bci_project.artifacts import predict_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="NPZ with X, channels, sfreq; units must be volts")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        predicted = predict_bundle(args.bundle, data["X"], data["channels"].tolist(), float(data["sfreq"]))
    meta = json.loads((args.bundle / "input.json").read_text())
    names = {v: k for k, v in meta["label_mapping"].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": predicted, "label": [names[int(i)] for i in predicted]}).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

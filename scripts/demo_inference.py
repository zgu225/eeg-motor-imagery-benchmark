"""Replay real held-out epochs using a saved model and the standalone CLI."""
import json
from pathlib import Path
import subprocess
import sys

import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bci_project.data.load_moabb import load_bnci2014_001


def main():
    mne.set_log_level('ERROR')
    bundle = ROOT / 'reports/benchmark_v2/cross_session/csp_lda/seed_42/subject_1'
    meta = json.loads((bundle / 'input.json').read_text())
    settings = meta['preprocessing']
    data = load_bnci2014_001(subjects=[1], **{k: settings[k] for k in ('tmin','tmax','fmin','fmax')})
    test = data.X[data.sessions == sorted(set(data.sessions))[1]]
    directory = ROOT / 'outputs/demo'
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / 'epochs.npz', X=test, channels=np.array(data.channels), sfreq=data.sfreq)
    subprocess.run([sys.executable, str(ROOT / 'scripts/predict.py'), '--bundle', str(bundle),
                    '--input', str(directory / 'epochs.npz'), '--output', str(directory / 'predictions.csv')], check=True)
    expected = pd.read_csv(bundle / 'predictions.csv').y_pred.to_numpy()
    actual = pd.read_csv(directory / 'predictions.csv').prediction.to_numpy()
    np.testing.assert_array_equal(actual, expected)
    print(f'CLI replay verified on {len(actual)} real held-out epochs: {directory}')


if __name__ == '__main__':
    main()

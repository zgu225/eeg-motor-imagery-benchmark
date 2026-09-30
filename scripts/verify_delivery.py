"""Audit delivered artifacts without retraining or downloading EEG."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bci_project.benchmark import complete_fold, sha256


def main():
    directory = ROOT / 'reports/benchmark_v2'
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, digest in manifest['source'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Source differs from recorded experiment: {name}')
    subjects = manifest['config']['dataset'].get('subjects') or list(range(1, 10))
    count = 0
    for protocol in manifest['protocols']:
        for seed in manifest['seeds']:
            for subject in subjects:
                reference = None
                for model in manifest['models']:
                    fold = directory / f'{protocol}/{model}/seed_{seed}/subject_{subject}'
                    if not complete_fold(fold):
                        raise ValueError(f'Incomplete fold: {fold}')
                    with np.load(fold / 'split.npz') as saved:
                        splits = {name: saved[name].copy() for name in saved.files}
                    for a, b in [('fit','validation'), ('fit','test'), ('validation','test')]:
                        if set(splits[a]) & set(splits[b]):
                            raise ValueError(f'Overlapping splits: {fold}')
                    if reference is not None:
                        for name in splits:
                            np.testing.assert_array_equal(reference[name], splits[name])
                    reference = splits
                    predictions = pd.read_csv(fold / 'predictions.csv')
                    np.testing.assert_array_equal(predictions.trial_index, splits['test'])
                    metrics = json.loads((fold / 'metrics.json').read_text())
                    np.testing.assert_allclose(metrics['accuracy'], accuracy_score(predictions.y_true, predictions.y_pred))
                    np.testing.assert_allclose(metrics['macro_f1'], f1_score(predictions.y_true, predictions.y_pred, average='macro'))
                    expected = confusion_matrix(predictions.y_true, predictions.y_pred, labels=[0,1,2,3])
                    np.testing.assert_array_equal(expected, json.loads((fold / 'confusion.json').read_text()))
                    count += 1
    print(f'Delivery audit passed: {count} folds, matching sources, checksums, disjoint and shared splits, predictions and metrics.')


if __name__ == '__main__':
    main()

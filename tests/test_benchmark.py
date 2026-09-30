"""Regression coverage for scientific splits and persisted inference."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bci_project.artifacts import save_bundle, predict_bundle
from bci_project.benchmark import split_indices, complete_fold, write_json, sha256
from bci_project.data.load_moabb import EEGDataset
from bci_project.data.preprocessing import standardize_epochs
from bci_project.models.deep_models import build_deep_model
from bci_project.training import train_model, predict_model


class BenchmarkTests(unittest.TestCase):
    def test_microvolt_scale_and_constant_channels(self):
        values = np.random.default_rng(1).normal(size=(4, 3, 128)).astype('float32')
        np.testing.assert_allclose(standardize_epochs(values), standardize_epochs(values * 1e-6), atol=1e-6)
        self.assertTrue(np.isfinite(standardize_epochs(np.ones((2, 3, 128)))).all())

    def test_grouped_splits(self):
        rows = np.array([(s, sess, r, y) for s in range(1, 5) for sess in range(2)
                         for r in range(3) for y in range(4)])
        data = EEGDataset(np.zeros((len(rows), 2, 128)), rows[:, 3], rows[:, 0],
                          rows[:, 1], ['C3', 'C4'], 250., dict(a=0,b=1,c=2,d=3), rows[:, 2])
        for protocol in ('cross_session', 'loso'):
            fit, val, test = split_indices(data, protocol, 1, 42)
            self.assertFalse(set(fit) & set(val) or set(fit) & set(test) or set(val) & set(test))
            if protocol == 'loso':
                self.assertFalse(set(data.subjects[fit]) & set(data.subjects[val]))
                self.assertNotIn(1, data.subjects[fit])
                self.assertNotIn(1, data.subjects[val])
            else:
                self.assertFalse(set(data.runs[fit]) & set(data.runs[val]))
                self.assertFalse(set(data.sessions[fit]) & set(data.sessions[test]))

    def test_model_initialization_does_not_update_batchnorm(self):
        for name in ('eegnet', 'lawhern_eegnet', 'shallow_convnet'):
            model = build_deep_model(name, 2, 128, 4, .5)
            for layer in model.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    self.assertEqual(layer.num_batches_tracked.item(), 0)
                    self.assertTrue(torch.all(layer.running_var == 1))

    def test_train_save_reload_and_input_contract(self):
        torch.set_num_threads(2)
        X = np.random.default_rng(2).normal(size=(16, 2, 128)).astype('float32') * 1e-6
        y = np.tile(np.arange(4), 4)
        model = build_deep_model('lawhern_eegnet', 2, 128, 4, .5)
        history = train_model(model, standardize_epochs(X[:12]), y[:12], epochs=2,
                              validation_data=(standardize_epochs(X[12:]), y[12:]))
        self.assertEqual(len(history.train_loss), 2)
        meta = dict(model='lawhern_eegnet', n_channels=2, n_times=128, channels=['C3','C4'],
                    sfreq=250., label_mapping=dict(a=0,b=1,c=2,d=3), dropout=.5)
        with tempfile.TemporaryDirectory() as directory:
            save_bundle(model, directory, meta)
            np.testing.assert_array_equal(predict_bundle(directory, X, ['C3','C4'], 250),
                                          predict_model(model, standardize_epochs(X)))
            with self.assertRaises(ValueError):
                predict_bundle(directory, X, ['C4','C3'], 250)
            with self.assertRaises(ValueError):
                predict_bundle(directory, X * np.nan, ['C3','C4'], 250)

    def test_resume_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_json(directory / 'metrics.json', dict(accuracy=.5))
            write_json(directory / 'complete.json', dict(checksums={'metrics.json': sha256(directory / 'metrics.json')}))
            self.assertTrue(complete_fold(directory))
            write_json(directory / 'metrics.json', dict(accuracy=.1))
            with self.assertRaises(ValueError):
                complete_fold(directory)

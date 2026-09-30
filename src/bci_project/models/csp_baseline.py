"""Classical Common Spatial Patterns classification baselines."""

from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def build_csp_lda(n_components: int = 8) -> Pipeline:
    """Construct a CSP feature extractor followed by LDA.

    CSP produces log-variance spatial features, and LDA performs classification
    in the resulting low-dimensional feature space.
    """

    return Pipeline(
        steps=[
            ("csp", CSP(n_components=n_components, reg=None, log=True, norm_trace=False)),
            ("classifier", LinearDiscriminantAnalysis()),
        ]
    )


def build_csp_svm(n_components: int = 8, c: float = 1.0) -> Pipeline:
    """Construct a CSP feature extractor followed by an RBF-SVM."""

    return Pipeline(
        steps=[
            ("csp", CSP(n_components=n_components, reg=None, log=True, norm_trace=False)),
            ("classifier", SVC(C=c, kernel="rbf", gamma="scale")),
        ]
    )


def fit_predict_baseline(
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Fit a scikit-learn pipeline and return test-set predictions."""

    model.fit(X_train, y_train)
    return model.predict(X_test)

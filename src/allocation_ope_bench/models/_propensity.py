"""Utility sklearn-compatible propensity wrappers."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ConstantPropensityModel(BaseEstimator, ClassifierMixin):
    """Always predicts a fixed P(T=1) for use with XLearner when propensity is known.

    Used for RCT datasets where the true propensity is the design probability.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        n = len(X)
        proba = np.empty((n, 2))
        proba[:, 0] = 1 - self.p
        proba[:, 1] = self.p
        return proba

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

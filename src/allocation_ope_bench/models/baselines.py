"""Simple baseline uplift estimators via scikit-uplift.

Wrapped estimators
------------------
- ClassTransformationEstimator — class-variable transformation (Lai/Kane)
- TwoModelEstimator            — two-model (T-learner) via scikit-uplift
- SoloModelEstimator           — S-learner via scikit-uplift
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from allocation_ope_bench.models.base import UpliftEstimator
from allocation_ope_bench.models.base_learners import make_base_learner


class ClassTransformationEstimator(UpliftEstimator):
    """Class-variable transformation (Lai 2006 / Kane 2014).

    Requires binary outcomes; transforms the problem into a single
    classification task: Z = Y*T + (1-Y)*(1-T), then CATE ≈ 2*P(Z=1|X) - 1.

    Uses scikit-uplift's ClassTransformation wrapper.
    """

    name = "class_transformation"

    def __init__(
        self,
        base_learner: Literal["lightgbm", "xgboost"] = "lightgbm",
        seed: int = 42,
        **hparams: Any,
    ):
        self.base_learner = base_learner
        self.seed = seed
        self.hparams = hparams
        self._model = None

    def fit(self, X, treatment, outcome, propensity=None):
        from sklift.models import ClassTransformation

        clf = make_base_learner(self.base_learner, "classification", self.seed, **self.hparams)
        X_arr, t_arr, y_arr, _ = self._to_numpy(X, treatment, outcome, propensity)
        self._model = ClassTransformation(estimator=clf)
        self._model.fit(X_arr, y_arr, t_arr)
        return self

    def predict_uplift(self, X):
        X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        return self._model.predict(X_arr)


class TwoModelEstimator(UpliftEstimator):
    """Two-model (T-learner) baseline via scikit-uplift.

    Fits separate classifiers for treated and control, returns P(Y=1|T=1,X) - P(Y=1|T=0,X).
    Equivalent to a T-learner with classification base learners.
    """

    name = "two_model"

    def __init__(
        self,
        base_learner: Literal["lightgbm", "xgboost"] = "lightgbm",
        seed: int = 42,
        **hparams: Any,
    ):
        self.base_learner = base_learner
        self.seed = seed
        self.hparams = hparams
        self._model = None

    def fit(self, X, treatment, outcome, propensity=None):
        from copy import deepcopy

        from sklift.models import TwoModels

        clf = make_base_learner(self.base_learner, "classification", self.seed, **self.hparams)
        X_arr, t_arr, y_arr, _ = self._to_numpy(X, treatment, outcome, propensity)
        self._model = TwoModels(
            estimator_trmnt=clf,
            estimator_ctrl=deepcopy(clf),
            method="vanilla",
        )
        self._model.fit(X_arr, y_arr, t_arr)
        return self

    def predict_uplift(self, X):
        X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        return self._model.predict(X_arr)


def _uniform_hash(X: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic Uniform(0,1) score per row, as a pure function of (row, seed).

    A hash of the row's raw bytes (FNV-style fold, then a splitmix64 finalizer for
    avalanche) mapped into [0, 1). Vectorized over rows.
    """
    A = np.ascontiguousarray(X, dtype=np.float64)
    # -0.0 and 0.0 must hash alike, or the score would depend on a sign bit that
    # carries no information.
    A = A + 0.0
    words = A.view(np.uint64).reshape(A.shape[0], -1)
    key = np.full(A.shape[0], np.uint64(0xCBF29CE484222325) ^ np.uint64(seed), dtype=np.uint64)
    prime = np.uint64(0x100000001B3)
    for j in range(words.shape[1]):
        key = (key ^ words[:, j]) * prime
    key = (key ^ (key >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    key = (key ^ (key >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    key = key ^ (key >> np.uint64(31))
    # Top 53 bits -> a double in [0, 1), the standard uint64->float64 conversion.
    return (key >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


class RandomUpliftEstimator(UpliftEstimator):
    """Random-score baseline — assigns Uniform(0,1) uplift scores, independent of
    the outcome, as a deterministic *function of the covariate row*.

    Used as a lower-bound candidate policy in RQ4 selection benchmarks so that
    estimators have at least one clearly-suboptimal policy to distinguish from
    learned policies.

    The score must be a function of ``x``, not a fresh draw per call. An earlier
    implementation held a stateful RNG advanced on every ``predict_uplift``; since
    ``AllocationPolicy.action_prob`` calls ``score`` internally, the logging policy,
    the target policy and the truth label of this candidate were each built from a
    *different* random score. That silently turned the self-aligned logger into a
    candidate-independent one for this candidate alone (ESS fraction 0.09 instead
    of 0.63) and mismatched its truth label. ``test_random_policy_score_is_stable``
    guards the invariant.
    """

    name = "random"

    def __init__(self, seed: int = 42, **_: object):
        self.seed = seed

    def fit(self, X, treatment, outcome, propensity=None):
        return self

    def predict_uplift(self, X):
        X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        return _uniform_hash(X_arr, self.seed)


class SoloModelEstimator(UpliftEstimator):
    """S-learner (Solo Model) baseline via scikit-uplift.

    Includes treatment as a feature in a single classifier.
    """

    name = "solo_model"

    def __init__(
        self,
        base_learner: Literal["lightgbm", "xgboost"] = "lightgbm",
        seed: int = 42,
        **hparams: Any,
    ):
        self.base_learner = base_learner
        self.seed = seed
        self.hparams = hparams
        self._model = None

    def fit(self, X, treatment, outcome, propensity=None):
        from sklift.models import SoloModel

        clf = make_base_learner(self.base_learner, "classification", self.seed, **self.hparams)
        X_arr, t_arr, y_arr, _ = self._to_numpy(X, treatment, outcome, propensity)
        self._model = SoloModel(estimator=clf)
        self._model.fit(X_arr, y_arr, t_arr)
        return self

    def predict_uplift(self, X):
        X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        return self._model.predict(X_arr)

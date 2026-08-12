"""EconML / scikit-uplift meta-learner wrappers.

All wrappers accept the same ``base_learner`` name and ``seed`` so that
the meta-strategy (not the base learner) drives performance differences.

Wrapped estimators
------------------
- SLearnerEstimator   — S-learner (EconML)
- TLearnerEstimator   — T-learner (EconML)
- XLearnerEstimator   — X-learner (EconML)
- RLearnerEstimator   — R-learner / NonParamDML (EconML)
- DRLearnerEstimator  — Doubly-robust learner (EconML)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from allocation_ope_bench.models.base import UpliftEstimator
from allocation_ope_bench.models.base_learners import make_base_learner


class SLearnerEstimator(UpliftEstimator):
    """S-learner: single model of E[Y | X, T], uplift = f(X,1) - f(X,0)."""

    name = "s_learner"

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
        from econml.metalearners import SLearner

        model = make_base_learner(self.base_learner, "regression", self.seed, **self.hparams)
        X_arr, t_arr, y_arr, _ = self._to_numpy(X, treatment, outcome, propensity)
        self._model = SLearner(overall_model=model)
        self._model.fit(y_arr, t_arr, X=X_arr)
        return self

    def predict_uplift(self, X):
        X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        return self._model.effect(X_arr)


class TLearnerEstimator(UpliftEstimator):
    """T-learner: separate models for treated and control, uplift = m1(X) - m0(X)."""

    name = "t_learner"

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
        from econml.metalearners import TLearner

        model = make_base_learner(self.base_learner, "regression", self.seed, **self.hparams)
        X_arr, t_arr, y_arr, _ = self._to_numpy(X, treatment, outcome, propensity)
        self._model = TLearner(models=model)
        self._model.fit(y_arr, t_arr, X=X_arr)
        return self

    def predict_uplift(self, X):
        X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        return self._model.effect(X_arr)


class XLearnerEstimator(UpliftEstimator):
    """X-learner (Künzel et al. 2019) via EconML."""

    name = "x_learner"

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
        from econml.metalearners import XLearner

        model = make_base_learner(self.base_learner, "regression", self.seed, **self.hparams)
        X_arr, t_arr, y_arr, p_arr = self._to_numpy(X, treatment, outcome, propensity)

        if p_arr is not None:
            # Wrap the known propensity as a constant sklearn estimator so EconML
            # can use it internally for the τ0/τ1 combination step.
            from allocation_ope_bench.models._propensity import ConstantPropensityModel

            prop_model = ConstantPropensityModel(float(p_arr.mean()))
        else:
            prop_model = None

        kwargs = {"models": model}
        if prop_model is not None:
            kwargs["propensity_model"] = prop_model
        self._model = XLearner(**kwargs)
        self._model.fit(y_arr, t_arr, X=X_arr)
        return self

    def predict_uplift(self, X):
        X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        return self._model.effect(X_arr)


class RLearnerEstimator(UpliftEstimator):
    """R-learner / NonParamDML (Robinson 1988, Nie & Wager 2021) via EconML."""

    name = "r_learner"

    def __init__(
        self,
        base_learner: Literal["lightgbm", "xgboost"] = "lightgbm",
        seed: int = 42,
        cv: int = 3,
        **hparams: Any,
    ):
        self.base_learner = base_learner
        self.seed = seed
        self.cv = cv
        self.hparams = hparams
        self._model = None

    def fit(self, X, treatment, outcome, propensity=None):
        from econml.dml import NonParamDML

        reg = make_base_learner(self.base_learner, "regression", self.seed, **self.hparams)
        # NonParamDML needs model_t to model the treatment; use a copy of the regressor
        # (binary treatment → treated as continuous residual; discrete_treatment=True handles it)
        from copy import deepcopy

        X_arr, t_arr, y_arr, _ = self._to_numpy(X, treatment, outcome, propensity)
        self._model = NonParamDML(
            model_y=reg,
            model_t=deepcopy(reg),
            model_final=deepcopy(reg),
            cv=self.cv,
            discrete_treatment=True,
            random_state=self.seed,
        )
        self._model.fit(y_arr, t_arr, X=X_arr)
        return self

    def predict_uplift(self, X):
        X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        return self._model.effect(X_arr)


class DRLearnerEstimator(UpliftEstimator):
    """Doubly-robust learner (Kennedy 2020) via EconML."""

    name = "dr_learner"

    def __init__(
        self,
        base_learner: Literal["lightgbm", "xgboost"] = "lightgbm",
        seed: int = 42,
        cv: int = 3,
        **hparams: Any,
    ):
        self.base_learner = base_learner
        self.seed = seed
        self.cv = cv
        self.hparams = hparams
        self._model = None

    def fit(self, X, treatment, outcome, propensity=None):
        from copy import deepcopy

        from econml.dr import DRLearner

        reg = make_base_learner(self.base_learner, "regression", self.seed, **self.hparams)
        clf = make_base_learner(self.base_learner, "classification", self.seed, **self.hparams)
        X_arr, t_arr, y_arr, _ = self._to_numpy(X, treatment, outcome, propensity)
        self._model = DRLearner(
            model_propensity=clf,
            model_regression=deepcopy(reg),
            model_final=deepcopy(reg),
            cv=self.cv,
            # Without random_state EconML's internal cross-fitting folds are drawn
            # unseeded, so refitting with the same seed yields different scores --
            # which makes the fitted policy (and hence its true value) irreproducible.
            random_state=self.seed,
        )
        self._model.fit(y_arr, t_arr, X=X_arr)
        return self

    def predict_uplift(self, X):
        X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        return self._model.effect(X_arr)

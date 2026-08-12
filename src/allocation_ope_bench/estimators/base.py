"""Core types for off-policy evaluation of allocation policies.

The OPE problem here: a deterministic (or smoothed) **target** allocation policy
``pi_e`` assigns a treat/withhold action to each unit under a budget; we observe
**logged** bandit feedback (context, the action actually taken under a known
logging policy ``pi_b``, the realized reward), and want the value the target
policy *would* have achieved, ``V(pi_e) = E[Y(a_pi_e(X))]``.

Binary action space {0, 1} (withhold / treat). For a unit i the per-action
probabilities are derived from a single "prob of treatment" vector:

    pi(a_i | x_i) = p_treat_i        if a_i == 1
                  = 1 - p_treat_i    if a_i == 0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class LoggedData:
    """Logged bandit feedback under a known logging policy pi_b."""

    context: np.ndarray  # (n, p)
    action: np.ndarray  # (n,) in {0, 1}
    reward: np.ndarray  # (n,)
    pscore: np.ndarray  # (n,) = pi_b(action_i | x_i), strictly in (0, 1]
    logging_prob_treat: np.ndarray | None = None  # (n,) = pi_b(1 | x_i); for diagnostics
    name: str = "logged"

    def __post_init__(self) -> None:
        self.context = np.asarray(self.context, dtype=float)
        self.action = np.asarray(self.action).astype(int)
        self.reward = np.asarray(self.reward, dtype=float)
        self.pscore = np.asarray(self.pscore, dtype=float)
        n = self.context.shape[0]
        for nm, arr in (("action", self.action), ("reward", self.reward), ("pscore", self.pscore)):
            if arr.shape != (n,):
                raise ValueError(f"{nm} must have shape ({n},), got {arr.shape}")
        if not set(np.unique(self.action).tolist()).issubset({0, 1}):
            raise ValueError("action must be binary 0/1")
        if np.any(self.pscore <= 0) or np.any(self.pscore > 1):
            raise ValueError("pscore must lie in (0, 1]")
        if self.logging_prob_treat is not None:
            self.logging_prob_treat = np.asarray(self.logging_prob_treat, dtype=float)
            if self.logging_prob_treat.shape != (n,):
                raise ValueError(f"logging_prob_treat must have shape ({n},)")

    @property
    def n(self) -> int:
        return self.context.shape[0]

    def subset(self, idx) -> "LoggedData":
        """Restrict to row indices (used for cross-fitting folds)."""
        idx = np.asarray(idx)
        lpt = None if self.logging_prob_treat is None else self.logging_prob_treat[idx]
        return LoggedData(
            context=self.context[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            pscore=self.pscore[idx],
            logging_prob_treat=lpt,
            name=self.name,
        )


@dataclass
class EstimateResult:
    estimator: str
    value: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> dict:
        return {
            "estimator": self.estimator,
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def policy_action_prob(prob_treat: np.ndarray, action: np.ndarray) -> np.ndarray:
    """pi(a_i | x_i) given P(treat | x_i) and the action a_i."""
    return np.where(action == 1, prob_treat, 1.0 - prob_treat)


class OutcomeModel:
    """Shared outcome regressor mu_hat(x, a) for DM / DR / Switch-DR.

    Two-model form: separate base learners for treated and control arms, fit on
    the logged data only. A single instance is shared across all estimators in a
    run so they reference an identical mu_hat (fairness of comparison).
    """

    def __init__(self, base_learner: str = "lightgbm", seed: int = 42):
        self.base_learner = base_learner
        self.seed = seed
        self._m1 = None
        self._m0 = None

    def fit(self, logged: LoggedData) -> "OutcomeModel":
        from allocation_ope_bench.models.base_learners import make_base_learner

        treated = logged.action == 1
        control = ~treated
        self._m1 = make_base_learner(self.base_learner, "regression", self.seed)
        self._m0 = make_base_learner(self.base_learner, "regression", self.seed)
        # Guard tiny arms (smoke data): fall back to the global mean predictor.
        self._m1 = _fit_or_constant(
            self._m1, logged.context[treated], logged.reward[treated], fallback=logged.reward.mean()
        )
        self._m0 = _fit_or_constant(
            self._m0, logged.context[control], logged.reward[control], fallback=logged.reward.mean()
        )
        return self

    def predict(self, X: np.ndarray, action: int) -> np.ndarray:
        model = self._m1 if action == 1 else self._m0
        return _predict(model, X)


class OutOfFoldOutcomeModel(OutcomeModel):
    """``mu_hat`` whose prediction for each logged row comes from a fold that excludes it.

    Drop-in for ``OutcomeModel``: every estimator in this package calls
    ``predict(logged.context, a)`` on the *same* rows the model was fit on, so K-fold
    out-of-fold predictions can simply be precomputed and served back by row position.
    Passing this instead of the plain model is what turns the main RQ1 comparison into
    an honest-nuisance one -- DM's predictions no longer see each unit's own outcome,
    and the DR residuals are no longer shrunk in-sample.

    ``predict`` refuses inputs that are not the fitted rows rather than silently
    refitting, since a wrong-row prediction would quietly reintroduce the leakage this
    class exists to remove.
    """

    def __init__(self, base_learner: str = "lightgbm", seed: int = 42, n_folds: int = 5):
        super().__init__(base_learner=base_learner, seed=seed)
        self.n_folds = n_folds
        self._oof: dict[int, np.ndarray] = {}
        self._fit_context: np.ndarray | None = None

    def fit(self, logged: LoggedData) -> "OutOfFoldOutcomeModel":
        n = logged.n
        self._fit_context = logged.context
        rng = np.random.default_rng(self.seed)
        folds = np.array_split(rng.permutation(n), self.n_folds)
        self._oof = {1: np.empty(n), 0: np.empty(n)}
        for f in folds:
            mask = np.ones(n, bool)
            mask[f] = False
            inner = OutcomeModel(self.base_learner, self.seed).fit(logged.subset(np.where(mask)[0]))
            for a in (0, 1):
                self._oof[a][f] = inner.predict(logged.context[f], a)
        # A full-sample fit backs the (unused by our estimators) off-row path.
        super().fit(logged)
        return self

    def predict(self, X: np.ndarray, action: int) -> np.ndarray:
        if self._fit_context is not None and X.shape == self._fit_context.shape:
            return self._oof[int(action)]
        raise ValueError(
            "OutOfFoldOutcomeModel serves out-of-fold predictions only for the rows it "
            "was fit on; got an array of shape "
            f"{X.shape} against fitted {None if self._fit_context is None else self._fit_context.shape}"
        )


class _Constant:
    def __init__(self, value: float):
        self.value = float(value)


def _fit_or_constant(model, X, y, fallback):
    if X.shape[0] < 5 or np.unique(y).size < 2:
        return _Constant(y.mean() if X.shape[0] > 0 else fallback)
    model.fit(X, y)
    return model


def _predict(model, X):
    if isinstance(model, _Constant):
        return np.full(X.shape[0], model.value)
    return np.asarray(model.predict(X), dtype=float)


class OPEEstimator(ABC):
    """Common interface: estimate V(pi_e) + a bootstrap CI from logged data."""

    name: str = "base"

    @abstractmethod
    def estimate(
        self,
        logged: LoggedData,
        target_prob_treat: np.ndarray,
        outcome_model: OutcomeModel | None = None,
        n_bootstrap: int = 500,
        seed: int = 42,
    ) -> EstimateResult:
        """Return the value estimate and its bootstrap CI.

        target_prob_treat : pi_e(treat | x_i) for each logged unit, shape (n,).
        outcome_model     : a shared, already-fit OutcomeModel (DM/DR/Switch).
        """

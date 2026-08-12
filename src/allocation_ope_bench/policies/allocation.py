"""Budget-constrained allocation policies (the thing being evaluated).

The policy scores units by predicted uplift (reusing Paper 1's models, fit on
training data only) and allocates the treatment under a budget. Three variants:

* ``deterministic`` — hard top-k under budget (the object of study; zero overlap
  on the withheld region).
* ``softmax``       — smoothed: P(treat | x) = sigmoid((score - c) / temperature)
  with the threshold c calibrated so expected spend == budget. Restores overlap.
* ``epsilon``       — eps-perturbed deterministic: mixes the hard rule with a
  base-rate-k coin flip, a minimal overlap fix.

``action_prob`` returns pi_e(treat | x) for the supplied contexts; the budget
threshold is computed over exactly those contexts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from allocation_ope_bench.data.ground_truth import allocate_under_budget


class AllocationPolicy:
    def __init__(
        self,
        uplift_model: str = "t_learner",
        variant: str = "deterministic",
        base_learner: str = "lightgbm",
        temperature: float = 1.0,
        epsilon: float = 0.1,
        seed: int = 42,
    ):
        if variant not in {"deterministic", "softmax", "epsilon"}:
            raise ValueError(f"unknown variant {variant!r}")
        self.uplift_model = uplift_model
        self.variant = variant
        self.base_learner = base_learner
        self.temperature = temperature
        self.epsilon = epsilon
        self.seed = seed
        self._model = None

    def fit(self, train) -> "AllocationPolicy":
        """Fit the uplift/score model on a training Dataset (train split only)."""
        from allocation_ope_bench.models import get_estimator

        self._model = get_estimator(
            self.uplift_model, base_learner=self.base_learner, seed=self.seed
        )
        Xdf = pd.DataFrame(train.X, columns=train.feature_names)
        self._model.fit(Xdf, pd.Series(train.treatment), pd.Series(train.outcome))
        return self

    def score(self, X: np.ndarray, feature_names=None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("policy not fit")
        cols = feature_names or [f"x{i}" for i in range(X.shape[1])]
        return np.asarray(self._model.predict_uplift(pd.DataFrame(X, columns=cols)), dtype=float)

    def action_prob(
        self,
        X: np.ndarray,
        budget_k: float,
        costs: np.ndarray | None = None,
        feature_names=None,
    ) -> np.ndarray:
        """pi_e(treat | x_i) for the given contexts under budget_k."""
        scores = self.score(X, feature_names=feature_names)
        n = X.shape[0]
        costs = np.ones(n) if costs is None else np.asarray(costs, dtype=float)

        if self.variant == "deterministic":
            return allocate_under_budget(scores, costs, budget_k).astype(float)

        if self.variant == "epsilon":
            hard = allocate_under_budget(scores, costs, budget_k).astype(float)
            return (1.0 - self.epsilon) * hard + self.epsilon * budget_k

        # softmax: calibrate threshold c so expected spend == budget_k * total cost.
        return self._softmax_probs(scores, costs, budget_k)

    def _softmax_probs(self, scores, costs, budget_k) -> np.ndarray:
        if budget_k <= 0.0:
            return np.zeros_like(scores)
        if budget_k >= 1.0:
            return np.ones_like(scores)
        target_spend = budget_k * costs.sum()
        s = (scores - scores.mean()) / (scores.std() + 1e-12)

        def spend(c: float) -> float:
            p = 1.0 / (1.0 + np.exp(-(s - c) / self.temperature))
            return float((p * costs).sum())

        # Bisection on the threshold c (spend is monotone decreasing in c).
        lo, hi = -50.0, 50.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if spend(mid) > target_spend:
                lo = mid
            else:
                hi = mid
        c = 0.5 * (lo + hi)
        return 1.0 / (1.0 + np.exp(-(s - c) / self.temperature))

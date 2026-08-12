"""Doubly-robust estimation with optimal weight shrinkage (Su et al., 2020).

The DRos ("optimistic shrinkage") estimator replaces the importance weight ``w`` in the
DR correction term with

    w_lam = lam * w / (w**2 + lam),        lam in [0, inf)

which interpolates between DM (``lam = 0``, correction removed) and plain DR
(``lam -> inf``, since ``w_lam -> w``). ``lam`` is chosen on the logged sample by
minimizing the estimated MSE bound from the paper: the squared bias introduced by
shrinking (the mean of ``(w - w_lam) * residual``) plus the variance of the shrunk
per-unit contributions. This mirrors exactly the tuning style used for Switch-DR's
``tau`` and mIPS's ``alpha`` (estimated-MSE proxy on the logged sample), so no estimator
gets a hand-picked advantage.

Included because the paper cites shrinkage-DR as the other standard tail-control
response to heavy weights and, until this estimator existed, stated only a reasoned
expectation about it. The sanity test ``lam -> inf`` recovering plain DR bit-for-bit is
in ``tests/test_estimators.py``.
"""

from __future__ import annotations

import numpy as np

from allocation_ope_bench.estimators.base import EstimateResult, OPEEstimator, OutcomeModel
from allocation_ope_bench.estimators.doubly_robust import _dr_terms
from allocation_ope_bench.metrics.stats import bootstrap_ci


class ShrinkageDR(OPEEstimator):
    name = "shrinkage_dr"
    # log-spaced grid spanning "almost DM" to "effectively plain DR" for weights <= 50.
    DEFAULT_LAMBDA_GRID = (0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 5000.0, float("inf"))

    def __init__(self, lam="auto", lambda_grid=DEFAULT_LAMBDA_GRID):
        self.lam = lam
        self.lambda_grid = tuple(lambda_grid)
        self.selected_lambda: float | None = None

    @staticmethod
    def _shrink(w: np.ndarray, lam: float) -> np.ndarray:
        if np.isinf(lam):
            return w
        if lam == 0.0:
            return np.zeros_like(w)
        return lam * w / (w**2 + lam)

    def estimate(self, logged, target_prob_treat, outcome_model=None, n_bootstrap=500, seed=42):
        if outcome_model is None:
            outcome_model = OutcomeModel(seed=seed).fit(logged)
        dm, w, residual = _dr_terms(logged, target_prob_treat, outcome_model)
        lam = self._select_lambda(dm, w, residual) if self.lam == "auto" else float(self.lam)
        self.selected_lambda = lam
        phi = dm + self._shrink(w, lam) * residual
        point, lo, hi = bootstrap_ci(np.mean, phi, n_bootstrap=n_bootstrap, seed=seed)
        return EstimateResult(self.name, point, lo, hi)

    def _select_lambda(self, dm, w, residual) -> float:
        n = len(w)
        best_lam, best_mse = self.lambda_grid[0], float("inf")
        for lam in self.lambda_grid:
            w_lam = self._shrink(w, lam)
            phi = dm + w_lam * residual
            var = float(np.var(phi, ddof=1)) / n if n > 1 else float(np.var(phi))
            bias = float(np.mean((w - w_lam) * residual))
            mse = bias**2 + var
            if mse < best_mse:
                best_lam, best_mse = lam, mse
        return best_lam

"""Doubly-robust value estimators: DR and Switch-DR."""

from __future__ import annotations

import numpy as np

from allocation_ope_bench.estimators.base import (
    EstimateResult,
    OPEEstimator,
    OutcomeModel,
)
from allocation_ope_bench.estimators.ips import _importance_weight
from allocation_ope_bench.metrics.stats import bootstrap_ci


def _dr_terms(logged, target_prob_treat, outcome_model):
    """Return (dm_term, weight, residual) per unit for the DR family."""
    mu1 = outcome_model.predict(logged.context, 1)
    mu0 = outcome_model.predict(logged.context, 0)
    dm = target_prob_treat * mu1 + (1.0 - target_prob_treat) * mu0
    mu_obs = np.where(logged.action == 1, mu1, mu0)
    w = _importance_weight(logged, target_prob_treat)
    residual = logged.reward - mu_obs
    return dm, w, residual


def dr_contributions(logged, target_prob_treat, outcome_model) -> np.ndarray:
    """Per-unit DR value contribution phi_i = dm_i + w_i (Y_i - mu_hat(x_i, a_i)).

    Shared by the perturbation-smoothed and cross-fitted DR estimators.
    """
    dm, w, residual = _dr_terms(logged, target_prob_treat, outcome_model)
    return dm + w * residual


class DoublyRobust(OPEEstimator):
    """V = mean_i [ dm_i + w_i (Y_i - mu_hat(x_i, a_i)) ].

    Unbiased if *either* the outcome model or the propensity is correct; the
    correction undoes DM's model bias on the supported region."""

    name = "dr"

    def estimate(self, logged, target_prob_treat, outcome_model=None, n_bootstrap=500, seed=42):
        if outcome_model is None:
            outcome_model = OutcomeModel(seed=seed).fit(logged)
        dm, w, residual = _dr_terms(logged, target_prob_treat, outcome_model)
        phi = dm + w * residual
        point, lo, hi = bootstrap_ci(np.mean, phi, n_bootstrap=n_bootstrap, seed=seed)
        return EstimateResult(self.name, point, lo, hi)


class SwitchDR(OPEEstimator):
    """Switch-DR: drop the IPS correction where the weight exceeds ``tau`` and
    fall back to the DM term there, capping variance from extreme weights
    (Wang et al. 2017).

    ``tau`` is **not** hard-coded: by default ("auto") it is selected on the
    logged sample by minimizing an estimated-MSE proxy (switching-bias bound ^2 +
    variance) over ``tau_grid`` (the standard Switch-DR MSE-bound selection) — the
    same fair, data-driven scheme used for BIPS's alpha. A fixed float or a grid
    (for a sensitivity sweep) may also be supplied."""

    name = "switch_dr"
    DEFAULT_TAU_GRID = (5.0, 10.0, 20.0, 50.0, 100.0, float("inf"))

    def __init__(self, tau="auto", tau_grid=DEFAULT_TAU_GRID):
        self.tau = tau
        self.tau_grid = tuple(tau_grid)
        self.selected_tau: float | None = None

    def estimate(self, logged, target_prob_treat, outcome_model=None, n_bootstrap=500, seed=42):
        if outcome_model is None:
            outcome_model = OutcomeModel(seed=seed).fit(logged)
        dm, w, residual = _dr_terms(logged, target_prob_treat, outcome_model)
        tau = self._select_tau(dm, w, residual) if self.tau == "auto" else float(self.tau)
        self.selected_tau = tau
        keep = (w <= tau).astype(float)
        phi = dm + keep * w * residual
        point, lo, hi = bootstrap_ci(np.mean, phi, n_bootstrap=n_bootstrap, seed=seed)
        return EstimateResult(self.name, point, lo, hi)

    def _select_tau(self, dm, w, residual) -> float:
        """Pick tau minimizing an estimated-MSE proxy on the logged sample.

        The switching bias is bounded by the magnitude of the dropped IPS
        correction on high-weight units; traded against the variance of the
        retained estimator (Wang et al. 2017, MSE-bound selection)."""
        n = len(w)
        best_tau, best_mse = self.tau_grid[0], float("inf")
        for tau in self.tau_grid:
            keep = (w <= tau).astype(float)
            phi = dm + keep * w * residual
            var = float(np.var(phi, ddof=1)) / n if n > 1 else float(np.var(phi))
            bias_bound = float(np.mean((1.0 - keep) * np.abs(w * residual)))
            mse = bias_bound**2 + var
            if mse < best_mse:
                best_tau, best_mse = tau, mse
        return best_tau

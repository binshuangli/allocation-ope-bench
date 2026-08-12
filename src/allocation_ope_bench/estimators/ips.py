"""Importance-sampling value estimators: IPS, SNIPS, and balanced IPS (BIPS)."""

from __future__ import annotations

import numpy as np

from allocation_ope_bench.estimators.base import (
    EstimateResult,
    OPEEstimator,
    policy_action_prob,
)
from allocation_ope_bench.metrics.stats import bootstrap_ci


def _importance_weight(logged, target_prob_treat):
    """w_i = pi_e(a_i | x_i) / pi_b(a_i | x_i)."""
    pi_e = policy_action_prob(target_prob_treat, logged.action)
    return pi_e / logged.pscore


class IPS(OPEEstimator):
    """V = mean_i [ w_i Y_i ]. Unbiased under common support; high variance and,
    for a deterministic target under tight budgets, near-zero effective sample
    size as most weights collapse to 0 (the support-deficiency pathology)."""

    name = "ips"

    def estimate(self, logged, target_prob_treat, outcome_model=None, n_bootstrap=500, seed=42):
        w = _importance_weight(logged, target_prob_treat)
        phi = w * logged.reward
        point, lo, hi = bootstrap_ci(np.mean, phi, n_bootstrap=n_bootstrap, seed=seed)
        return EstimateResult(self.name, point, lo, hi)


class ClippedIPS(OPEEstimator):
    """IPS with the importance weights truncated at a fixed ceiling M:
    V = mean_i [ min(w_i, M) Y_i ].

    Weight clipping is the standard practitioner response to exactly the pathology
    this benchmark documents -- a few units with enormous weights dominating the
    estimate under weak overlap. It trades a downward bias (clipped units are
    under-counted) against a large variance reduction. We include it so the RQ1
    claim is "weight control helps this much and no further" rather than merely
    "raw IPS is fragile".
    """

    name = "clipped_ips"

    def __init__(self, clip_max: float = 10.0):
        self.clip_max = float(clip_max)

    def estimate(self, logged, target_prob_treat, outcome_model=None, n_bootstrap=500, seed=42):
        w = np.minimum(_importance_weight(logged, target_prob_treat), self.clip_max)
        phi = w * logged.reward
        point, lo, hi = bootstrap_ci(np.mean, phi, n_bootstrap=n_bootstrap, seed=seed)
        return EstimateResult(self.name, point, lo, hi)


class SNIPS(OPEEstimator):
    """Self-normalized IPS: V = sum_i w_i Y_i / sum_i w_i. Lower variance and
    scale-stable, at the cost of a small bias."""

    name = "snips"

    def estimate(self, logged, target_prob_treat, outcome_model=None, n_bootstrap=500, seed=42):
        w = _importance_weight(logged, target_prob_treat)
        wy = w * logged.reward

        def ratio(a, b):
            denom = b.sum()
            return float(a.sum() / denom) if denom != 0 else float("nan")

        point, lo, hi = bootstrap_ci(ratio, wy, w, n_bootstrap=n_bootstrap, seed=seed)
        return EstimateResult(self.name, point, lo, hi)


class BIPS(OPEEstimator):
    """Balanced / mixed-logging IPS (coupon-paper fix).

    Divides by a mixture logging propensity ``(1-alpha)*pi_b + alpha*0.5`` so the
    weights stay bounded and the deterministic target keeps support. Faithful
    when logging actually included an alpha-exploration mixture (see
    ``policies.logging.make_logged_data(mixture_alpha=...)``); otherwise it trades
    a small bias for robustness.

    ``alpha`` is **not** hard-coded: by default ("auto") it is selected on the
    logged sample by minimizing an estimated-MSE proxy (deviation-from-IPS bias
    proxy ^2 + variance) over ``alpha_grid`` — the same fair, data-driven tuning
    scheme used for Switch-DR's tau. A fixed float or a grid (for a sensitivity
    sweep, see ``estimators.tuning.hyperparam_sweep``) may also be supplied.
    """

    name = "bips"
    DEFAULT_ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.5)

    def __init__(self, alpha="auto", alpha_grid=DEFAULT_ALPHA_GRID):
        self.alpha = alpha
        self.alpha_grid = tuple(alpha_grid)
        self.selected_alpha: float | None = None

    def estimate(self, logged, target_prob_treat, outcome_model=None, n_bootstrap=500, seed=42):
        pi_e = policy_action_prob(target_prob_treat, logged.action)
        alpha = self._select_alpha(pi_e, logged) if self.alpha == "auto" else float(self.alpha)
        self.selected_alpha = alpha
        denom = (1.0 - alpha) * logged.pscore + alpha * 0.5
        phi = (pi_e / denom) * logged.reward
        point, lo, hi = bootstrap_ci(np.mean, phi, n_bootstrap=n_bootstrap, seed=seed)
        return EstimateResult(self.name, point, lo, hi)

    def _select_alpha(self, pi_e, logged) -> float:
        """Pick alpha minimizing an estimated-MSE proxy on the logged sample.

        IPS (alpha=0) is ~unbiased but high-variance; deviation of the BIPS mean
        from the IPS mean proxies the bias introduced by mixing, traded against
        the (shrinking) variance.
        """
        n = logged.n
        w0 = pi_e / logged.pscore
        ips_val = float(np.mean(w0 * logged.reward))
        best_alpha, best_mse = self.alpha_grid[0], float("inf")
        for a in self.alpha_grid:
            denom = (1.0 - a) * logged.pscore + a * 0.5
            phi = (pi_e / denom) * logged.reward
            var = float(np.var(phi, ddof=1)) / n if n > 1 else float(np.var(phi))
            bias_proxy = float(np.mean(phi)) - ips_val
            mse = bias_proxy**2 + var
            if mse < best_mse:
                best_alpha, best_mse = a, mse
        return best_alpha

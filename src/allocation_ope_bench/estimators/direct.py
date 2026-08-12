"""Direct Method (a.k.a. naive model-based) value estimator."""

from __future__ import annotations

import numpy as np

from allocation_ope_bench.estimators.base import (
    EstimateResult,
    OPEEstimator,
    OutcomeModel,
)
from allocation_ope_bench.metrics.stats import bootstrap_ci


class DirectMethod(OPEEstimator):
    """V = mean_i [ pi_e(1|x) mu1_hat(x) + pi_e(0|x) mu0_hat(x) ].

    Model-based; unbiased only if the outcome model is correct. Tends to look
    *deceptively accurate* under a tight budget with poor overlap (no importance
    weights to expose the support gap) — the headline failure mode of the paper.
    """

    name = "dm"

    def estimate(self, logged, target_prob_treat, outcome_model=None, n_bootstrap=500, seed=42):
        if outcome_model is None:
            outcome_model = OutcomeModel(seed=seed).fit(logged)
        mu1 = outcome_model.predict(logged.context, 1)
        mu0 = outcome_model.predict(logged.context, 0)
        phi = target_prob_treat * mu1 + (1.0 - target_prob_treat) * mu0
        point, lo, hi = bootstrap_ci(np.mean, phi, n_bootstrap=n_bootstrap, seed=seed)
        return EstimateResult(self.name, point, lo, hi)

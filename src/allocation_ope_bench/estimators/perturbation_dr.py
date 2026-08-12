"""Perturbation-smoothed DR (Guo-inspired) — secondary optimization-aware estimator.

A heuristic for the optimization-bias regime, *inspired by* (not a transcription
of) Guo, Jordan & Zhou (NeurIPS 2022), "OPE with Policy-Dependent Optimization
Response." The hard top-k argmax is what creates the optimizer's curse, so we
smooth it: re-solve the budgeted allocation many times with i.i.d. noise added to
the scores and average the DR value of each perturbed allocation. A
finite-difference shadow price ``lambda`` (sensitivity of value to the budget) is
reported as the dual of the budget constraint.

This is the SECONDARY comparison. The recommended optimization-aware estimator is
``cross_fitted_dr`` (proper sample splitting). Both are reported next to plain DR
in the optimization-bias regime so the benchmark shows which actually removes the
bias. The whole optimization-aware estimator set is flagged for confirmation at
the WP3 protocol gate.
"""

from __future__ import annotations

import numpy as np

from allocation_ope_bench.data.ground_truth import allocate_under_budget
from allocation_ope_bench.estimators.base import EstimateResult, OPEEstimator, OutcomeModel
from allocation_ope_bench.estimators.doubly_robust import dr_contributions
from allocation_ope_bench.metrics.stats import bootstrap_ci


class PerturbationSmoothedDR(OPEEstimator):
    name = "perturbation_dr"

    def __init__(
        self,
        n_perturbations: int = 25,
        perturbation_scale: float = 0.5,
        delta: float = 0.05,
        seed: int = 42,
    ):
        self.n_perturbations = n_perturbations
        self.perturbation_scale = perturbation_scale
        self.delta = delta
        self.seed = seed
        self.last_shadow_price: float | None = None

    def estimate(
        self,
        logged,
        target_prob_treat,
        outcome_model=None,
        n_bootstrap=500,
        seed=42,
        *,
        scores=None,
        costs=None,
        budget_k=None,
    ):
        if scores is None or budget_k is None:
            raise ValueError(
                "PerturbationSmoothedDR requires per-unit `scores` and `budget_k` "
                "(it must re-solve the budgeted allocation)."
            )
        if outcome_model is None:
            outcome_model = OutcomeModel(seed=seed).fit(logged)
        scores = np.asarray(scores, dtype=float)
        costs = np.ones(logged.n) if costs is None else np.asarray(costs, dtype=float)
        rng = np.random.default_rng(seed)
        s_sd = scores.std() + 1e-12

        # 1) Perturbation-smoothed DR value at the budgeted optimization response.
        perturbed_vals = []
        for _ in range(self.n_perturbations):
            noise = rng.normal(scale=self.perturbation_scale * s_sd, size=logged.n)
            z = allocate_under_budget(scores + noise, costs, budget_k).astype(float)
            perturbed_vals.append(float(dr_contributions(logged, z, outcome_model).mean()))
        value = float(np.mean(perturbed_vals))

        # 2) Finite-difference dual (shadow price) at the unperturbed response.
        kp = min(1.0, budget_k + self.delta)
        km = max(0.0, budget_k - self.delta)
        v_plus = float(
            dr_contributions(
                logged, allocate_under_budget(scores, costs, kp).astype(float), outcome_model
            ).mean()
        )
        v_minus = float(
            dr_contributions(
                logged, allocate_under_budget(scores, costs, km).astype(float), outcome_model
            ).mean()
        )
        self.last_shadow_price = (v_plus - v_minus) / (kp - km) if kp > km else float("nan")

        # CI: bootstrap the base-allocation DR contributions, recentered on `value`.
        z_base = allocate_under_budget(scores, costs, budget_k).astype(float)
        phi_base = dr_contributions(logged, z_base, outcome_model)
        point_base, lo, hi = bootstrap_ci(np.mean, phi_base, n_bootstrap=n_bootstrap, seed=seed)
        shift = value - point_base
        return EstimateResult(self.name, value, lo + shift, hi + shift)

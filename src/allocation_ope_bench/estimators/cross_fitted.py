"""Cross-fitted DR — the principled optimization-aware value estimator.

When the allocation policy is *learned on the same data it is evaluated on*, the
optimizer selects units whose scores are inflated by their own estimation noise,
so plain DR over-states the deployed value (the optimizer's curse / optimization
bias). Cross-fitting removes this by sample splitting.

Two modes, with DIFFERENT estimands — do not mix them up:

* **Frozen-policy mode** (``target_prob_treat`` given): the policy is fixed
  outside the estimator; only the outcome-model *nuisance* is cross-fitted (fit
  on out-of-fold data, evaluated on the held-out fold). Estimand: the value of
  THE GIVEN policy — the same estimand as plain DR, so their biases are directly
  comparable. This isolates evaluation-side overfitting (the same in-sample
  model both selecting units and predicting their outcomes).
* **Fold-policy mode** (``target_prob_treat=None``): for each fold the implied
  uplift score AND the allocation are re-derived from the out-of-fold model.
  Estimand: the expected value of the *learning algorithm* (an average over
  fold-specific policies), NOT the value of any single fixed policy. Scoring
  this mode against a full-sample policy's truth is an estimand mismatch; the
  experiment must supply a fold-matched reference value.
"""

from __future__ import annotations

import numpy as np

from allocation_ope_bench.data.ground_truth import allocate_under_budget
from allocation_ope_bench.estimators.base import EstimateResult, OPEEstimator, OutcomeModel
from allocation_ope_bench.estimators.doubly_robust import dr_contributions
from allocation_ope_bench.metrics.stats import bootstrap_ci


class CrossFittedDR(OPEEstimator):
    name = "cross_fitted_dr"

    def __init__(self, n_folds: int = 5, base_learner: str = "lightgbm", seed: int = 42):
        if n_folds < 2:
            raise ValueError("cross-fitting needs at least 2 folds")
        self.n_folds = n_folds
        self.base_learner = base_learner
        self.seed = seed

    def estimate(
        self,
        logged,
        target_prob_treat=None,
        outcome_model=None,
        n_bootstrap=500,
        seed=42,
        *,
        budget_k=None,
        costs=None,
        scores=None,  # accepted for interface symmetry; ignored (derived per fold)
    ):
        n = logged.n
        costs = np.ones(n) if costs is None else np.asarray(costs, dtype=float)
        rng = np.random.default_rng(seed)
        folds = np.array_split(rng.permutation(n), self.n_folds)
        all_idx = np.arange(n)

        frozen = target_prob_treat is not None
        if not frozen and budget_k is None:
            raise ValueError("Fold-policy mode requires `budget_k` (it re-solves per fold).")
        if frozen:
            target = np.asarray(target_prob_treat, dtype=float)

        phi = np.empty(n)
        for fold in folds:
            train_idx = np.setdiff1d(all_idx, fold, assume_unique=False)
            om = OutcomeModel(self.base_learner, seed=seed).fit(logged.subset(train_idx))
            if frozen:
                # Frozen-policy mode: same estimand as plain DR — only the
                # nuisance is out-of-fold.
                a_f = target[fold]
            else:
                Xf = logged.context[fold]
                # Fold-policy mode: score AND allocation from the out-of-fold
                # model => estimates the learning-algorithm value.
                s_f = om.predict(Xf, 1) - om.predict(Xf, 0)
                a_f = allocate_under_budget(s_f, costs[fold], budget_k).astype(float)
            phi[fold] = dr_contributions(logged.subset(fold), a_f, om)

        value = float(phi.mean())
        _, lo, hi = bootstrap_ci(np.mean, phi, n_bootstrap=n_bootstrap, seed=seed)
        return EstimateResult(self.name, value, lo, hi)

"""WP3 — optimization-aware estimators: cross-fitted DR (primary) and
perturbation-smoothed DR (Guo-inspired, secondary), vs plain DR in the
optimization-bias regime."""

import numpy as np
import pytest

from allocation_ope_bench.data import make_synthetic, train_eval_split, true_allocation_value
from allocation_ope_bench.data.ground_truth import allocate_under_budget
from allocation_ope_bench.estimators import OutcomeModel, get_ope_estimator, list_estimators
from allocation_ope_bench.policies import make_logged_data


def test_optimization_aware_estimators_registered():
    assert "cross_fitted_dr" in list_estimators()
    assert "perturbation_dr" in list_estimators()


def test_cross_fitted_requires_budget_k():
    ds = make_synthetic(n=600, seed=0)
    logged = make_logged_data(ds, ds.mu1 - ds.mu0, temperature=0.0, seed=0)
    with pytest.raises(ValueError, match="budget_k"):
        get_ope_estimator("cross_fitted_dr").estimate(logged)


def test_perturbation_requires_scores_and_budget():
    ds = make_synthetic(n=600, seed=0)
    logged = make_logged_data(ds, ds.mu1 - ds.mu0, temperature=0.0, seed=0)
    with pytest.raises(ValueError, match="requires"):
        get_ope_estimator("perturbation_dr").estimate(logged, np.zeros(logged.n))


def _optimization_bias_cell(seed, k=0.1):
    """Return (truth, plain_dr, cross_fitted_dr, perturbation_dr) values."""
    ds = make_synthetic(n=1600, seed=seed, effect_scale=1.0, noise_scale=2.0)
    _, ev = train_eval_split(ds, seed=seed)
    eval_ds = ds.subset(ev)
    logged = make_logged_data(eval_ds, eval_ds.mu1 - eval_ds.mu0, temperature=0.0, seed=seed)
    # In-sample outcome model => estimated uplift overfits => optimizer's curse.
    model = OutcomeModel(seed=seed).fit(logged)
    scores = model.predict(logged.context, 1) - model.predict(logged.context, 0)
    target = allocate_under_budget(scores, np.ones(logged.n), k).astype(float)

    truth = true_allocation_value(eval_ds, scores, k)
    dr = get_ope_estimator("dr").estimate(logged, target, outcome_model=model, seed=seed).value
    cf = get_ope_estimator("cross_fitted_dr").estimate(logged, budget_k=k, seed=seed).value
    pdr = (
        get_ope_estimator("perturbation_dr")
        .estimate(logged, target, outcome_model=model, scores=scores, budget_k=k, seed=seed)
        .value
    )
    return truth, dr, cf, pdr


def test_cross_fitted_dr_removes_optimization_bias_best():
    """Plain DR is optimistic; cross-fitted DR de-biases most; perturbation DR
    helps modestly. Averaged over seeds for a non-flaky ordering."""
    dr_b, cf_b, pd_b = [], [], []
    for seed in range(4):
        truth, dr, cf, pdr = _optimization_bias_cell(seed)
        dr_b.append(abs(dr - truth))
        cf_b.append(abs(cf - truth))
        pd_b.append(abs(pdr - truth))
    mean_dr, mean_cf, mean_pd = np.mean(dr_b), np.mean(cf_b), np.mean(pd_b)

    assert mean_cf < mean_dr  # cross-fitting removes the optimization bias
    assert mean_cf < mean_pd  # and does so better than perturbation smoothing
    assert mean_pd <= mean_dr + 1e-9  # perturbation smoothing at least doesn't worsen


def test_all_optimization_aware_return_finite_with_ci():
    ds = make_synthetic(n=1200, seed=5)
    logged = make_logged_data(ds, ds.mu1 - ds.mu0, temperature=0.0, seed=5)
    scores = ds.mu1 - ds.mu0
    target = allocate_under_budget(scores, np.ones(logged.n), 0.2).astype(float)
    for name, extra in (
        ("cross_fitted_dr", {"budget_k": 0.2}),
        ("perturbation_dr", {"scores": scores, "budget_k": 0.2}),
    ):
        res = get_ope_estimator(name).estimate(logged, target, n_bootstrap=100, seed=5, **extra)
        assert np.isfinite(res.value)
        assert res.ci_low <= res.value <= res.ci_high

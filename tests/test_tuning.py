"""WP3 — data-driven hyperparameter tuning (Switch-DR tau, BIPS alpha) + sweeps."""

import numpy as np

from allocation_ope_bench.data import make_synthetic, train_eval_split
from allocation_ope_bench.estimators import get_ope_estimator, hyperparam_sweep
from allocation_ope_bench.estimators.doubly_robust import SwitchDR
from allocation_ope_bench.estimators.ips import BIPS
from allocation_ope_bench.policies import AllocationPolicy, make_logged_data


def _logged_setup(seed=0, budget_k=0.2, temperature=2.0):
    ds = make_synthetic(n=4000, seed=seed)
    tr, ev = train_eval_split(ds, seed=seed)
    eval_ds = ds.subset(ev)
    policy = AllocationPolicy(variant="deterministic", seed=seed).fit(ds.subset(tr))
    score = policy.score(eval_ds.X, feature_names=eval_ds.feature_names)
    logged = make_logged_data(eval_ds, score, temperature=temperature, seed=seed)
    target = policy.action_prob(logged.context, budget_k, feature_names=eval_ds.feature_names)
    return logged, target, budget_k


def test_switch_dr_auto_selects_tau_in_grid():
    logged, target, _ = _logged_setup()
    est = SwitchDR(tau="auto")
    res = est.estimate(logged, target, seed=0)
    assert est.selected_tau in SwitchDR.DEFAULT_TAU_GRID
    assert np.isfinite(res.value)


def test_bips_auto_selects_alpha_in_grid():
    logged, target, _ = _logged_setup()
    est = BIPS(alpha="auto")
    res = est.estimate(logged, target, seed=0)
    assert est.selected_alpha in BIPS.DEFAULT_ALPHA_GRID
    assert np.isfinite(res.value)


def test_registry_defaults_are_auto():
    # No hard-coded magic numbers leak through the registry defaults.
    assert get_ope_estimator("switch_dr").tau == "auto"
    assert get_ope_estimator("bips").alpha == "auto"


def test_fixed_value_still_supported():
    logged, target, _ = _logged_setup()
    res = SwitchDR(tau=10.0).estimate(logged, target, seed=0)
    assert np.isfinite(res.value)


def test_hyperparam_sweep_over_grid():
    logged, target, _ = _logged_setup()
    sweep = hyperparam_sweep(BIPS, "alpha", [0.0, 0.1, 0.5], logged, target, seed=0)
    assert set(sweep.keys()) == {0.0, 0.1, 0.5}
    assert all(np.isfinite(r.value) for r in sweep.values())


def test_cross_fitted_default_folds_is_five():
    assert get_ope_estimator("cross_fitted_dr").n_folds == 5

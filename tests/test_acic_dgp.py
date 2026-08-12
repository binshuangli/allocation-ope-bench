"""ACIC-2017-style known-effect DGPs on real IHDP covariates."""

import numpy as np
import pytest

from allocation_ope_bench.data.acic_dgp import N_SETTINGS, make_acic_ihdp

# The DGPs read the cached IHDP covariate file; first use needs one download.
pytestmark = pytest.mark.network


def test_invalid_setting_raises():
    with pytest.raises(ValueError, match="setting"):
        make_acic_ihdp(setting=0)
    with pytest.raises(ValueError, match="setting"):
        make_acic_ihdp(setting=N_SETTINGS + 1)


def test_dataset_contract():
    ds = make_acic_ihdp(setting=1, seed=0)
    assert ds.name == "acic_s1"
    assert ds.has_ground_truth_effect
    assert ds.propensity is None  # IHDP-style surface-logging path
    assert ds.mu0 is not None and ds.mu1 is not None
    assert ds.X.shape[0] == ds.n == len(ds.mu0) == len(ds.mu1)
    assert np.isfinite(ds.X).all()
    assert np.isfinite(ds.outcome).all()
    assert set(np.unique(ds.treatment)) <= {0, 1}


def test_truth_fixed_across_seeds_observables_vary():
    a = make_acic_ihdp(setting=3, seed=0)
    b = make_acic_ihdp(setting=3, seed=99)
    np.testing.assert_array_equal(a.mu0, b.mu0)
    np.testing.assert_array_equal(a.mu1, b.mu1)
    assert not np.array_equal(a.outcome, b.outcome)


def test_settings_have_distinct_truths():
    taus = [make_acic_ihdp(setting=s, seed=0) for s in range(1, N_SETTINGS + 1)]
    effects = [ds.mu1 - ds.mu0 for ds in taus]
    for i in range(len(effects)):
        for j in range(i + 1, len(effects)):
            assert not np.allclose(effects[i], effects[j]), f"settings {i+1},{j+1} share a truth"


def test_high_noise_setting_is_noisier():
    # Settings (1,2) share the linear truth; only noise differs. Residual std
    # of the factual outcome around its own surface must be ~4x larger.
    lo = make_acic_ihdp(setting=1, seed=0)
    hi = make_acic_ihdp(setting=2, seed=0)
    mu_lo = np.where(lo.treatment == 1, lo.mu1, lo.mu0)
    mu_hi = np.where(hi.treatment == 1, hi.mu1, hi.mu0)
    r_lo = (lo.outcome - mu_lo).std()
    r_hi = (hi.outcome - mu_hi).std()
    assert r_hi > 2.5 * r_lo


def test_hillstrom_covariate_source():
    a = make_acic_ihdp(setting=2, seed=0, covariates="hillstrom", n_rows=2000)
    b = make_acic_ihdp(setting=2, seed=9, covariates="hillstrom", n_rows=2000)
    assert a.name == "acic_hi_s2"
    assert a.n == 2000
    np.testing.assert_array_equal(a.X, b.X)  # fixed subsample across seeds
    np.testing.assert_array_equal(a.mu0, b.mu0)  # truth fixed per setting
    assert not np.array_equal(a.outcome, b.outcome)


def test_runner_factory_builds_acic():
    from allocation_ope_bench.experiments.runner import _build_dataset

    ds = _build_dataset({"name": "acic", "setting": 4}, seed=7)
    assert ds.name == "acic_s4"
    assert ds.has_ground_truth_effect


def test_surface_logging_path_works():
    from allocation_ope_bench.policies import make_logged_data

    ds = make_acic_ihdp(setting=5, seed=1)
    rng = np.random.default_rng(0)
    scores = rng.normal(size=ds.n)
    logged = make_logged_data(ds, scores, temperature=2.0, seed=0)
    assert logged.n == ds.n  # surface path samples every unit
    assert np.isfinite(logged.reward).all()

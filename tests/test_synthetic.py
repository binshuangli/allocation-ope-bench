"""WP1 — synthetic generator."""

import numpy as np

from allocation_ope_bench.data import make_synthetic


def test_synthetic_shapes_and_schema():
    ds = make_synthetic(n=1000, n_features=8, seed=0)
    assert ds.n == 1000
    assert ds.n_features == 8
    assert ds.has_ground_truth_effect
    assert ds.mu0.shape == (1000,)
    assert ds.mu1.shape == (1000,)
    assert ds.propensity.shape == (1000,)
    assert set(np.unique(ds.treatment)).issubset({0, 1})


def test_synthetic_rct_propensity_constant():
    ds = make_synthetic(n=2000, seed=1, logging="rct", rct_propensity=0.5)
    assert np.allclose(ds.propensity, 0.5)


def test_synthetic_logistic_overlap_varies():
    sharp = make_synthetic(n=5000, seed=2, logging="logistic", logging_temperature=4.0)
    mild = make_synthetic(n=5000, seed=2, logging="logistic", logging_temperature=0.5)
    # Sharper logging => propensities spread further toward the clip bounds.
    assert sharp.propensity.std() > mild.propensity.std()


def test_synthetic_effect_present():
    ds = make_synthetic(n=2000, seed=3, effect_scale=2.0)
    tau = ds.mu1 - ds.mu0
    assert tau.std() > 0  # heterogeneous
    assert np.any(tau > 0) and np.any(tau < 0)  # sign-varying


def test_synthetic_heterogeneous_cost():
    ds = make_synthetic(n=500, seed=4, cost="heterogeneous")
    assert ds.cost.std() > 0
    assert np.all(ds.cost > 0)

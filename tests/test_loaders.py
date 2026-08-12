"""WP1 — real / semi-synthetic loaders. Network-marked (downloads required)."""

import numpy as np
import pytest

from allocation_ope_bench.data import load_dataset, true_allocation_value


def _basic_schema_checks(ds):
    assert ds.n > 0
    assert ds.X.ndim == 2
    assert set(np.unique(ds.treatment)).issubset({0, 1})
    assert np.all(ds.cost > 0)
    if ds.has_ground_truth_effect:
        assert ds.mu0 is not None and ds.mu1 is not None
    else:
        assert ds.propensity is not None
    # A value is computable for an arbitrary scoring at a mid budget.
    rng = np.random.default_rng(0)
    v = true_allocation_value(ds, rng.normal(size=ds.n), budget_k=0.3)
    assert np.isfinite(v)


@pytest.mark.network
@pytest.mark.parametrize("name", ["hillstrom", "lenta", "x5", "criteo"])
def test_marketing_rct_loaders(name):
    ds = load_dataset(name)
    assert ds.has_ground_truth_effect is False
    assert ds.propensity is not None
    _basic_schema_checks(ds)


@pytest.mark.network
def test_ihdp_has_ground_truth_effect():
    ds = load_dataset("ihdp")
    assert ds.has_ground_truth_effect is True
    _basic_schema_checks(ds)


@pytest.mark.network
def test_jobs_is_randomized_policy_risk():
    ds = load_dataset("jobs")
    # Per WP1 decision: no individual counterfactuals -> oracle via propensity.
    assert ds.has_ground_truth_effect is False
    assert ds.propensity is not None
    _basic_schema_checks(ds)

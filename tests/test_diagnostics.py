"""WP3 — trust diagnostics (RQ2), known-answer tests."""

import numpy as np
import pytest

from allocation_ope_bench.data import make_synthetic
from allocation_ope_bench.metrics import (
    budget_tightness,
    compute_diagnostics,
    effective_sample_size,
    importance_weights,
    max_importance_weight,
    support_deficiency,
    support_deficiency_sensitivity,
)
from allocation_ope_bench.policies import make_logged_data


def test_importance_weights_known():
    w = importance_weights([1.0, 0.0], action=[1, 0], pscore=[0.5, 0.5])
    assert w.tolist() == [2.0, 2.0]


def test_ess_equal_weights_is_n():
    assert effective_sample_size([1.0, 1.0, 1.0, 1.0]) == pytest.approx(4.0)


def test_ess_collapses_with_one_dominant_weight():
    ess = effective_sample_size([100.0, 1e-6, 1e-6, 1e-6])
    assert ess < 1.1  # effectively one unit


def test_max_weight():
    assert max_importance_weight([0.5, 3.0, 1.0]) == 3.0


def test_support_deficiency_known():
    # unit0 treated by policy but logging treat-prob 0.01 < eps => unsupported.
    defic = support_deficiency(
        target_prob_treat=[1.0, 1.0, 0.0],
        logging_prob_treat=[0.01, 0.5, 0.5],
        eps=0.05,
    )
    assert defic == pytest.approx(1.0 / 3.0)


def test_support_deficiency_none_when_well_supported():
    defic = support_deficiency([1.0, 0.0], [0.5, 0.5], eps=0.05)
    assert defic == 0.0


def test_budget_tightness():
    assert budget_tightness(0.3) == pytest.approx(0.7)


def test_support_deficiency_sensitivity_monotone():
    # Looser eps flags at least as much mass as unsupported.
    target = [1.0, 1.0, 0.0, 1.0]
    logging = [0.005, 0.03, 0.5, 0.08]
    sens = support_deficiency_sensitivity(target, logging, eps_grid=(0.01, 0.05, 0.10))
    assert sens[0.01] <= sens[0.05] <= sens[0.10]
    assert sens[0.01] == pytest.approx(1.0 / 4.0)  # only the 0.005 unit
    assert sens[0.10] == pytest.approx(3.0 / 4.0)  # 0.005, 0.03, 0.08 units


def test_compute_diagnostics_bundle():
    ds = make_synthetic(n=2000, seed=0)
    logged = make_logged_data(ds, ds.mu1 - ds.mu0, temperature=0.0, seed=0)
    target = (ds.mu1 - ds.mu0 > 0).astype(float)[: logged.n]
    diag = compute_diagnostics(logged, target, budget_k=0.3)
    for key in (
        "ess",
        "ess_fraction",
        "max_weight",
        "support_deficiency",
        "support_deficiency_eps001",
        "support_deficiency_eps010",
        "budget_tightness",
    ):
        assert key in diag and np.isfinite(diag[key])
    assert 0.0 <= diag["ess_fraction"] <= 1.0
    # Sensitivity ordering holds inside the bundle too (eps 0.01 <= 0.05 <= 0.10).
    assert (
        diag["support_deficiency_eps001"]
        <= diag["support_deficiency"]
        <= diag["support_deficiency_eps010"]
    )

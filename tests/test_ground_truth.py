"""WP1 — budget allocation + ground-truth allocation value (the gold standard)."""

import numpy as np
import pytest

from allocation_ope_bench.data import Dataset, make_synthetic, true_allocation_value
from allocation_ope_bench.data.ground_truth import allocate_under_budget

# ── allocate_under_budget ─────────────────────────────────────────────────────


def test_allocate_unit_cost_top_k():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    costs = np.ones(4)
    a = allocate_under_budget(scores, costs, budget_k=0.5)  # top 2
    assert a.tolist() == [0, 1, 0, 1]


def test_allocate_budget_zero_and_one():
    scores = np.array([3.0, 1.0, 2.0])
    costs = np.ones(3)
    assert allocate_under_budget(scores, costs, 0.0).tolist() == [0, 0, 0]
    assert allocate_under_budget(scores, costs, 1.0).tolist() == [1, 1, 1]


def test_allocate_heterogeneous_cost_respects_budget():
    scores = np.array([10.0, 9.0, 8.0, 1.0])
    costs = np.array([1.0, 5.0, 1.0, 1.0])  # total 8, budget 0.25 -> 2.0
    a = allocate_under_budget(scores, costs, budget_k=0.25)
    # Greedy by score: take unit0 (cost1, cum1<=2), unit1 needs 5 -> 6>2 skip,
    # unit2 (cost1, cum 1+5+1? no — cumcost is over sorted order incl. skipped).
    # cumcost over [u0,u1,u2,u3] = [1,6,7,8]; <=2 -> only u0.
    assert a.tolist() == [1, 0, 0, 0]


def test_allocate_ties_stable():
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    a = allocate_under_budget(scores, np.ones(4), budget_k=0.5)
    assert a.sum() == 2  # exactly half, ties broken by original order


# ── true_allocation_value: exact (ground-truth-effect) branch ─────────────────


def test_true_value_exact_closed_form():
    # 4 units, unit cost, treat top-2 by score.
    ds = Dataset(
        name="toy",
        X=np.zeros((4, 1)),
        treatment=np.array([0, 0, 1, 1]),
        outcome=np.zeros(4),
        cost=np.ones(4),
        propensity=None,
        has_ground_truth_effect=True,
        mu0=np.array([0.0, 0.0, 0.0, 0.0]),
        mu1=np.array([10.0, 5.0, 1.0, 0.0]),
    )
    scores = ds.mu1 - ds.mu0  # = [10, 5, 1, 0]
    # budget 0.5 -> treat units 0,1: value = mean([10, 5, 0, 0]) = 3.75
    assert true_allocation_value(ds, scores, budget_k=0.5) == pytest.approx(3.75)


def test_true_value_oracle_optimal_at_positive_set():
    ds = make_synthetic(n=8000, seed=7, effect_scale=1.5, noise_scale=1.0)
    tau = ds.mu1 - ds.mu0
    pos_frac = float(np.mean(tau > 0))

    v_none = true_allocation_value(ds, tau, budget_k=0.0)
    v_all = true_allocation_value(ds, tau, budget_k=1.0)
    v_pos = true_allocation_value(ds, tau, budget_k=pos_frac)

    assert v_none == pytest.approx(float(ds.mu0.mean()))
    assert v_all == pytest.approx(float(ds.mu1.mean()))
    # Treating exactly the positive-effect units maximizes gross value.
    assert v_pos >= v_none - 1e-9
    assert v_pos >= v_all - 1e-9


# ── true_allocation_value: known-propensity IPS oracle ────────────────────────


def test_ips_oracle_recovers_exact_value_on_rct():
    """The IPS branch (RCT) must agree with the exact (mu) branch on a large RCT.

    This is the WP1 acceptance check: true value recovered on a known case.
    """
    ds = make_synthetic(
        n=60000, seed=11, logging="rct", rct_propensity=0.5, effect_scale=1.0, noise_scale=1.0
    )
    tau = ds.mu1 - ds.mu0

    # Force the IPS branch by dropping the known effects but keeping propensity.
    ds_rct = Dataset(
        name="synthetic_rct",
        X=ds.X,
        treatment=ds.treatment,
        outcome=ds.outcome,
        cost=ds.cost,
        propensity=ds.propensity,
        has_ground_truth_effect=False,
    )

    for k in (0.1, 0.3, 0.5, 0.8):
        exact = true_allocation_value(ds, tau, budget_k=k)
        ips = true_allocation_value(ds_rct, tau, budget_k=k)
        assert ips == pytest.approx(exact, abs=0.08), f"budget {k}: {ips} vs {exact}"


def test_ips_oracle_handles_skewed_propensity():
    # Criteo-like: ~85% treated. Control IPS weights ~1/0.15, still bounded.
    ds = make_synthetic(n=60000, seed=13, logging="rct", rct_propensity=0.85)
    tau = ds.mu1 - ds.mu0
    ds_rct = Dataset(
        name="syn_skew",
        X=ds.X,
        treatment=ds.treatment,
        outcome=ds.outcome,
        cost=ds.cost,
        propensity=ds.propensity,
        has_ground_truth_effect=False,
    )
    exact = true_allocation_value(ds, tau, budget_k=0.3)
    ips = true_allocation_value(ds_rct, tau, budget_k=0.3)
    assert ips == pytest.approx(exact, abs=0.1)

"""WP3 — estimator error metrics (RQ1), known-answer tests."""

import numpy as np
import pandas as pd
import pytest

from allocation_ope_bench.metrics import cell_error, error_vs_budget


def test_cell_error_zero_when_exact():
    out = cell_error([3.0, 3.0, 3.0], 3.0)
    assert out["bias"] == 0.0
    assert out["rmse"] == 0.0
    assert out["rel_rmse"] == 0.0


def test_cell_error_constant_offset():
    out = cell_error([5.0, 5.0, 5.0], 3.0)  # +2 everywhere
    assert out["bias"] == pytest.approx(2.0)
    assert out["rmse"] == pytest.approx(2.0)
    assert out["rel_rmse"] == pytest.approx(2.0 / 3.0)
    assert out["rel_bias"] == pytest.approx(2.0 / 3.0)


def test_cell_error_rmse_vs_bias_with_variance():
    out = cell_error([2.0, 4.0], 3.0)  # errors -1, +1 => bias 0, rmse 1
    assert out["bias"] == pytest.approx(0.0)
    assert out["rmse"] == pytest.approx(1.0)


def test_cell_error_small_denominator_is_nan():
    out = cell_error([1.0, -1.0], 0.0)
    assert np.isnan(out["rel_rmse"])


def test_error_vs_budget_groups():
    df = pd.DataFrame(
        {
            "dataset": ["s", "s", "s", "s"],
            "estimator": ["ips", "ips", "ips", "ips"],
            "budget_k": [0.1, 0.1, 0.5, 0.5],
            "estimate": [1.0, 3.0, 5.0, 5.0],
            "true_value": [2.0, 2.0, 5.0, 5.0],
        }
    )
    out = error_vs_budget(df).sort_values("budget_k").reset_index(drop=True)
    assert len(out) == 2
    # budget 0.1: estimates [1,3] vs 2 => bias 0, rmse 1
    assert out.loc[0, "bias"] == pytest.approx(0.0)
    assert out.loc[0, "rmse"] == pytest.approx(1.0)
    # budget 0.5: exact
    assert out.loc[1, "rmse"] == pytest.approx(0.0)

"""WP3 — selection regret + SharpeRatio@k (RQ4), known-answer tests."""

import numpy as np
import pytest

from allocation_ope_bench.metrics import (
    mean_sharpe_ratio,
    selection_regret,
    sharpe_ratio_at_k,
    sharpe_ratio_curve,
)


def test_selection_regret_wrong_pick():
    true = [1.0, 2.0, 3.0]
    est = [3.0, 2.0, 1.0]  # estimator thinks policy 0 is best, truly worst
    out = selection_regret(true, est)
    assert out["selected_policy"] == 0
    assert out["best_policy"] == 2
    assert out["regret"] == pytest.approx(2.0)  # 3 - 1
    assert out["regret_normalized"] == pytest.approx(1.0)  # 2 / (3-1)
    assert out["correct"] is False


def test_selection_regret_correct_pick():
    out = selection_regret([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert out["regret"] == 0.0
    assert out["correct"] is True


def test_sharpe_ratio_at_k_known():
    true = [1.0, 2.0, 3.0, 4.0]
    est = [4.0, 3.0, 2.0, 1.0]  # est ranks policies 0,1,2,3
    # k=2 => topk {0,1}, true returns [1,2], best=2, risk=std([1,2])=0.5
    sr = sharpe_ratio_at_k(true, est, behavior_value=0.0, k=2)
    assert sr == pytest.approx((2.0 - 0.0) / 0.5)


def test_sharpe_ratio_k1_riskfree_is_inf():
    sr = sharpe_ratio_at_k([1.0, 2.0], [2.0, 1.0], behavior_value=0.0, k=1)
    assert np.isinf(sr)  # single policy => zero risk, positive excess


def test_sharpe_ratio_curve_covers_all_k():
    curve = sharpe_ratio_curve([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], behavior_value=0.5)
    assert set(curve.keys()) == {1, 2, 3}


def test_mean_sharpe_ratio_skips_degenerate_k1():
    # k=1 would be +inf; mean over k>=2 must stay finite.
    true = [1.0, 2.0, 3.0, 4.0]
    est = [4.0, 3.0, 2.0, 1.0]
    mean_sr = mean_sharpe_ratio(true, est, behavior_value=0.0, k_min=2)
    assert np.isfinite(mean_sr)

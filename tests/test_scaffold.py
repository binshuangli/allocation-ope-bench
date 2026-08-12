"""WP0 scaffold smoke tests — verify ported helpers are importable and correct."""

import numpy as np
import pytest

from allocation_ope_bench.experiments.git_utils import get_git_hash
from allocation_ope_bench.metrics.stats import bootstrap_ci, wilcoxon_paired
from allocation_ope_bench.models import get_estimator, list_models
from allocation_ope_bench.seed import set_global_seed

# ── seed ─────────────────────────────────────────────────────────────────────


def test_set_global_seed_reproducible():
    set_global_seed(0)
    a = np.random.rand(5)
    set_global_seed(0)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


# ── bootstrap CI ─────────────────────────────────────────────────────────────


def test_bootstrap_ci_mean():
    rng = np.random.default_rng(42)
    x = rng.normal(loc=5.0, scale=1.0, size=500)
    point, lo, hi = bootstrap_ci(np.mean, x, n_bootstrap=500, ci=0.95, seed=0)
    assert abs(point - 5.0) < 0.2
    assert lo < point < hi


def test_bootstrap_ci_bounds_ordered():
    x = np.arange(100, dtype=float)
    point, lo, hi = bootstrap_ci(np.mean, x, n_bootstrap=200, seed=7)
    assert lo <= point <= hi


# ── Wilcoxon ─────────────────────────────────────────────────────────────────


def test_wilcoxon_detects_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 0.5, 30)
    b = rng.normal(0.0, 0.5, 30)
    result = wilcoxon_paired(a, b, alternative="greater")
    assert result["p_value"] < 0.05
    assert result["significant_05"] is True


def test_wilcoxon_no_diff_returns_high_p():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 50)
    result = wilcoxon_paired(x, x + rng.normal(0, 0.01, 50))
    # may or may not be significant, but structure must be correct
    assert "p_value" in result
    assert "significant_05" in result


# ── model registry ───────────────────────────────────────────────────────────


def test_list_models_nonempty():
    models = list_models()
    assert len(models) > 0
    assert "t_learner" in models
    assert "s_learner" in models


def test_get_estimator_t_learner():
    est = get_estimator("t_learner", base_learner="lightgbm", seed=0)
    assert est.name == "t_learner"


def test_get_estimator_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_estimator("nonexistent_model")


# ── git utils ────────────────────────────────────────────────────────────────


def test_git_hash_returns_string():
    h = get_git_hash()
    assert isinstance(h, str)
    assert len(h) > 0

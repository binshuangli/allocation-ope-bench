"""WP1 — leakage guards: no treatment/outcome in features, disjoint train/eval."""

import numpy as np
import pytest

from allocation_ope_bench.data import (
    Dataset,
    assert_no_feature_leakage,
    make_synthetic,
    train_eval_split,
    true_allocation_value,
)
from allocation_ope_bench.models import get_estimator


def test_clean_synthetic_passes_leakage_guard():
    ds = make_synthetic(n=1000, seed=0)
    assert_no_feature_leakage(ds)  # should not raise


def test_treatment_in_features_is_caught():
    ds = make_synthetic(n=500, seed=1)
    X_leaky = np.column_stack([ds.X, ds.treatment.astype(float)])
    leaky = Dataset(
        name="leaky_t",
        X=X_leaky,
        treatment=ds.treatment,
        outcome=ds.outcome,
        cost=ds.cost,
        propensity=ds.propensity,
        has_ground_truth_effect=True,
        mu0=ds.mu0,
        mu1=ds.mu1,
    )
    with pytest.raises(AssertionError, match="treatment"):
        assert_no_feature_leakage(leaky)


def test_outcome_in_features_is_caught():
    ds = make_synthetic(n=500, seed=2)
    X_leaky = np.column_stack([ds.X, ds.outcome])
    leaky = Dataset(
        name="leaky_y",
        X=X_leaky,
        treatment=ds.treatment,
        outcome=ds.outcome,
        cost=ds.cost,
        propensity=ds.propensity,
        has_ground_truth_effect=True,
        mu0=ds.mu0,
        mu1=ds.mu1,
    )
    with pytest.raises(AssertionError, match="outcome"):
        assert_no_feature_leakage(leaky)


def test_train_eval_split_disjoint_and_covering():
    ds = make_synthetic(n=1000, seed=3)
    train_idx, eval_idx = train_eval_split(ds, eval_frac=0.5, seed=3)
    assert len(np.intersect1d(train_idx, eval_idx)) == 0
    assert len(train_idx) + len(eval_idx) == ds.n
    assert set(np.concatenate([train_idx, eval_idx]).tolist()) == set(range(ds.n))


def test_model_fit_on_train_only_then_value_on_eval():
    """End-to-end leakage discipline: fit score model on train, value on eval."""
    import pandas as pd

    ds = make_synthetic(n=4000, seed=5, effect_scale=1.0)
    train_idx, eval_idx = train_eval_split(ds, eval_frac=0.5, seed=5)
    assert len(np.intersect1d(train_idx, eval_idx)) == 0

    train = ds.subset(train_idx)
    eval_ds = ds.subset(eval_idx)

    est = get_estimator("t_learner", base_learner="lightgbm", seed=5)
    est.fit(
        pd.DataFrame(train.X, columns=train.feature_names),
        pd.Series(train.treatment),
        pd.Series(train.outcome),
    )
    scores = est.predict_uplift(pd.DataFrame(eval_ds.X, columns=eval_ds.feature_names))
    assert scores.shape == (eval_ds.n,)

    v = true_allocation_value(eval_ds, scores, budget_k=0.3)
    # A trained scorer should beat treating nobody (value at budget 0).
    v0 = true_allocation_value(eval_ds, scores, budget_k=0.0)
    assert v >= v0 - 1e-6

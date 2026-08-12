"""WP4 — experiment runner smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from allocation_ope_bench.data import make_synthetic, train_eval_split
from allocation_ope_bench.estimators import (
    fixed_target_estimators,
    needs_policy_kwargs,
)
from allocation_ope_bench.estimators.base import OutcomeModel
from allocation_ope_bench.metrics import compute_diagnostics
from allocation_ope_bench.models import list_models
from allocation_ope_bench.policies import AllocationPolicy, make_logged_data

# ── Model registry ────────────────────────────────────────────────────────────


def test_random_policy_in_registry():
    assert "random" in list_models()


def test_random_policy_gives_uniform_scores():
    from allocation_ope_bench.models import get_estimator

    ds = make_synthetic(n=200, seed=0)
    m = get_estimator("random", seed=0).fit(ds.X, ds.treatment, ds.outcome)
    scores = m.predict_uplift(ds.X)
    assert scores.shape == (200,)
    assert 0.0 <= scores.min() and scores.max() <= 1.0


# ── Subsample cap (max_n) ─────────────────────────────────────────────────────


def test_subsample_caps_large_dataset_and_preserves_propensity():
    from allocation_ope_bench.experiments.runner import _subsample

    ds = make_synthetic(n=5000, seed=0, rct_propensity=0.5)
    capped = _subsample(ds, max_n=1000, seed=0)
    assert capped.n == 1000
    # Constant RCT propensity is preserved under uniform subsampling.
    assert np.allclose(capped.propensity, 0.5)
    # No-op when already small enough.
    assert _subsample(ds, max_n=10000, seed=0).n == 5000
    assert _subsample(ds, max_n=None, seed=0).n == 5000


def test_subsample_is_seed_deterministic():
    from allocation_ope_bench.experiments.runner import _subsample

    ds = make_synthetic(n=3000, seed=1)
    a = _subsample(ds, max_n=500, seed=7)
    b = _subsample(ds, max_n=500, seed=7)
    assert np.array_equal(a.X, b.X)


# ── Leakage guard in runner ───────────────────────────────────────────────────


def test_runner_policy_fit_on_train_only():
    """Policy must be fit on train split; eval split must not be seen by model."""
    ds = make_synthetic(n=1000, seed=1)
    train_idx, eval_idx = train_eval_split(ds, eval_frac=0.5, seed=1)
    train_ds = ds.subset(train_idx)
    eval_ds = ds.subset(eval_idx)
    # Sanity: disjoint indices.
    assert len(set(train_idx) & set(eval_idx)) == 0
    # Policy fit on train — must not raise.
    policy = AllocationPolicy(uplift_model="t_learner", variant="deterministic", seed=1).fit(
        train_ds
    )
    # Score on eval — must work.
    scores = policy.score(eval_ds.X, feature_names=eval_ds.feature_names)
    assert scores.shape == (len(eval_idx),)


# ── One-cell integration test ─────────────────────────────────────────────────


def _run_one_cell(seed: int = 0, budget_k: float = 0.3, overlap_temp: float = 2.0):
    """Run a single (seed, budget_k, overlap_temp) cell, return (est_rows, sel_rows)."""
    from allocation_ope_bench.data import true_allocation_value
    from allocation_ope_bench.estimators import get_ope_estimator
    from allocation_ope_bench.metrics import selection_regret

    ds = make_synthetic(n=1000, seed=seed)
    train_idx, eval_idx = train_eval_split(ds, eval_frac=0.5, seed=seed)
    train_ds = ds.subset(train_idx)
    eval_ds = ds.subset(eval_idx)

    candidate_names = ["t_learner", "s_learner"]
    ests = fixed_target_estimators()
    est_rows, sel_true, sel_hat = [], {e: [] for e in ests}, {e: [] for e in ests}

    for cname in candidate_names:
        policy = AllocationPolicy(uplift_model=cname, variant="deterministic", seed=seed).fit(
            train_ds
        )
        scores = policy.score(eval_ds.X, feature_names=eval_ds.feature_names)
        logged = make_logged_data(eval_ds, scores, temperature=overlap_temp, seed=seed)
        true_val = true_allocation_value(eval_ds, scores, budget_k)
        target_prob = policy.action_prob(
            logged.context, budget_k, feature_names=eval_ds.feature_names
        )
        scores_logged = policy.score(logged.context, feature_names=eval_ds.feature_names)
        shared_om = OutcomeModel(seed=seed).fit(logged)
        diag = compute_diagnostics(logged, target_prob, budget_k)

        for ename in ests:
            est = get_ope_estimator(ename)
            if needs_policy_kwargs(ename):
                if ename == "cross_fitted_dr":
                    res = est.estimate(logged, budget_k=budget_k, seed=seed)
                else:
                    res = est.estimate(
                        logged,
                        target_prob,
                        outcome_model=shared_om,
                        scores=scores_logged,
                        budget_k=budget_k,
                        seed=seed,
                    )
            else:
                res = est.estimate(logged, target_prob, outcome_model=shared_om, seed=seed)
            est_rows.append(
                {
                    "estimator": ename,
                    "candidate": cname,
                    "true_value": true_val,
                    "value_hat": res.value,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    **{f"diag_{k}": v for k, v in diag.items()},
                }
            )
            if not needs_policy_kwargs(ename):
                sel_true[ename].append(true_val)
                sel_hat[ename].append(res.value)

    sel_rows = []
    for ename, tv in sel_true.items():
        if len(tv) >= 2:
            sr = selection_regret(np.array(tv), np.array(sel_hat[ename]))
            sel_rows.append({"estimator": ename, **sr})

    return est_rows, sel_rows


def test_one_cell_runs_all_estimators():
    est_rows, sel_rows = _run_one_cell()
    est_df = pd.DataFrame(est_rows)
    assert len(est_df) > 0
    # RQ1 fixed-target set runs (cross_fitted_dr is in the opt-bias experiment).
    assert set(est_df["estimator"].unique()) == set(fixed_target_estimators())
    assert "cross_fitted_dr" not in est_df["estimator"].values
    # All estimates must be finite.
    assert est_df["value_hat"].notna().all()
    assert est_df["value_hat"].apply(np.isfinite).all()
    # CI must be ordered.
    assert (est_df["ci_low"] <= est_df["value_hat"]).all()
    assert (est_df["value_hat"] <= est_df["ci_high"]).all()


def test_one_cell_diagnostics_present():
    est_rows, _ = _run_one_cell()
    est_df = pd.DataFrame(est_rows)
    for col in (
        "diag_ess",
        "diag_ess_fraction",
        "diag_max_weight",
        "diag_support_deficiency",
        "diag_budget_tightness",
    ):
        assert col in est_df.columns, f"Missing diagnostic column: {col}"
    assert (est_df["diag_ess_fraction"] >= 0).all()
    assert (est_df["diag_ess_fraction"] <= 1).all()


def test_selection_rows_produced():
    _, sel_rows = _run_one_cell()
    sel_df = pd.DataFrame(sel_rows)
    assert len(sel_df) > 0
    # Only selection-compatible estimators (not cross_fitted_dr / perturbation_dr).
    assert "cross_fitted_dr" not in sel_df["estimator"].values
    assert "perturbation_dr" not in sel_df["estimator"].values
    assert (
        "correct" in sel_df.columns
        or "correct_selection" in sel_df.columns
        or "correct" in sel_df.columns
    )


def test_overlap_temperature_affects_logged_n():
    """High temperature (smooth) retains more units after rejection sampling than low."""
    ds = make_synthetic(n=2000, seed=5)
    _, eval_idx = train_eval_split(ds, eval_frac=0.5, seed=5)
    eval_ds = ds.subset(eval_idx)
    policy = AllocationPolicy(variant="deterministic", seed=5).fit(
        ds.subset(list(set(range(ds.X.shape[0])) - set(eval_idx)))
    )
    scores = policy.score(eval_ds.X, feature_names=eval_ds.feature_names)
    logged_high = make_logged_data(eval_ds, scores, temperature=5.0, seed=5)
    logged_low = make_logged_data(eval_ds, scores, temperature=0.3, seed=5)
    # High temperature = smooth = better overlap = more units retained.
    assert logged_high.n >= logged_low.n


# ── Hydra runner integration (subprocess) ─────────────────────────────────────


@pytest.mark.slow
def test_hydra_smoke_run_writes_parquet(tmp_path):
    """Invoke the runner via subprocess with smoke config; check parquet output."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "allocation_ope_bench.experiments.runner",
            "experiment=smoke",
            f"results_dir={tmp_path}",
            "smoke=true",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(Path(__file__).parent.parent),
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
    assert result.returncode == 0, f"Runner failed:\n{result.stderr[-2000:]}"
    est_path = tmp_path / "estimates.parquet"
    assert est_path.exists(), "estimates.parquet not written"
    df = pd.read_parquet(est_path)
    assert len(df) > 0
    assert "estimator" in df.columns
    assert "true_value" in df.columns
    assert "value_hat" in df.columns
    assert "git_hash" in df.columns
    assert df["value_hat"].apply(np.isfinite).all()

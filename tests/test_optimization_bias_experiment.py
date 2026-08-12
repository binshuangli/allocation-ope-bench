"""WP5 — dedicated optimization-bias experiment (separate-regime evaluation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from allocation_ope_bench.data import make_synthetic, train_eval_split
from allocation_ope_bench.estimators import (
    fixed_target_estimators,
    optimization_bias_estimators,
)
from allocation_ope_bench.experiments.optimization_bias import _opt_bias_cell

# ── Registry partition ─────────────────────────────────────────────────────────


def test_cross_fitted_excluded_from_fixed_target():
    ft = fixed_target_estimators()
    assert "cross_fitted_dr" not in ft
    # The fixed-target set still has the standard estimators + perturbation_dr.
    for e in ("dm", "ips", "snips", "dr", "switch_dr", "bips", "perturbation_dr"):
        assert e in ft


def test_optimization_bias_set_is_dr_and_opt_aware():
    ob = optimization_bias_estimators()
    assert set(ob) == {"dr", "cross_fitted_dr", "perturbation_dr"}


# ── One opt-bias cell ──────────────────────────────────────────────────────────


def _one_cell(seed: int = 0, budget_k: float = 0.1):
    ds = make_synthetic(n=1600, seed=seed, effect_scale=1.0, noise_scale=2.0)
    train_idx, eval_idx = train_eval_split(ds, seed=seed)
    return _opt_bias_cell(
        dataset=ds,
        train_idx=train_idx,
        eval_idx=eval_idx,
        seed=seed,
        budget_k=budget_k,
        git_hash="testhash",
    )


def test_opt_bias_cell_produces_four_estimators():
    rows = _one_cell()
    df = pd.DataFrame(rows)
    assert set(df["estimator"]) == {
        "dr",
        "cross_fitted_dr",
        "perturbation_dr",
        "cross_fitted_dr_algo",
    }
    # The three FIXED-POLICY estimators share the same truth (the in-sample
    # policy's value). The fold-policy variant targets a DIFFERENT estimand
    # (the learning-algorithm value) and carries its own fold-matched truth.
    fixed = df[df.estimator != "cross_fitted_dr_algo"]
    assert fixed["true_value"].nunique() == 1
    assert df["regime"].unique().tolist() == ["optimization_bias"]
    # Finite estimates with ordered CIs.
    assert df["value_hat"].apply(np.isfinite).all()
    assert (df["ci_low"] <= df["value_hat"] + 1e-9).all()
    assert (df["value_hat"] <= df["ci_high"] + 1e-9).all()


def test_frozen_mode_estimand_matches_plain_dr():
    """cross_fitted_dr (frozen-policy mode) must evaluate the SAME policy as
    plain DR — the estimand-mismatch fix. Its estimate should be close to
    plain DR's on average (same target, different nuisance fitting), NOT
    tracking the fold-policy variant."""
    df = pd.DataFrame(_one_cell(seed=1, budget_k=0.3))
    fixed_truth = df.loc[df.estimator == "dr", "true_value"].iloc[0]
    cf_truth = df.loc[df.estimator == "cross_fitted_dr", "true_value"].iloc[0]
    assert fixed_truth == cf_truth  # same estimand by construction


def test_honest_pipeline_beats_plain_dr_on_average():
    """Averaged over seeds, the honest fold-policy pipeline (cross_fitted_dr_algo
    scored against its fold-matched truth) has lower |bias| than plain DR scored
    against the in-sample policy's truth. Each pipeline is scored against ITS OWN
    estimand — the estimand-coherent version of the optimizer's-curse claim.

    Note: frozen-policy nuisance cross-fitting alone (cross_fitted_dr) does NOT
    remove the curse — the policy-data correlation re-enters through the DR
    correction term — which is exactly why the honest pipeline is the remedy."""
    dr_b, algo_b = [], []
    for seed in range(4):
        df = pd.DataFrame(_one_cell(seed=seed, budget_k=0.1))
        dr_b.append(float(df.loc[df.estimator == "dr", "abs_bias"].iloc[0]))
        algo_b.append(float(df.loc[df.estimator == "cross_fitted_dr_algo", "abs_bias"].iloc[0]))
    assert np.mean(algo_b) < np.mean(dr_b)

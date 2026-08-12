"""WP6 — analysis aggregation, figures, and tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from allocation_ope_bench.analysis import aggregate as agg
from allocation_ope_bench.analysis import figures, tables

# ── Fixtures: small synthetic result frames ─────────────────────────────────────


def _estimates(n_seeds: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for ds, vscale in [("synthetic", 0.7), ("ihdp", 4.0)]:
        for est, err in [("dm", 0.02), ("ips", 0.2)]:
            for temp in (0.5, 2.0):
                for k in (0.1, 0.3):
                    for s in range(n_seeds):
                        tv = vscale * (1 + 0.05 * rng.standard_normal())
                        vh = tv + err * vscale * rng.standard_normal()
                        rows.append(
                            {
                                "dataset": ds,
                                "estimator": est,
                                "seed": s,
                                "overlap_temperature": temp,
                                "budget_k": k,
                                "candidate_policy": "t_learner",
                                "true_value": tv,
                                "value_hat": vh,
                                "bias": vh - tv,
                                "abs_bias": abs(vh - tv),
                                "rel_bias": (vh - tv) / abs(tv),
                                "diag_ess_fraction": 0.5 - 0.3 * (temp < 1),
                                "diag_support_deficiency": 0.1 * (temp < 1),
                                "diag_max_weight": 5.0,
                            }
                        )
    return pd.DataFrame(rows)


def _optbias() -> pd.DataFrame:
    """synthetic+ihdp have a positive (curse) DR bias; lenta has a tiny negative one."""
    rows = []
    specs = {
        "synthetic": (0.7, +0.18, 0.02),  # curse: DR optimistic
        "ihdp": (4.0, +0.24, -0.02),  # curse
        "lenta": (0.11, -0.007, -0.07),  # no curse (DR ~0, negative)
    }
    rng = np.random.default_rng(1)
    for ds, (v, dr_bias, cf_bias) in specs.items():
        for s in range(6):
            for est, b in [
                ("dr", dr_bias),
                ("cross_fitted_dr", cf_bias),
                ("perturbation_dr", dr_bias * 0.8),
            ]:
                bias = b + 0.01 * rng.standard_normal()
                rows.append(
                    {
                        "dataset": ds,
                        "estimator": est,
                        "seed": s,
                        "budget_k": 0.1,
                        "true_value": v,
                        "value_hat": v + bias,
                        "bias": bias,
                        "abs_bias": abs(bias),
                        "rel_bias": bias / v,
                    }
                )
    return pd.DataFrame(rows)


def _selection() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for est, rate in [("dm", 0.6), ("ips", 0.3)]:
        for _ in range(50):
            correct = rng.random() < rate
            rows.append(
                {
                    "dataset": "synthetic",
                    "estimator": est,
                    "correct_selection": correct,
                    "regret_normalized": 0.0 if correct else rng.uniform(0.1, 0.5),
                    "mean_sharpe_ratio_k2plus": rng.uniform(5, 12),
                }
            )
    return pd.DataFrame(rows)


# ── RQ1 ─────────────────────────────────────────────────────────────────────────


def test_rq1_accuracy_table_structure():
    tab = agg.rq1_accuracy_table(_estimates())
    # Cross-dataset averages are reported separately by reference type, never
    # pooled: exact-value ground truth and noisy HT references are not comparable.
    assert "mean_exact" in tab.columns
    assert "mean_across_datasets" not in tab.columns
    assert {"synthetic", "ihdp"}.issubset(set(tab.columns))
    # DM should be more accurate than IPS (lower relative RMSE) on average.
    assert tab.loc["dm", "mean_exact"] < tab.loc["ips", "mean_exact"]


def test_rq1_rel_rmse_is_scale_free():
    # Same relative error magnitude on a 0.7-scale and a 4.0-scale dataset should
    # give comparable relative RMSE (not 5x apart as raw RMSE would).
    cells = agg.rq1_cell_rel_rmse(_estimates())
    syn = cells[(cells.dataset == "synthetic") & (cells.estimator == "dm")]["rel_rmse"].median()
    ihdp = cells[(cells.dataset == "ihdp") & (cells.estimator == "dm")]["rel_rmse"].median()
    assert syn == pytest.approx(ihdp, abs=0.05)


def test_rq2_diagnostic_correlation_signs():
    # Construct estimates where higher support deficiency => higher IPS error and
    # higher ESS fraction => lower error; check the Spearman signs come out right.
    rng = np.random.default_rng(3)
    rows = []
    for i in range(120):
        defic = rng.uniform(0, 0.2)
        ess = rng.uniform(0.05, 0.9)
        err = 2.0 * defic - 0.3 * ess + 0.02 * rng.standard_normal()
        rows.append(
            {
                "dataset": "synthetic",
                "estimator": "ips",
                "seed": i,
                "overlap_temperature": 1.0,
                "budget_k": 0.3,
                "candidate_policy": "t",
                "true_value": 1.0,
                "value_hat": 1.0 + err,
                "bias": err,
                "abs_bias": abs(err),
                "rel_bias": err,
                "diag_ess_fraction": ess,
                "diag_support_deficiency": defic,
                "diag_max_weight": 1 + 10 * defic,
            }
        )
    corr = agg.rq2_diagnostic_correlation(pd.DataFrame(rows), estimator="ips").set_index(
        "diagnostic"
    )
    assert corr.loc["diag_ess_fraction", "spearman_rho"] < 0  # more support -> less error
    assert corr.loc["diag_support_deficiency", "spearman_rho"] > 0  # more deficiency -> more


# ── RQ3 — curse classification (the fixed bug) ──────────────────────────────────


def test_rq3_curse_requires_positive_dr_bias():
    a = agg.rq3_optimization_bias(_optbias())
    curse = a.groupby("dataset")["curse_present"].first()
    # Positive-bias datasets are flagged; the negative-bias one is NOT.
    assert curse["synthetic"] and curse["ihdp"]
    assert not curse["lenta"]


def test_rq3_debias_pct_only_for_curse_datasets():
    summ = agg.rq3_debias_summary(_optbias()).set_index("dataset")
    # Curse datasets report a finite de-biasing %; no-curse datasets are NaN.
    assert np.isfinite(summ.loc["synthetic", "cross_fitted_dr_pct_removed"])
    assert np.isnan(summ.loc["lenta", "cross_fitted_dr_pct_removed"])
    # cross-fitting removes a large share where the curse exists.
    assert summ.loc["synthetic", "cross_fitted_dr_pct_removed"] > 50


# ── RQ4 ─────────────────────────────────────────────────────────────────────────


def test_rq4_selection_table_ranks_by_correct_rate():
    tab = agg.rq4_selection_table(_selection())
    assert tab.iloc[0]["estimator"] == "dm"  # higher correct rate first
    assert tab.iloc[0]["correct_rate"] >= tab.iloc[-1]["correct_rate"]


# ── Figures + tables smoke ──────────────────────────────────────────────────────


def test_figures_generate(tmp_path):
    results = {
        "estimates": _estimates(),
        "optimization_bias": _optbias(),
        "selection": _selection(),
    }
    paths = figures.make_all_figures(results, tmp_path / "figures")
    assert len(paths) == 5
    for p in paths:
        assert p.exists() and p.suffix == ".pdf"


def test_tables_generate_latex(tmp_path):
    results = {
        "estimates": _estimates(),
        "optimization_bias": _optbias(),
        "selection": _selection(),
    }
    paths = tables.make_all_tables(results, tmp_path / "tables")
    # rq1 + rq3 + rq4-selection + rq4-reftype
    assert len(paths) == 4
    tex = (tmp_path / "tables" / "tab_rq1_accuracy.tex").read_text()
    assert r"\begin{tabular}" in tex and r"\bottomrule" in tex
    # perturbation_dr is excluded from the primary RQ1 ranking (estimand mismatch)
    assert "perturbation" not in tex


def test_load_results_roundtrip(tmp_path):
    est = _estimates()
    est.to_parquet(tmp_path / "estimates.parquet")
    out = agg.load_results(tmp_path)
    assert "estimates" in out and len(out["estimates"]) == len(est)

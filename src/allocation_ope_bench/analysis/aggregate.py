"""WP6 — aggregate the raw result parquets into the paper's analysis tables.

Design rule (the Paper 1 "metric is the message" guard): the cross-dataset
headline is **relative RMSE** = RMSE / |V_true|, never raw error, because outcome
scale spans ~100x across datasets (lenta V~0.11 vs IHDP V~4.4). Within a single
dataset, raw signed bias is reported (e.g. the optimization-bias regime).

Four research questions:

* RQ1 accuracy        — relative RMSE per (dataset, estimator), and vs budget.
* RQ2 trust           — error vs the overlap-temperature stress knob, plus the
                        diagnostic→error relationship (ESS / support deficiency).
* RQ3 optimization    — per-dataset *signed* bias of DR vs cross_fitted_dr vs
                        perturbation_dr, conditioned on whether the optimizer's
                        curse is actually present (plain-DR bias materially > 0).
* RQ4 selection       — correct-selection rate, normalized regret, SharpeRatio@k.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from allocation_ope_bench.metrics.error import cell_error

# PRIMARY estimators target the fixed deterministic policy V(pi_e) and are ranked
# directly. perturbation_dr targets a smoothed-policy estimand E_eps[V(pi_{s+eps})],
# so it is NOT estimand-coherent with a direct ranking against V(pi_e); it is
# reported separately (policy-smoothing sensitivity), never in the primary tables.
PRIMARY_ESTIMATORS = ["dm", "dr", "switch_dr", "snips", "ips", "bips"]
ESTIMATOR_ORDER = PRIMARY_ESTIMATORS  # primary display order (perturbation_dr excluded)
# RQ3 fixed-policy comparison: same-estimand estimators only (perturbation_dr excluded).
OPTBIAS_ORDER = ["dr", "cross_fitted_dr", "cross_fitted_dr_algo"]

# A dataset is in the "optimizer's-curse-present" regime if plain DR's mean signed
# bias is POSITIVE (optimistic — the defining direction of the curse) and exceeds
# this fraction of |V_true| in the optimization-bias experiment. A negative or
# negligible plain-DR bias means there is no curse to remove, so the de-biasing %
# of an optimization-aware estimator is undefined there (cross-fitting only adds
# variance).
CURSE_REL_BIAS_THRESHOLD = 0.02


def load_results(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the three result frames (estimates, selection, optimization_bias)."""
    results_dir = Path(results_dir)
    out: dict[str, pd.DataFrame] = {}
    for name in ("estimates", "selection", "optimization_bias"):
        p = results_dir / f"{name}.parquet"
        if p.exists():
            out[name] = pd.read_parquet(p)
    if "estimates" not in out:
        raise FileNotFoundError(f"No estimates.parquet in {results_dir}")
    # RQ4 selection is degenerate at a full budget (k covering everyone): every
    # top-k policy treats all units, so all candidates share one value and "which
    # is best" is an arbitrary tie. Drop full-budget rows from the selection frame
    # so the reported selection quality is not contaminated by ties. (The raw
    # parquet is untouched; only the analysis view is filtered.)
    if "selection" in out and "budget_k" in out["selection"].columns:
        sel = out["selection"]
        out["selection"] = sel[sel["budget_k"] < 1.0].reset_index(drop=True)
    return out


# ── RQ1 — accuracy ─────────────────────────────────────────────────────────────


def rq1_cell_rel_rmse(estimates: pd.DataFrame) -> pd.DataFrame:
    """Relative RMSE per estimand cell (dataset, estimator, overlap, budget, policy).

    RMSE is taken across seeds within the cell, divided by |mean true_value|.
    """
    by = ["dataset", "estimator", "overlap_temperature", "budget_k", "candidate_policy"]
    rows = []
    for keys, g in estimates.groupby(by):
        rec = dict(zip(by, keys))
        ce = cell_error(g["value_hat"].to_numpy(), g["true_value"].to_numpy())
        rec.update({"rel_rmse": ce["rel_rmse"], "rmse": ce["rmse"], "abs_bias": ce["abs_bias"]})
        rows.append(rec)
    return pd.DataFrame(rows)


def rq1_accuracy_table(estimates: pd.DataFrame) -> pd.DataFrame:
    """Headline RQ1 table: median relative RMSE per (dataset, estimator).

    Median over cells is robust to the heavy tails of IPS-family estimators under
    poor overlap. Returns a (dataset x estimator) wide table plus a 'rank' helper.
    """
    cells = rq1_cell_rel_rmse(estimates)
    tab = (
        cells.groupby(["dataset", "estimator"])["rel_rmse"]
        .median()
        .reset_index()
        .pivot(index="estimator", columns="dataset", values="rel_rmse")
    )
    tab = tab.reindex([e for e in ESTIMATOR_ORDER if e in tab.index])
    # Report the cross-dataset average SEPARATELY for exact-value and
    # HT-reference datasets. A single pooled mean would mix error against exact
    # ground truth with error against a noisy reference estimate -- the pooling
    # this paper avoids elsewhere (Section 3, and Table rq4-reftype for RQ4).
    exact_cols = [c for c in tab.columns if c in EXACT_VALUE_DATASETS or c.startswith("acic")]
    rct_cols = [c for c in tab.columns if c not in exact_cols]
    if exact_cols:
        tab["mean_exact"] = tab[exact_cols].mean(axis=1)
    if rct_cols:
        tab["mean_rct"] = tab[rct_cols].mean(axis=1)
    return tab


def rq1_error_vs_budget(estimates: pd.DataFrame) -> pd.DataFrame:
    """Median relative RMSE per (dataset, estimator, budget_k) — the budget sweep."""
    cells = rq1_cell_rel_rmse(estimates)
    return cells.groupby(["dataset", "estimator", "budget_k"])["rel_rmse"].median().reset_index()


# ── RQ2 — trust diagnostics ─────────────────────────────────────────────────────


def rq2_error_vs_overlap(estimates: pd.DataFrame) -> pd.DataFrame:
    """Median relative RMSE per (dataset, estimator, overlap_temperature).

    Low temperature = sharp logging = poor overlap (the stress knob). Expect
    IPS-family error to rise as overlap worsens; DR-family to stay flatter.
    """
    cells = rq1_cell_rel_rmse(estimates)
    return (
        cells.groupby(["dataset", "estimator", "overlap_temperature"])["rel_rmse"]
        .median()
        .reset_index()
    )


def rq2_diagnostic_vs_error(estimates: pd.DataFrame, estimator: str = "ips") -> pd.DataFrame:
    """Row-level |relative bias| with trust diagnostics, for one estimator.

    Used for the trust-map scatter: does a measurable-from-logged-data diagnostic
    (ESS fraction, support deficiency) predict the realized error? Defaults to IPS,
    the estimator the diagnostics are designed to flag.
    """
    sub = estimates[estimates["estimator"] == estimator].copy()
    sub["abs_rel_bias"] = sub["rel_bias"].abs()
    cols = [
        "dataset",
        "overlap_temperature",
        "budget_k",
        "diag_ess_fraction",
        "diag_support_deficiency",
        "diag_max_weight",
        "abs_rel_bias",
    ]
    return sub[cols].reset_index(drop=True)


def rq2_diagnostic_correlation(estimates: pd.DataFrame, estimator: str = "ips") -> pd.DataFrame:
    """Spearman rank correlation between each trust diagnostic and |relative bias|.

    Quantifies the trust-map claim: a good diagnostic should correlate POSITIVELY
    with realized error for risk-signalling diagnostics (support deficiency,
    max weight) and NEGATIVELY for ESS fraction (more support => less error).
    """
    from scipy.stats import spearmanr

    d = rq2_diagnostic_vs_error(estimates, estimator=estimator)
    rows = []
    for diag in ("diag_ess_fraction", "diag_support_deficiency", "diag_max_weight"):
        x = d[diag].to_numpy()
        y = d["abs_rel_bias"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3 or np.ptp(x[mask]) == 0:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = spearmanr(x[mask], y[mask])
        rows.append({"diagnostic": diag, "spearman_rho": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows)


# ── RQ3 — optimization bias ─────────────────────────────────────────────────────


def rq3_optimization_bias(optbias: pd.DataFrame) -> pd.DataFrame:
    """Per-(dataset, estimator) signed & abs bias in the optimizer's-curse regime,
    plus a per-dataset 'curse_present' flag and cross_fitted_dr de-biasing %.
    """
    agg = (
        optbias.groupby(["dataset", "estimator"])
        .agg(
            mean_bias=("bias", "mean"),
            mean_abs_bias=("abs_bias", "mean"),
            mean_true=("true_value", "mean"),
        )
        .reset_index()
    )
    # Curse-present where plain DR's *positive* (optimistic) signed bias relative to
    # |V| exceeds threshold. Negative/negligible DR bias => no curse to remove.
    dr = agg[agg.estimator == "dr"].set_index("dataset")
    curse = (dr["mean_bias"] / dr["mean_true"].abs()) > CURSE_REL_BIAS_THRESHOLD
    agg["curse_present"] = agg["dataset"].map(curse).fillna(False)
    return agg


def rq3_debias_summary(optbias: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset % of plain-DR |bias| removed by each optimization-aware estimator.

    Reported per dataset (NOT pooled) — pooling hides that the curse only exists on
    datasets where plain DR is materially biased (continuous, known-effect data).
    """
    agg = rq3_optimization_bias(optbias)
    wide = agg.pivot(index="dataset", columns="estimator", values="mean_abs_bias")
    # |V_ref| per dataset (from the DR rows' mean_true) to report a scale-free bias:
    # raw |bias| is not comparable across datasets with very different outcome scales.
    vref = agg[agg.estimator == "dr"].set_index("dataset")["mean_true"].abs()
    curse = agg.groupby("dataset")["curse_present"].first()
    rows = []
    for ds in wide.index:
        dr = wide.loc[ds, "dr"]
        is_curse = bool(curse.get(ds, False))
        v = vref.get(ds, float("nan"))
        rec = {
            "dataset": ds,
            "curse_present": is_curse,
            "dr_abs_bias": dr,
            "dr_rel_bias": (dr / v) if v and v > 1e-12 else float("nan"),
            # Signed relative bias -- the quantity the material-bias flag actually
            # thresholds (>2% optimism), so the flag is auditable from the table.
            "dr_signed_rel_bias": (
                (agg[(agg.estimator == "dr") & (agg.dataset == ds)]["mean_bias"].iloc[0] / v)
                if v and v > 1e-12
                else float("nan")
            ),
        }
        # perturbation_dr excluded: it targets a smoothed-policy estimand, not the
        # fixed in-sample policy these columns compare against.
        for est in ("cross_fitted_dr", "cross_fitted_dr_algo"):
            # The de-biasing % is only meaningful where a curse exists (DR biased).
            # NOTE: cross_fitted_dr (frozen-policy nuisance cross-fitting) shares
            # DR's estimand; cross_fitted_dr_algo is the honest fold-policy
            # pipeline scored against its own fold-matched truth.
            if is_curse and est in wide.columns and dr > 1e-12:
                rec[f"{est}_pct_removed"] = 100.0 * (1.0 - wide.loc[ds, est] / dr)
            else:
                rec[f"{est}_pct_removed"] = float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


# ── RQ4 — selection ─────────────────────────────────────────────────────────────


def rq4_selection_table(selection: pd.DataFrame) -> pd.DataFrame:
    """Per-estimator selection quality: correct rate, normalized regret, SharpeRatio."""
    tab = (
        selection.groupby("estimator")
        .agg(
            correct_rate=("correct_selection", "mean"),
            mean_regret_norm=("regret_normalized", "mean"),
            mean_sharpe_k2plus=("mean_sharpe_ratio_k2plus", "mean"),
            n=("correct_selection", "count"),
        )
        .reset_index()
        .sort_values("correct_rate", ascending=False)
    )
    return tab


def rq4_selection_by_dataset(selection: pd.DataFrame) -> pd.DataFrame:
    """Correct-selection rate per (dataset, estimator) — the per-dataset breakdown."""
    return (
        selection.groupby(["dataset", "estimator"])["correct_selection"]
        .mean()
        .reset_index()
        .rename(columns={"correct_selection": "correct_rate"})
    )


# Datasets with exact potential-outcome truth vs those scored against a noisy
# HT reference estimate — on the latter, "correct" means agreement with the
# reference-best policy, not the genuinely value-maximizing one.
EXACT_VALUE_DATASETS = {"synthetic", "ihdp"}


def rq4_selection_by_reference_type(selection: pd.DataFrame) -> pd.DataFrame:
    """Correct-selection rate per estimator, split by exact-value vs RCT-reference
    datasets. On exact-value data 'correct' = recovering the truly best policy; on
    RCT data it = agreement with the noisy HT-reference-best policy."""
    sel = selection.copy()
    sel["ref_type"] = sel["dataset"].apply(
        lambda d: "exact" if d in EXACT_VALUE_DATASETS else "rct"
    )
    tab = (
        sel.groupby(["ref_type", "estimator"])["correct_selection"]
        .mean()
        .reset_index()
        .rename(columns={"correct_selection": "correct_rate"})
    )
    return tab.pivot(index="estimator", columns="ref_type", values="correct_rate")

"""Clustered inference + actionable thresholds for the RQ2 trust map.

The 9,450 evaluation cells are not independent: they share datasets, seeds,
candidate policies, budgets, and (within a dataset x seed) underlying
observations. A conventional Spearman p-value over all cells is therefore
anti-conservative. This module provides the review-grade replacements:

1. cluster bootstrap CIs for each diagnostic's Spearman rho with IPS
   |relative bias| — resampling whole clusters (dataset, and dataset x seed);
2. within-dataset correlations (removes all cross-dataset confounds);
3. leave-one-dataset-out (LODO) pooled correlations (between-dataset
   generalization of the pooled number);
4. a simple threshold rule fit on one benchmark run and validated on a
   DIFFERENT run (out-of-DGP): flag a cell as "unsafe" when ESS fraction is
   below t_ess or support deficiency above t_def; report the error rate among
   flagged / unflagged cells (false-safe rate) and each diagnostic's ROC-AUC
   for predicting |rel bias| > delta on the held-out run.

CLI:
    python -m allocation_ope_bench.analysis.trust_inference \
        --train-dir results/full_run --test-dir results/acic_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DIAGNOSTICS = ["diag_ess_fraction", "diag_support_deficiency", "diag_max_weight"]
ERROR_DELTA = 0.10  # "material error": |rel bias| > 10%


def _ips_frame(results_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(Path(results_dir) / "estimates.parquet")
    df = df[df.estimator == "ips"].copy()
    df["abs_rel_bias"] = df["rel_bias"].abs()
    return df.dropna(subset=DIAGNOSTICS + ["abs_rel_bias"])


def _rho(df: pd.DataFrame, diag: str) -> float:
    if df[diag].nunique() < 2:
        return float("nan")
    r, _ = spearmanr(df[diag], df["abs_rel_bias"])
    return float(r)


def cluster_bootstrap_ci(
    df: pd.DataFrame, diag: str, cluster_cols: list[str], n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for Spearman rho, resampling whole clusters."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(cluster_cols)]
    point = _rho(df, diag)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), size=len(groups))
        b = pd.concat([groups[i] for i in idx])
        boots.append(_rho(b, diag))
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def within_dataset_rhos(df: pd.DataFrame, diag: str) -> pd.Series:
    return df.groupby("dataset").apply(lambda g: _rho(g, diag), include_groups=False)


def lodo_rhos(df: pd.DataFrame, diag: str) -> pd.Series:
    out = {}
    for ds in df["dataset"].unique():
        out[ds] = _rho(df[df.dataset != ds], diag)
    return pd.Series(out)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the rank formulation (no sklearn dependency)."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_threshold(train: pd.DataFrame, delta: float = ERROR_DELTA) -> dict:
    """Pick (t_ess, t_def) on the train run: the per-diagnostic cut that
    maximizes Youden's J for predicting |rel bias| > delta."""
    y = (train["abs_rel_bias"] > delta).to_numpy()

    def best_cut(col: str, unsafe_below: bool) -> tuple[float, float]:
        vals = train[col]
        if not unsafe_below:
            # "unsafe above" diagnostics can be mostly zero (e.g. support
            # deficiency); grid over the positive mass so the cut is meaningful.
            pos = vals[vals > 0]
            vals = pos if len(pos) >= 20 else vals
        grid = np.quantile(vals, np.linspace(0.05, 0.95, 37))
        best_t, best_j = float("nan"), -np.inf
        for t in np.unique(grid):
            flag = (train[col] < t) if unsafe_below else (train[col] > t)
            tpr = flag[y].mean() if y.any() else 0.0
            fpr = flag[~y].mean() if (~y).any() else 0.0
            j = tpr - fpr
            if j > best_j:
                best_j, best_t = j, float(t)
        return best_t, best_j

    t_ess, j_ess = best_cut("diag_ess_fraction", unsafe_below=True)
    t_def, j_def = best_cut("diag_support_deficiency", unsafe_below=False)
    return {"delta": delta, "t_ess": t_ess, "j_ess": j_ess, "t_def": t_def, "j_def": j_def}


def evaluate_threshold(test: pd.DataFrame, rule: dict) -> dict:
    """Held-out performance of the flag rule: ESS < t_ess OR support-def > t_def."""
    y = (test["abs_rel_bias"] > rule["delta"]).to_numpy()
    flag = (
        (test["diag_ess_fraction"] < rule["t_ess"])
        | (test["diag_support_deficiency"] > rule["t_def"])
    ).to_numpy()
    out = {
        "n_cells": int(len(test)),
        "base_error_rate": float(y.mean()),
        "flag_rate": float(flag.mean()),
        "error_rate_flagged": float(y[flag].mean()) if flag.any() else float("nan"),
        "error_rate_unflagged": float(y[~flag].mean()) if (~flag).any() else float("nan"),
        "recall_of_errors": float(flag[y].mean()) if y.any() else float("nan"),
    }
    for diag in DIAGNOSTICS:
        score = -test[diag] if diag == "diag_ess_fraction" else test[diag]
        out[f"auc_{diag}"] = _auc(y, score.to_numpy())
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Clustered inference for the RQ2 trust map")
    ap.add_argument("--train-dir", default="results/full_run")
    ap.add_argument("--test-dir", default="results/acic_run")
    ap.add_argument("--out", default=None, help="output md path (default: train-dir)")
    args = ap.parse_args(argv)

    train = _ips_frame(Path(args.train_dir))
    test = _ips_frame(Path(args.test_dir))
    lines = ["# Trust-map clustered inference\n"]

    lines.append("## Spearman rho with cluster-bootstrap 95% CIs (main run)\n")
    lines.append("| diagnostic | rho | CI (dataset clusters) | CI (dataset x seed) |")
    lines.append("| --- | --- | --- | --- |")
    for diag in DIAGNOSTICS:
        pt, lo_d, hi_d = cluster_bootstrap_ci(train, diag, ["dataset"])
        _, lo_s, hi_s = cluster_bootstrap_ci(train, diag, ["dataset", "seed"])
        lines.append(
            f"| {diag} | {pt:+.3f} | [{lo_d:+.3f}, {hi_d:+.3f}] | [{lo_s:+.3f}, {hi_s:+.3f}] |"
        )

    lines.append("\n## Within-dataset rho (cross-dataset confounds removed)\n")
    wd = pd.DataFrame({d: within_dataset_rhos(train, d) for d in DIAGNOSTICS})
    lines.append("```\n" + str(wd.round(3)) + "\n```")

    lines.append("\n## Leave-one-dataset-out pooled rho\n")
    lodo = pd.DataFrame({d: lodo_rhos(train, d) for d in DIAGNOSTICS})
    lines.append("```\n" + str(lodo.round(3)) + "\n```")

    lines.append("\n## Threshold rule (fit on main run, validated out-of-DGP)\n")
    rule = fit_threshold(train)
    perf = evaluate_threshold(test, rule)
    lines.append(
        f"Rule (delta={rule['delta']}): flag when ESS fraction < {rule['t_ess']:.3f} "
        f"OR support deficiency > {rule['t_def']:.3f}\n"
    )
    lines.append(f"Held-out run: {args.test_dir}\n")
    for k, v in perf.items():
        lines.append(f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}")

    out_path = Path(args.out) if args.out else Path(args.train_dir) / "trust_inference.md"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

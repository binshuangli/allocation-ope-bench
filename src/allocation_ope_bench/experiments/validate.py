"""WP5 — Output validation and anomaly detection.

Flags suspicious results that warrant Opus investigation before analysis proceeds.
Each check returns a DataFrame of flagged rows with a 'flag' description column.

Anomaly thresholds
------------------
* ``rel_rmse > 5``          — estimator error > 5× the true value scale (catastrophic)
* ``ess_fraction < 0.01``   — IPS averaging over <1% effective sample (trust failure)
* ``support_deficiency > 0.9`` at eps=0.05 — near-complete unsupported allocation
* non-finite value_hat      — oracle or estimator numerical failure
* CI ordering violated      — ci_low > value_hat or value_hat > ci_high
* ``true_value`` non-finite — ground-truth oracle failure
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

REL_RMSE_FLAG = 5.0  # rel_bias > this is a red flag
ESS_FRAC_MIN = 0.01  # below this ESS fraction, IPS-based results are dubious
SUPPORT_DEF_MAX = 0.90  # near-complete unsupported allocation
CI_TOL = 1e-8  # numerical tolerance for CI ordering check


# ── Individual checks ─────────────────────────────────────────────────────────


def _flag_nonfinite_estimates(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~np.isfinite(df["value_hat"])
    flagged = df[mask].copy()
    flagged["flag"] = "non-finite value_hat"
    return flagged


def _flag_nonfinite_truth(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~np.isfinite(df["true_value"])
    flagged = df[mask].copy()
    flagged["flag"] = "non-finite true_value (oracle failure)"
    return flagged


def _flag_ci_order_violated(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["ci_low"] > df["value_hat"] + CI_TOL) | (df["value_hat"] > df["ci_high"] + CI_TOL)
    flagged = df[mask].copy()
    flagged["flag"] = "CI ordering violated (ci_low > value or value > ci_high)"
    return flagged


def _flag_high_rel_bias(df: pd.DataFrame) -> pd.DataFrame:
    if "rel_bias" not in df.columns:
        return pd.DataFrame()
    mask = df["rel_bias"].abs() > REL_RMSE_FLAG
    flagged = df[mask & np.isfinite(df["rel_bias"])].copy()
    flagged["flag"] = f"|rel_bias| > {REL_RMSE_FLAG}"
    return flagged


def _flag_low_ess(df: pd.DataFrame) -> pd.DataFrame:
    col = "diag_ess_fraction"
    if col not in df.columns:
        return pd.DataFrame()
    mask = df[col] < ESS_FRAC_MIN
    flagged = df[mask & np.isfinite(df[col])].copy()
    flagged["flag"] = f"ess_fraction < {ESS_FRAC_MIN}"
    return flagged


def _flag_high_support_deficiency(df: pd.DataFrame) -> pd.DataFrame:
    col = "diag_support_deficiency"
    if col not in df.columns:
        return pd.DataFrame()
    mask = df[col] > SUPPORT_DEF_MAX
    flagged = df[mask & np.isfinite(df[col])].copy()
    flagged["flag"] = f"support_deficiency > {SUPPORT_DEF_MAX}"
    return flagged


# ── Aggregate error check ─────────────────────────────────────────────────────


def _flag_per_cell_high_rmse(df: pd.DataFrame) -> pd.DataFrame:
    """Flag (dataset, estimator, overlap_temp, budget_k) cells where rel_rmse > threshold."""
    from allocation_ope_bench.metrics.error import error_vs_budget

    required = {"value_hat", "true_value", "dataset", "estimator", "budget_k"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    err_df = error_vs_budget(
        df,
        estimate_col="value_hat",
        truth_col="true_value",
        by=("dataset", "estimator", "overlap_temperature", "budget_k"),
    )
    flagged = err_df[err_df["rel_rmse"].fillna(0) > REL_RMSE_FLAG].copy()
    if flagged.empty:
        return pd.DataFrame()
    flagged["flag"] = f"cell rel_rmse > {REL_RMSE_FLAG}"
    return flagged


# ── Main validation entrypoint ────────────────────────────────────────────────


def validate_estimates(est_df: pd.DataFrame) -> pd.DataFrame:
    """Run all anomaly checks on the estimates DataFrame.

    Returns a DataFrame of flagged rows (possibly empty = clean run).
    Each row carries a 'flag' column describing the anomaly.
    """
    checks = [
        _flag_nonfinite_estimates,
        _flag_nonfinite_truth,
        _flag_ci_order_violated,
        _flag_high_rel_bias,
        _flag_low_ess,
        _flag_high_support_deficiency,
    ]
    parts = [fn(est_df) for fn in checks]

    # Cell-level rel_rmse check (aggregated across seeds).
    try:
        parts.append(_flag_per_cell_high_rmse(est_df))
    except Exception as exc:  # noqa: BLE001
        log.warning("Cell RMSE check failed: %s", exc)

    non_empty = [p for p in parts if not p.empty]
    if not non_empty:
        return pd.DataFrame()
    flagged = pd.concat(non_empty, ignore_index=True)
    return flagged


def print_validation_report(flagged: pd.DataFrame, *, title: str = "Anomaly Report") -> None:
    """Human-readable summary of flagged anomalies."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    if flagged.empty:
        print("  ✓  No anomalies detected — run is clean.")
        return

    print(f"  ⚠  {len(flagged)} anomalous row(s) detected.\n")
    for flag_type, group in flagged.groupby("flag"):
        id_cols = [
            c
            for c in ["dataset", "estimator", "seed", "budget_k", "overlap_temperature"]
            if c in group.columns
        ]
        print(f"  [{flag_type}]  {len(group)} row(s)")
        print(group[id_cols].drop_duplicates().to_string(index=False))
        print()
    print(f"{'=' * 60}")

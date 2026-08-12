"""Estimator accuracy metrics vs the true allocation value (RQ1).

Headline metric (WP3 decision): **relative RMSE** = RMSE / |V_true|, the obp /
SCOPE-RL convention, which is scale-free and therefore safe to aggregate across
datasets of different outcome magnitude (the Paper 1 cross-scale guard). Signed
bias and raw RMSE are always retained for the per-dataset view.

``V_true`` here is a positive outcome *level* (an expected outcome under the
allocation), so |V_true| is normally bounded away from zero; a small-denominator
floor returns NaN rather than a spurious huge ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DENOM_FLOOR = 1e-8


def cell_error(estimates, truths) -> dict:
    """Error summary for one (dataset, estimator, budget) cell across seeds.

    Parameters
    ----------
    estimates : array of value estimates V_hat across seeds, shape (s,).
    truths    : the true value(s) V_true — scalar or array broadcastable to (s,).
    """
    estimates = np.asarray(estimates, dtype=float)
    truths = np.asarray(truths, dtype=float)
    if truths.ndim == 0:
        truths = np.full_like(estimates, float(truths))
    err = estimates - truths

    bias = float(err.mean())
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = abs(float(np.mean(truths)))
    rel_rmse = rmse / denom if denom > _DENOM_FLOOR else float("nan")
    rel_bias = bias / denom if denom > _DENOM_FLOOR else float("nan")

    return {
        "bias": bias,
        "abs_bias": abs(bias),
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "rel_bias": rel_bias,
        "true_value": float(np.mean(truths)),
        "n_seeds": int(estimates.size),
    }


def error_vs_budget(
    df: pd.DataFrame,
    estimate_col: str = "estimate",
    truth_col: str = "true_value",
    by: tuple[str, ...] = ("dataset", "estimator", "budget_k"),
) -> pd.DataFrame:
    """Aggregate a tidy results frame into error metrics grouped by ``by``.

    Each group (typically one dataset × estimator × budget across seeds) yields
    one row of :func:`cell_error` metrics — the RQ1 error-vs-budget table.
    """
    by = [c for c in by if c in df.columns]
    rows = []
    for keys, g in df.groupby(list(by)):
        rec = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        rec.update(cell_error(g[estimate_col].to_numpy(), g[truth_col].to_numpy()))
        rows.append(rec)
    return pd.DataFrame(rows)

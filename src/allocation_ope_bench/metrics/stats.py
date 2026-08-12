"""Statistical helpers: bootstrapped CIs and paired Wilcoxon test."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as scipy_stats


def bootstrap_ci(
    fn: Callable[..., float],
    *arrays,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    **fn_kwargs,
) -> tuple[float, float, float]:
    """Non-parametric bootstrap confidence interval for any scalar metric.

    Parameters
    ----------
    fn         : metric function with signature fn(*arrays, **fn_kwargs) -> float
    *arrays    : aligned arrays passed to fn (all same length n)
    n_bootstrap: number of bootstrap resamples
    ci         : confidence level (e.g. 0.95 for 95% CI)
    seed       : random seed for reproducibility
    **fn_kwargs: additional keyword arguments forwarded to fn

    Returns
    -------
    (point_estimate, lower_bound, upper_bound)
    """
    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    rng = np.random.default_rng(seed)

    point = fn(*arrays, **fn_kwargs)
    boot_vals = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled = tuple(a[idx] for a in arrays)
        try:
            boot_vals[i] = fn(*resampled, **fn_kwargs)
        except Exception:
            boot_vals[i] = float("nan")

    alpha = 1 - ci
    lower = float(np.nanpercentile(boot_vals, 100 * alpha / 2))
    upper = float(np.nanpercentile(boot_vals, 100 * (1 - alpha / 2)))
    return float(point), lower, upper


def wilcoxon_paired(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided",
) -> dict:
    """Paired Wilcoxon signed-rank test between two sets of per-fold metric scores.

    Parameters
    ----------
    scores_a   : metric values for model A across folds/seeds, shape (k,)
    scores_b   : metric values for model B across folds/seeds, shape (k,)
    alternative: 'two-sided', 'greater', or 'less'
                 'greater' tests that A > B

    Returns
    -------
    dict with keys: statistic, p_value, significant_05, significant_01
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    diff = scores_a - scores_b

    # Drop pairs where diff == 0 (Wilcoxon requires non-zero differences)
    nonzero = diff != 0
    if nonzero.sum() < 2:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "significant_05": False,
            "significant_01": False,
        }

    result = scipy_stats.wilcoxon(diff[nonzero], alternative=alternative)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant_05": bool(result.pvalue < 0.05),
        "significant_01": bool(result.pvalue < 0.01),
    }

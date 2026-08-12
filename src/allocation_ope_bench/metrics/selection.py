"""Policy-selection quality metrics (RQ4).

Given several candidate allocation policies, does the estimator pick the right
one under a budget?

* **Selection regret** — true-value gap between the policy the estimator selects
  (argmax of the estimate) and the truly best policy. Zero = perfect selection.
* **SharpeRatio@k** (Kiyohara & Saito, SCOPE-RL, ICLR 2024) — deployment
  risk-return: among the top-k policies ranked by the *estimate*, the best
  *true* return over the behavior policy, divided by the risk (std of the true
  returns in the top-k). Rewards selecting high-return, low-variance policies.
"""

from __future__ import annotations

import numpy as np


def selection_regret(true_values, estimated_values) -> dict:
    """True-value regret of the estimator's selected policy.

    Parameters
    ----------
    true_values      : true V_true per candidate policy, shape (m,).
    estimated_values : estimator V_hat per candidate policy, shape (m,).
    """
    true_values = np.asarray(true_values, dtype=float)
    estimated_values = np.asarray(estimated_values, dtype=float)
    if true_values.shape != estimated_values.shape or true_values.ndim != 1:
        raise ValueError("true_values and estimated_values must be 1-D, same length")

    selected = int(np.argmax(estimated_values))
    best = int(np.argmax(true_values))
    regret = float(true_values[best] - true_values[selected])
    spread = float(true_values.max() - true_values.min())
    return {
        "regret": regret,
        "regret_normalized": regret / spread if spread > 1e-12 else 0.0,
        "selected_policy": selected,
        "best_policy": best,
        "correct": selected == best,
    }


def sharpe_ratio_at_k(true_values, estimated_values, behavior_value: float, k: int) -> float:
    """SCOPE-RL SharpeRatio@k.

    SharpeRatio@k = ( max_{j in topk} V_true(j) - V_behavior ) / std( V_true(topk) ),
    where topk are the k policies with the highest *estimated* value.
    Returns +inf for a risk-free gain (std 0, positive numerator) and 0.0 for a
    risk-free non-gain, matching the degenerate-portfolio convention.
    """
    true_values = np.asarray(true_values, dtype=float)
    estimated_values = np.asarray(estimated_values, dtype=float)
    m = true_values.size
    if not 1 <= k <= m:
        raise ValueError(f"k must be in [1, {m}]")

    topk = np.argsort(-estimated_values, kind="stable")[:k]
    returns = true_values[topk]
    best = float(returns.max())
    risk = float(returns.std())
    excess = best - behavior_value
    if risk <= 1e-12:
        return float("inf") if excess > 0 else 0.0
    return excess / risk


def sharpe_ratio_curve(true_values, estimated_values, behavior_value: float) -> dict:
    """SharpeRatio@k for k = 1..m (the full risk-return curve).

    Note: ``k=1`` is a degenerate single-policy "portfolio" (zero risk => +inf for
    any gain); analysis should emphasize ``k>=2`` (see :func:`mean_sharpe_ratio`).
    """
    m = np.asarray(true_values).size
    return {
        k: sharpe_ratio_at_k(true_values, estimated_values, behavior_value, k)
        for k in range(1, m + 1)
    }


def mean_sharpe_ratio(
    true_values, estimated_values, behavior_value: float, k_min: int = 2
) -> float:
    """Mean SharpeRatio@k over k = k_min..m (finite values only).

    Defaults to ``k_min=2`` so the degenerate, always-+inf ``k=1`` point does not
    dominate the headline risk-return summary.
    """
    m = np.asarray(true_values).size
    if m < k_min:
        return float("nan")
    vals = [
        sharpe_ratio_at_k(true_values, estimated_values, behavior_value, k)
        for k in range(k_min, m + 1)
    ]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")

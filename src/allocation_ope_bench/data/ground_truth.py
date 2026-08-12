"""Ground-truth value of a budget-constrained allocation policy.

The estimand (the thing every offline estimator is scored against):

    V(pi) = E[ Y( a_pi(X) ) ]                                    (gross outcome)

where ``a_pi`` allocates the treatment to the highest-scoring units until the
budget is exhausted (a hard constraint on *who* may be treated; cost is not
subtracted from the outcome — see WP1 estimand decision).

Two computation paths, by dataset type:

* ``has_ground_truth_effect`` (synthetic, IHDP): exact, from the known
  potential-outcome means — ``V = mean( a*mu1 + (1-a)*mu0 )``.
* RCT / Jobs randomized subset: **known-propensity IPS** (Horvitz-Thompson) on
  the held-out randomized split — unbiased and model-free because the
  randomization probability is known:

      V = mean( 1{T_i = a_i} * Y_i / P(T_i = a_i) ),
      P(T_i = a_i) = p_i   if a_i = 1 else 1 - p_i.

Both paths operate on the *evaluation* subset (the held-out randomized split),
so the caller passes a `Dataset` already restricted to those units.
"""

from __future__ import annotations

import numpy as np

from allocation_ope_bench.data.base import Dataset


def allocate_under_budget(
    scores: np.ndarray,
    costs: np.ndarray,
    budget_k: float,
) -> np.ndarray:
    """Greedy top-score allocation under a cost budget.

    The budget is ``budget_k`` of the *total* cost across all units. Units are
    taken in descending score order while cumulative cost stays within budget.
    With unit costs this reduces to "treat the top ``budget_k`` fraction".

    Parameters
    ----------
    scores   : per-unit allocation score (higher = treat first), shape (n,)
    costs    : per-unit treatment cost, shape (n,)
    budget_k : budget as a fraction of total cost, in [0, 1]

    Returns
    -------
    actions  : binary allocation, shape (n,) — 1 = treat, 0 = withhold.
    """
    scores = np.asarray(scores, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if scores.shape != costs.shape:
        raise ValueError("scores and costs must have the same shape")
    if not 0.0 <= budget_k <= 1.0:
        raise ValueError("budget_k must lie in [0, 1]")

    n = scores.shape[0]
    actions = np.zeros(n, dtype=int)
    if budget_k <= 0.0:
        return actions
    if budget_k >= 1.0:
        return np.ones(n, dtype=int)

    budget = budget_k * costs.sum()
    # Stable sort so ties resolve deterministically by original order.
    order = np.argsort(-scores, kind="stable")
    cumcost = np.cumsum(costs[order])
    selected = order[cumcost <= budget]
    actions[selected] = 1
    return actions


def true_allocation_value(
    eval_data: Dataset,
    scores: np.ndarray,
    budget_k: float,
) -> float:
    """True value of allocating by ``scores`` under budget ``budget_k``.

    Parameters
    ----------
    eval_data : Dataset restricted to the held-out evaluation units. For RCT /
                Jobs this must be the randomized split; ``scores`` align to it.
    scores    : per-unit allocation score from the policy, shape (eval_data.n,).
    budget_k  : budget fraction in [0, 1].

    Returns
    -------
    The ground-truth expected outcome under the allocation, a float.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (eval_data.n,):
        raise ValueError(
            f"scores must have shape ({eval_data.n},), got {scores.shape}"
        )

    actions = allocate_under_budget(scores, eval_data.cost, budget_k)

    if eval_data.has_ground_truth_effect:
        values = np.where(actions == 1, eval_data.mu1, eval_data.mu0)
        return float(values.mean())

    # Known-propensity IPS oracle on the randomized holdout.
    p = eval_data.propensity
    p_match = np.where(actions == 1, p, 1.0 - p)
    matched = (eval_data.treatment == actions).astype(float)
    return float(np.mean(matched * eval_data.outcome / p_match))

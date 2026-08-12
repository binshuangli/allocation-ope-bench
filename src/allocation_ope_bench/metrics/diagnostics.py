"""Trust diagnostics computed under the allocation policy (RQ2).

These are the *measurable-from-logged-data* predictors of whether offline
evaluation is trustworthy — the features of the practitioner trust map:

* **ESS** — effective sample size of the target-policy importance weights,
  ``(sum w)^2 / sum w^2``; reported as a fraction of n. Low ESS => IPS is
  effectively averaging over a handful of units.
* **max importance weight** — the single largest weight; a few huge weights
  signal a fragile estimate.
* **support deficiency** (WP3 decision) — share of the target policy's
  chosen-action *mass* whose logging propensity ``pi_b(a | x) < eps``, i.e.,
  effectively unsupported. Directly measures the common-support violation.
* **budget tightness** — ``1 - budget_k``; tighter budgets (small k) make the
  deterministic target more selective and overlap scarcer.

All take the target ``prob_treat`` (pi_e(1|x)) plus the logged action / pscore;
support deficiency additionally needs the logging treat-probability pi_b(1|x).
"""

from __future__ import annotations

import numpy as np

from allocation_ope_bench.estimators.base import LoggedData, policy_action_prob


def importance_weights(target_prob_treat, action, pscore) -> np.ndarray:
    """w_i = pi_e(a_i | x_i) / pi_b(a_i | x_i)."""
    pi_e = policy_action_prob(np.asarray(target_prob_treat, float), np.asarray(action, int))
    return pi_e / np.asarray(pscore, float)


def effective_sample_size(weights) -> float:
    w = np.asarray(weights, dtype=float)
    denom = np.sum(w**2)
    return float((w.sum() ** 2) / denom) if denom > 0 else 0.0


def max_importance_weight(weights) -> float:
    w = np.asarray(weights, dtype=float)
    return float(w.max()) if w.size else float("nan")


def support_deficiency(target_prob_treat, logging_prob_treat, eps: float = 0.05) -> float:
    """Share of target chosen-action mass landing on unsupported (x, a).

    deficiency = mean_i [ pi_e(1|x) 1{pi_b(1|x) < eps} + pi_e(0|x) 1{pi_b(0|x) < eps} ].

    Since sum_a pi_e(a|x) = 1, this is a weighted fraction in [0, 1].
    """
    pe1 = np.asarray(target_prob_treat, dtype=float)
    pb1 = np.asarray(logging_prob_treat, dtype=float)
    unsupported_treat = (pb1 < eps).astype(float)
    unsupported_withhold = ((1.0 - pb1) < eps).astype(float)
    mass = pe1 * unsupported_treat + (1.0 - pe1) * unsupported_withhold
    return float(mass.mean())


def support_deficiency_sensitivity(
    target_prob_treat,
    logging_prob_treat,
    eps_grid=(0.01, 0.05, 0.10),
) -> dict:
    """Support deficiency across an eps grid — robustness of the threshold choice.

    Monotone non-decreasing in eps (a looser threshold flags at least as much
    mass as unsupported)."""
    return {
        eps: support_deficiency(target_prob_treat, logging_prob_treat, eps=eps) for eps in eps_grid
    }


def budget_tightness(budget_k: float) -> float:
    """Simplest tightness proxy: fraction of budget withheld (tighter => larger)."""
    return float(1.0 - budget_k)


def compute_diagnostics(
    logged: LoggedData,
    target_prob_treat,
    budget_k: float,
    eps: float = 0.05,
) -> dict:
    """All trust diagnostics for one logged sample under the target policy."""
    w = importance_weights(target_prob_treat, logged.action, logged.pscore)
    ess = effective_sample_size(w)
    diag = {
        "ess": ess,
        "ess_fraction": ess / logged.n if logged.n else float("nan"),
        "max_weight": max_importance_weight(w),
        "budget_tightness": budget_tightness(budget_k),
    }
    if logged.logging_prob_treat is not None:
        # Headline at eps, plus a sensitivity check at 0.01 / 0.05 / 0.10.
        diag["support_deficiency"] = support_deficiency(
            target_prob_treat, logged.logging_prob_treat, eps=eps
        )
        diag["support_deficiency_eps001"] = support_deficiency(
            target_prob_treat, logged.logging_prob_treat, eps=0.01
        )
        diag["support_deficiency_eps010"] = support_deficiency(
            target_prob_treat, logged.logging_prob_treat, eps=0.10
        )
    else:
        diag["support_deficiency"] = float("nan")
        diag["support_deficiency_eps001"] = float("nan")
        diag["support_deficiency_eps010"] = float("nan")
    return diag

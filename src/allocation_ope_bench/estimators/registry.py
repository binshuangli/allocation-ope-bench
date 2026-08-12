"""Estimator registry — instantiate any OPE estimator by name.

Two optimization-aware estimators target the optimization-bias condition (policy
learned on the same logged data it is evaluated on):

* ``cross_fitted_dr`` — the recommended, principled one (sample splitting).
* ``perturbation_dr`` — perturbation-smoothed DR (Guo-inspired), a secondary
  comparison (see ``estimators.perturbation_dr``).

Both need extra ``budget_k`` (and ``scores`` for the perturbation variant) at
estimate time. The optimization-aware set is flagged for the WP3 protocol gate.
"""

from __future__ import annotations

from allocation_ope_bench.estimators.base import (
    EstimateResult,
    LoggedData,
    OPEEstimator,
    OutcomeModel,
)
from allocation_ope_bench.estimators.cross_fitted import CrossFittedDR
from allocation_ope_bench.estimators.direct import DirectMethod
from allocation_ope_bench.estimators.doubly_robust import DoublyRobust, SwitchDR
from allocation_ope_bench.estimators.ips import BIPS, IPS, SNIPS, ClippedIPS
from allocation_ope_bench.estimators.perturbation_dr import PerturbationSmoothedDR
from allocation_ope_bench.estimators.shrinkage import ShrinkageDR

_REGISTRY: dict[str, type[OPEEstimator]] = {
    "dm": DirectMethod,
    "ips": IPS,
    "snips": SNIPS,
    "clipped_ips": ClippedIPS,
    "dr": DoublyRobust,
    "switch_dr": SwitchDR,
    "bips": BIPS,
    "cross_fitted_dr": CrossFittedDR,
    "shrinkage_dr": ShrinkageDR,
    "perturbation_dr": PerturbationSmoothedDR,
}

# Estimators needing extra estimate-time kwargs beyond (logged, target_prob_treat).
_NEEDS_POLICY_KWARGS = ("cross_fitted_dr", "perturbation_dr")

# RQ1 fixed-target accuracy set: estimators that evaluate the *given* target
# policy. cross_fitted_dr is excluded — it derives its OWN allocation per fold, so
# its value targets a different policy than the candidate's true value (an
# apples-to-oranges comparison in the fixed-target table). perturbation_dr stays:
# it smooths the *given* policy's allocation, so it still targets the candidate.
_FIXED_TARGET = ("dm", "ips", "snips", "dr", "switch_dr", "bips", "perturbation_dr")

# Optimization-bias regime set: when the policy is fit on the same data it is
# evaluated on (optimizer's curse), compare plain DR (biased baseline) against the
# two optimization-aware estimators. Each is scored against the true value of the
# in-sample-optimized policy.
_OPTIMIZATION_BIAS = ("dr", "cross_fitted_dr", "perturbation_dr")


def list_estimators() -> list[str]:
    return sorted(_REGISTRY.keys())


def fixed_target_estimators() -> list[str]:
    """RQ1 estimators that evaluate the given target policy (excludes cross_fitted_dr)."""
    return list(_FIXED_TARGET)


def optimization_bias_estimators() -> list[str]:
    """Estimators compared in the optimizer's-curse regime (plain DR + opt-aware)."""
    return list(_OPTIMIZATION_BIAS)


def needs_policy_kwargs(name: str) -> bool:
    return name in _NEEDS_POLICY_KWARGS


def get_ope_estimator(name: str, **kwargs) -> OPEEstimator:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown estimator {name!r}. Available: {list_estimators()}")
    return _REGISTRY[name](**kwargs)


__all__ = [
    "get_ope_estimator",
    "list_estimators",
    "needs_policy_kwargs",
    "EstimateResult",
    "LoggedData",
    "OPEEstimator",
    "OutcomeModel",
]

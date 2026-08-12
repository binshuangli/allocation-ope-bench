"""Hyperparameter sweep helper for estimator sensitivity analysis.

Switch-DR's ``tau`` and BIPS's ``alpha`` are data-tuned by default ("auto"); this
helper produces the *sensitivity sweep* alternative — the estimate across a grid
of values — so WP6 can report robustness rather than rely on a single setting.
"""

from __future__ import annotations

from allocation_ope_bench.estimators.base import EstimateResult


def hyperparam_sweep(
    estimator_cls,
    param: str,
    grid,
    logged,
    target_prob_treat,
    **estimate_kwargs,
) -> dict[float, EstimateResult]:
    """Estimate across a grid of one hyperparameter (e.g. tau or alpha)."""
    results: dict[float, EstimateResult] = {}
    for value in grid:
        est = estimator_cls(**{param: value})
        results[value] = est.estimate(logged, target_prob_treat, **estimate_kwargs)
    return results

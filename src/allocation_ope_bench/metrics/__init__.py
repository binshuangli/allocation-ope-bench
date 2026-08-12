"""Metrics: estimator error (RQ1), diagnostics (RQ2), selection (RQ4), stats."""

from allocation_ope_bench.metrics.diagnostics import (
    budget_tightness,
    compute_diagnostics,
    effective_sample_size,
    importance_weights,
    max_importance_weight,
    support_deficiency,
    support_deficiency_sensitivity,
)
from allocation_ope_bench.metrics.error import cell_error, error_vs_budget
from allocation_ope_bench.metrics.selection import (
    mean_sharpe_ratio,
    selection_regret,
    sharpe_ratio_at_k,
    sharpe_ratio_curve,
)
from allocation_ope_bench.metrics.stats import bootstrap_ci, wilcoxon_paired

__all__ = [
    # error (RQ1)
    "cell_error",
    "error_vs_budget",
    # diagnostics (RQ2)
    "importance_weights",
    "effective_sample_size",
    "max_importance_weight",
    "support_deficiency",
    "support_deficiency_sensitivity",
    "budget_tightness",
    "compute_diagnostics",
    # selection (RQ4)
    "selection_regret",
    "sharpe_ratio_at_k",
    "sharpe_ratio_curve",
    "mean_sharpe_ratio",
    # stats
    "bootstrap_ci",
    "wilcoxon_paired",
]

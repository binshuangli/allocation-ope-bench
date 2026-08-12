"""Uplift / CATE model wrappers (ported from uplift-honest-bench)."""

from allocation_ope_bench.models.base import UpliftEstimator
from allocation_ope_bench.models.registry import get_estimator, list_models

__all__ = ["UpliftEstimator", "get_estimator", "list_models"]

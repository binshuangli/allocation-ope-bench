"""Data loaders, synthetic generator, and ground-truth allocation value."""

from __future__ import annotations

from typing import Callable

from allocation_ope_bench.data.base import (
    Dataset,
    assert_no_feature_leakage,
    train_eval_split,
)
from allocation_ope_bench.data.ground_truth import (
    allocate_under_budget,
    true_allocation_value,
)
from allocation_ope_bench.data.acic_dgp import make_acic_ihdp
from allocation_ope_bench.data.synthetic import make_synthetic

# Real / semi-synthetic loaders are imported lazily so that `import
# allocation_ope_bench.data` stays cheap and offline-safe.
_LOADERS: dict[str, str] = {
    "synthetic": "make_synthetic",  # handled specially (no download)
    "hillstrom": "load_hillstrom",
    "lenta": "load_lenta",
    "x5": "load_x5",
    "criteo": "load_criteo",
    "ihdp": "load_ihdp",
    "jobs": "load_jobs",
}


def list_datasets() -> list[str]:
    return list(_LOADERS.keys())


def load_dataset(name: str, **kwargs) -> Dataset:
    """Load any benchmark dataset by name."""
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list_datasets()}")
    if name == "synthetic":
        return make_synthetic(**kwargs)
    import importlib

    loaders = importlib.import_module("allocation_ope_bench.data.loaders")
    fn: Callable[..., Dataset] = getattr(loaders, _LOADERS[name])
    return fn(**kwargs)


__all__ = [
    "Dataset",
    "train_eval_split",
    "assert_no_feature_leakage",
    "allocate_under_budget",
    "true_allocation_value",
    "make_synthetic",
    "make_acic_ihdp",
    "load_dataset",
    "list_datasets",
]

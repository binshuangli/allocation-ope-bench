"""Model registry — get any estimator by name."""

from __future__ import annotations

from typing import Any, Literal

from allocation_ope_bench.models.base import UpliftEstimator

# Map of name -> (module_path, class_name, extra_kwargs)
_REGISTRY: dict[str, tuple[str, str, dict]] = {
    # ---- baselines ----
    "random": ("allocation_ope_bench.models.baselines", "RandomUpliftEstimator", {}),
    "solo_model": ("allocation_ope_bench.models.baselines", "SoloModelEstimator", {}),
    "two_model": ("allocation_ope_bench.models.baselines", "TwoModelEstimator", {}),
    "class_transformation": (
        "allocation_ope_bench.models.baselines",
        "ClassTransformationEstimator",
        {},
    ),
    # ---- meta-learners ----
    "s_learner": ("allocation_ope_bench.models.meta_learners", "SLearnerEstimator", {}),
    "t_learner": ("allocation_ope_bench.models.meta_learners", "TLearnerEstimator", {}),
    "x_learner": ("allocation_ope_bench.models.meta_learners", "XLearnerEstimator", {}),
    "r_learner": ("allocation_ope_bench.models.meta_learners", "RLearnerEstimator", {}),
    "dr_learner": ("allocation_ope_bench.models.meta_learners", "DRLearnerEstimator", {}),
    # ---- forest / direct uplift ----
    "causal_forest": ("allocation_ope_bench.models.forest_learners", "CausalForestEstimator", {}),
    "uplift_rf_kl": (
        "allocation_ope_bench.models.forest_learners",
        "UpliftRFEstimator",
        {"evaluationFunction": "KL"},
    ),
    "uplift_rf_ed": (
        "allocation_ope_bench.models.forest_learners",
        "UpliftRFEstimator",
        {"evaluationFunction": "ED"},
    ),
    "uplift_rf_chi": (
        "allocation_ope_bench.models.forest_learners",
        "UpliftRFEstimator",
        {"evaluationFunction": "Chi"},
    ),
}


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_estimator(
    name: str,
    base_learner: Literal["lightgbm", "xgboost"] = "lightgbm",
    seed: int = 42,
    **kwargs: Any,
) -> UpliftEstimator:
    """Instantiate an estimator by name.

    Parameters
    ----------
    name         : model key (see list_models())
    base_learner : 'lightgbm' or 'xgboost' — shared across all meta-learners
    seed         : random seed
    **kwargs     : additional constructor kwargs (e.g. hyperparameters)
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {list_models()}")

    module_path, cls_name, extra = _REGISTRY[name]
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)

    # Forest / tree models don't take base_learner
    import inspect

    sig = inspect.signature(cls.__init__)
    merged = {**extra, **kwargs}
    if "base_learner" in sig.parameters:
        merged["base_learner"] = base_learner
    if "seed" in sig.parameters:
        merged["seed"] = seed

    return cls(**merged)

"""Shared base learner factory and hyperparameter search space.

All meta-learners use the same base learner (LightGBM or XGBoost) with an
identical search space so that performance differences reflect the meta-strategy,
not tuning luck.
"""

from __future__ import annotations

from typing import Any, Literal

BASE_LEARNER_NAMES = ("lightgbm", "xgboost")


def make_base_learner(
    name: Literal["lightgbm", "xgboost", "ridge", "stump"] = "lightgbm",
    task: Literal["regression", "classification"] = "regression",
    seed: int = 42,
    **overrides: Any,
):
    """Return a fresh sklearn-compatible estimator.

    Parameters
    ----------
    name     : 'lightgbm' or 'xgboost'
    task     : 'regression' or 'classification'
    seed     : random seed
    overrides: extra kwargs forwarded to the constructor (used during tuning)
    """
    if name == "lightgbm":
        if task == "regression":
            from lightgbm import LGBMRegressor

            defaults = dict(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                random_state=seed,
                verbosity=-1,
                n_jobs=1,
            )
            defaults.update(overrides)
            return LGBMRegressor(**defaults)
        else:
            from lightgbm import LGBMClassifier

            defaults = dict(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                random_state=seed,
                verbosity=-1,
                n_jobs=1,
            )
            defaults.update(overrides)
            return LGBMClassifier(**defaults)

    elif name == "ridge":
        # Deliberately MISSPECIFIED baseline for the outcome-model-degradation
        # experiment: a linear model cannot represent nonlinear response
        # surfaces or interactions, stressing DM/DR's model dependence.
        if task == "regression":
            from sklearn.linear_model import Ridge

            return Ridge(alpha=1.0, random_state=seed)
        else:
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(C=1.0, random_state=seed, max_iter=200)

    elif name == "stump":
        # Deliberately WEAK learner (depth-2 trees, few rounds): predictive but
        # crude — the middle rung of the outcome-model-quality ladder.
        if task == "regression":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=20,
                max_depth=2,
                num_leaves=3,
                learning_rate=0.1,
                random_state=seed,
                verbosity=-1,
                n_jobs=1,
                **overrides,
            )
        else:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=20,
                max_depth=2,
                num_leaves=3,
                learning_rate=0.1,
                random_state=seed,
                verbosity=-1,
                n_jobs=1,
                **overrides,
            )

    elif name == "xgboost":
        if task == "regression":
            from xgboost import XGBRegressor

            defaults = dict(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                random_state=seed,
                verbosity=0,
                n_jobs=1,
            )
            defaults.update(overrides)
            return XGBRegressor(**defaults)
        else:
            from xgboost import XGBClassifier

            defaults = dict(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                random_state=seed,
                verbosity=0,
                n_jobs=1,
            )
            defaults.update(overrides)
            return XGBClassifier(**defaults)
    else:
        raise ValueError(f"Unknown base learner {name!r}. Choose from {BASE_LEARNER_NAMES}.")


# ---------------------------------------------------------------------------
# Shared hyperparameter search space (used by WP4 nested CV tuner)
# ---------------------------------------------------------------------------

SHARED_HPARAM_SPACE: dict[str, dict] = {
    "lightgbm": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_samples": [10, 20, 50],
    },
    "xgboost": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [3, 5, 10],
    },
}

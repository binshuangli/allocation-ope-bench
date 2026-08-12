"""Abstract base class for all uplift / CATE estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class UpliftEstimator(ABC):
    """Common interface for every uplift / CATE model in the benchmark.

    Subclasses must implement ``fit`` and ``predict_uplift``.
    The ``name`` class attribute is used as the model key in results tables.
    """

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
        propensity: Optional[pd.Series] = None,
    ) -> "UpliftEstimator":
        """Fit the estimator.

        Parameters
        ----------
        X          : feature matrix (n, p)
        treatment  : binary treatment indicator (n,)
        outcome    : observed outcome (n,)
        propensity : P(T=1|X) — used by DR/R-learners; ignored by others.
                     Pass None for RCT datasets (propensity estimated internally).
        """

    @abstractmethod
    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted CATE / uplift for each row of X, shape (n,)."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
        propensity: Optional[pd.Series],
    ):
        X_arr = X.values.astype(float)
        t_arr = treatment.values.astype(float)
        y_arr = outcome.values.astype(float)
        p_arr = propensity.values.astype(float) if propensity is not None else None
        return X_arr, t_arr, y_arr, p_arr

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

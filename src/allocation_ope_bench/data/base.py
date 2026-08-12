"""Common dataset interface for allocation-OPE benchmarking.

Every loader returns a :class:`Dataset` exposing a uniform schema:

    X            features (n, p)
    treatment    binary treatment indicator (n,)
    outcome      observed outcome (n,)
    cost         per-unit treatment cost (n,)  — defines the budget constraint
    propensity   known P(T=1 | X) for RCTs, else None
    has_ground_truth_effect
                 True iff individual potential-outcome means (mu0, mu1) are
                 known (synthetic / IHDP) → exact constrained-allocation value.
    mu0, mu1     true E[Y(0)|X], E[Y(1)|X]; present iff has_ground_truth_effect.

The split between *training* (fit the uplift/score model) and *evaluation*
(compute ground-truth value on a held-out randomized split) is produced by
:func:`train_eval_split`, which guarantees disjoint indices (leakage guard).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    cost: np.ndarray
    propensity: Optional[np.ndarray]
    has_ground_truth_effect: bool
    mu0: Optional[np.ndarray] = None
    mu1: Optional[np.ndarray] = None
    feature_names: Optional[list[str]] = field(default=None)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {self.X.shape}")
        n = self.X.shape[0]

        self.treatment = np.asarray(self.treatment).astype(int)
        self.outcome = np.asarray(self.outcome, dtype=float)
        self.cost = np.asarray(self.cost, dtype=float)

        for attr in ("treatment", "outcome", "cost"):
            arr = getattr(self, attr)
            if arr.shape != (n,):
                raise ValueError(f"{attr} must have shape ({n},), got {arr.shape}")

        uniq = set(np.unique(self.treatment).tolist())
        if not uniq.issubset({0, 1}):
            raise ValueError(f"treatment must be binary 0/1, found values {uniq}")

        if np.any(self.cost <= 0):
            raise ValueError("cost must be strictly positive (defines budget shares)")

        if self.propensity is not None:
            self.propensity = np.asarray(self.propensity, dtype=float)
            if self.propensity.shape != (n,):
                raise ValueError(f"propensity must have shape ({n},)")
            if np.any(self.propensity <= 0) or np.any(self.propensity >= 1):
                raise ValueError("propensity must lie strictly in (0, 1)")

        if self.has_ground_truth_effect:
            if self.mu0 is None or self.mu1 is None:
                raise ValueError(
                    "has_ground_truth_effect=True requires mu0 and mu1 to be provided"
                )
            self.mu0 = np.asarray(self.mu0, dtype=float)
            self.mu1 = np.asarray(self.mu1, dtype=float)
            for attr in ("mu0", "mu1"):
                if getattr(self, attr).shape != (n,):
                    raise ValueError(f"{attr} must have shape ({n},)")
        else:
            # Ground-truth value will come from a held-out randomized split, so a
            # known propensity is required for the IPS oracle.
            if self.propensity is None:
                raise ValueError(
                    "Datasets without ground-truth effects need a known propensity "
                    "(the IPS oracle estimates value on the randomized holdout)."
                )

        if self.feature_names is None:
            self.feature_names = [f"x{i}" for i in range(self.X.shape[1])]

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def subset(self, idx: np.ndarray) -> "Dataset":
        """Return a new Dataset restricted to the given row indices."""
        idx = np.asarray(idx)
        return Dataset(
            name=self.name,
            X=self.X[idx],
            treatment=self.treatment[idx],
            outcome=self.outcome[idx],
            cost=self.cost[idx],
            propensity=None if self.propensity is None else self.propensity[idx],
            has_ground_truth_effect=self.has_ground_truth_effect,
            mu0=None if self.mu0 is None else self.mu0[idx],
            mu1=None if self.mu1 is None else self.mu1[idx],
            feature_names=self.feature_names,
        )


def train_eval_split(
    dataset: Dataset,
    eval_frac: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices into disjoint (train, eval) sets.

    The *eval* split plays the role of the held-out randomized sample on which
    ground-truth allocation value is computed; the *train* split is where the
    uplift/score model is fit. Disjointness is the core leakage guard.
    """
    if not 0.0 < eval_frac < 1.0:
        raise ValueError("eval_frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    n = dataset.n
    perm = rng.permutation(n)
    n_eval = int(round(eval_frac * n))
    eval_idx = np.sort(perm[:n_eval])
    train_idx = np.sort(perm[n_eval:])
    return train_idx, eval_idx


def assert_no_feature_leakage(dataset: Dataset) -> None:
    """Raise if treatment or outcome leaked into the feature matrix.

    Checks that no feature column is identical (or near-identical) to the
    treatment or outcome vector.
    """
    for j in range(dataset.n_features):
        col = dataset.X[:, j]
        if np.array_equal(col, dataset.treatment.astype(float)):
            raise AssertionError(
                f"Feature {dataset.feature_names[j]!r} is identical to treatment (leakage)."
            )
        if np.array_equal(col, dataset.outcome):
            raise AssertionError(
                f"Feature {dataset.feature_names[j]!r} is identical to outcome (leakage)."
            )
        # Guard against a trivially-rescaled outcome leak.
        if np.std(col) > 0 and np.std(dataset.outcome) > 0:
            corr = np.corrcoef(col, dataset.outcome)[0, 1]
            if np.abs(corr) > 0.999:
                raise AssertionError(
                    f"Feature {dataset.feature_names[j]!r} is ~perfectly correlated "
                    f"with outcome (corr={corr:.4f}) — likely leakage."
                )

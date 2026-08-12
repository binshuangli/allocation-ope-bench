"""Synthetic generator for controlled budget/overlap stress tests.

Replaces OBP's ``SyntheticBanditDataset`` (which pins Python <3.11 and will not
install on the project's py312 toolchain) with a self-contained generator that
provides exactly what the benchmark needs: known potential-outcome means
(mu0, mu1) for an *exact* ground-truth allocation value, a per-unit cost, a
tunable budget, and a controllable logging-overlap knob.

Treatment can be assigned either fully at random (RCT, constant propensity) or
by a logistic logging policy whose sharpness controls overlap — the latter is
what later WPs use to stress common-support violation. Either way the *true*
propensity is recorded, so the IPS oracle and the exact-effect oracle agree.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from allocation_ope_bench.data.base import Dataset


def make_synthetic(
    n: int = 5000,
    n_features: int = 10,
    *,
    seed: int = 42,
    effect_scale: float = 1.0,
    noise_scale: float = 1.0,
    logging: str = "rct",
    logging_temperature: float = 1.0,
    rct_propensity: float = 0.5,
    cost: str = "unit",
    name: str = "synthetic",
) -> Dataset:
    """Generate a synthetic allocation dataset with known ground truth.

    Parameters
    ----------
    n, n_features      : sample size and feature dimension.
    effect_scale       : multiplies the heterogeneous treatment effect tau(X).
    noise_scale        : std of the outcome noise.
    logging            : 'rct' (constant propensity) or 'logistic' (overlap knob).
    logging_temperature: for logging='logistic', larger => sharper (worse overlap).
    rct_propensity     : constant P(T=1) when logging='rct'.
    cost               : 'unit' (all costs = 1) or 'heterogeneous' (positive, random).

    Returns
    -------
    Dataset with has_ground_truth_effect=True (mu0, mu1 known) and a known
    propensity vector.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))

    # Baseline response surface mu0(X) and a heterogeneous, sign-varying effect.
    beta = rng.normal(size=n_features) / np.sqrt(n_features)
    mu0 = X @ beta + 0.5 * X[:, 1] ** 2
    tau = effect_scale * (X[:, 0] + np.sin(2.0 * X[:, 2]))
    mu1 = mu0 + tau

    # Treatment assignment + recorded (true) propensity.
    if logging == "rct":
        if not 0.0 < rct_propensity < 1.0:
            raise ValueError("rct_propensity must be in (0, 1)")
        propensity = np.full(n, float(rct_propensity))
    elif logging == "logistic":
        # Overlap shrinks as temperature grows; clipped away from {0,1}.
        logits = logging_temperature * (X[:, 0] - 0.5 * X[:, 3])
        propensity = 1.0 / (1.0 + np.exp(-logits))
        propensity = np.clip(propensity, 0.02, 0.98)
    else:
        raise ValueError("logging must be 'rct' or 'logistic'")

    treatment = rng.binomial(1, propensity)
    noise = rng.normal(scale=noise_scale, size=n)
    outcome = np.where(treatment == 1, mu1, mu0) + noise

    if cost == "unit":
        cost_vec: Optional[np.ndarray] = np.ones(n)
    elif cost == "heterogeneous":
        cost_vec = rng.uniform(0.5, 1.5, size=n)
    else:
        raise ValueError("cost must be 'unit' or 'heterogeneous'")

    return Dataset(
        name=name,
        X=X,
        treatment=treatment,
        outcome=outcome,
        cost=cost_vec,
        propensity=propensity,
        has_ground_truth_effect=True,
        mu0=mu0,
        mu1=mu1,
    )

"""Six known-effect DGP settings on the real IHDP covariates (ACIC-2017 style).

The ACIC 2017 data-analysis challenge (Hahn, Dorie & Murray, 2019) established
"real covariates + simulated response surfaces" as the standard construction for
credible known-effect ground truth: covariates keep their real joint distribution
(correlations, skew, discreteness) while the potential-outcome means are known
exactly. We follow that design on the IHDP covariate matrix (25 covariates from
the Infant Health and Development Program, via the standard Shalit et al.
replication file): six DGP settings crossing three response-surface families with
two noise levels.

    setting  surface     noise
    1        linear      low          4   nonlinear   high
    2        linear      high         5   step        low
    3        nonlinear   low          6   step        high

Design rules
------------
* Surface coefficients are drawn from an rng seeded by the *setting only*, so
  mu0/mu1 (and hence the ground-truth allocation value) are identical across
  seeds within a setting; the per-run ``seed`` controls only the factual
  treatment draw and outcome noise. This mirrors the competition design, where
  each DGP setting fixes a truth and replications redraw the observables.
* Factual treatment is confounded (logistic in two covariates, clipped to
  [0.15, 0.85]) as in the ACIC settings, and ``propensity=None`` is recorded:
  the benchmark treats these datasets exactly like IHDP, synthesizing logged
  feedback from the known surfaces rather than from an RCT propensity.
* Noise is calibrated to the surface scale: sigma = noise_mult * std(tau), with
  noise_mult 0.5 (low) / 2.0 (high), so "high noise" genuinely stresses the
  outcome model without drowning every setting equally.
* mu0 carries a positive intercept (+2.0, ~2 sd of the centered surface) so the
  gross-outcome estimand is bounded away from zero at every budget — like the
  real datasets' outcome levels. A centered surface would put V(pi) near 0 in
  some cells and blow up every relative-error metric (the small-denominator
  artifact); the intercept removes the artifact without changing the effects.
"""

from __future__ import annotations

import numpy as np

from allocation_ope_bench.data.base import Dataset

N_SETTINGS = 6

# (surface_family, noise_mult) per setting, 1-indexed.
_SETTINGS: dict[int, tuple[str, float]] = {
    1: ("linear", 0.5),
    2: ("linear", 2.0),
    3: ("nonlinear", 0.5),
    4: ("nonlinear", 2.0),
    5: ("step", 0.5),
    6: ("step", 2.0),
}


def _zscore(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    return (X - mean) / std


def _real_covariates(source: str, n_rows: int | None) -> np.ndarray:
    """Real covariate matrix from a public dataset, z-scored.

    ``source='ihdp'``: the IHDP covariates (n=672, p=25) — small but canonical.
    ``source='hillstrom'``: the Hillstrom marketing-RCT covariates (n=64k,
    p=8) — a larger, independent real covariate set; ``n_rows`` caps it by a
    FIXED subsample (rng seeded by the source name only) so the covariate
    population, and hence each setting's truth, is identical across runs/seeds.
    """
    if source == "ihdp":
        from allocation_ope_bench.data.loaders import load_ihdp

        X = load_ihdp(replication=0).X.copy()
    elif source == "hillstrom":
        from allocation_ope_bench.data.loaders import load_hillstrom

        X = load_hillstrom().X.copy()
    else:
        raise ValueError(f"Unknown covariate source {source!r}")
    if n_rows is not None and X.shape[0] > n_rows:
        import zlib

        rng = np.random.default_rng(zlib.crc32(source.encode()))
        idx = np.sort(rng.choice(X.shape[0], size=int(n_rows), replace=False))
        X = X[idx]
    return _zscore(X)


def _surfaces(Z: np.ndarray, family: str, rng: np.random.Generator):
    """Known (mu0, tau) for one surface family on standardized covariates Z."""
    n, p = Z.shape
    if family == "linear":
        b0 = rng.normal(size=p) / np.sqrt(p)
        bt = np.zeros(p)
        bt[rng.choice(p, size=5, replace=False)] = rng.normal(scale=0.5, size=5)
        mu0 = Z @ b0
        tau = 0.3 + Z @ bt
    elif family == "nonlinear":
        b0 = rng.normal(size=p) / np.sqrt(p)
        j = rng.choice(p, size=4, replace=False)
        mu0 = np.sin(Z @ b0 * 2.0) + 0.5 * Z[:, j[0]] * Z[:, j[1]]
        tau = 0.3 + 0.5 * np.exp(-(Z[:, j[2]] ** 2)) + 0.3 * Z[:, j[3]] ** 2
    elif family == "step":
        b0 = rng.normal(size=p) / np.sqrt(p)
        j = rng.choice(p, size=3, replace=False)
        mu0 = Z @ b0
        # Sparse subgroup effects: large benefit in one cell, harm in another.
        tau = np.full(n, 0.1)
        tau = tau + 0.8 * ((Z[:, j[0]] > 0) & (Z[:, j[1]] > 0))
        tau = tau - 0.5 * ((Z[:, j[0]] <= 0) & (Z[:, j[2]] > 0.5))
    else:  # pragma: no cover - guarded by _SETTINGS
        raise ValueError(f"Unknown surface family {family!r}")
    return mu0, tau


def make_acic_ihdp(
    setting: int = 1,
    seed: int = 42,
    covariates: str = "ihdp",
    n_rows: int | None = None,
) -> Dataset:
    """One ACIC-2017-style known-effect dataset on real covariates.

    ``setting`` in 1..6 picks the (surface family, noise level) pair; ``seed``
    redraws only the factual treatment and outcome noise (the truth is fixed
    per setting). ``covariates`` chooses the real covariate source ('ihdp' or
    'hillstrom'); ``n_rows`` caps large sources by a fixed subsample.
    """
    if setting not in _SETTINGS:
        raise ValueError(f"setting must be in 1..{N_SETTINGS}, got {setting}")
    family, noise_mult = _SETTINGS[setting]

    Z = _real_covariates(covariates, n_rows)
    n = Z.shape[0]

    rng_surface = np.random.default_rng(1000 + setting)  # truth: setting only
    mu0, tau = _surfaces(Z, family, rng_surface)
    mu0 = mu0 + 2.0  # positive outcome level: keeps V(pi) off zero (see docstring)
    mu1 = mu0 + tau
    sigma = float(noise_mult * max(tau.std(), 1e-6))

    rng = np.random.default_rng(seed)  # observables: per-run seed
    logits = 0.6 * Z[:, 0] - 0.6 * Z[:, 2]
    e = np.clip(1.0 / (1.0 + np.exp(-logits)), 0.15, 0.85)
    treatment = rng.binomial(1, e)
    outcome = np.where(treatment == 1, mu1, mu0) + rng.normal(scale=sigma, size=n)

    prefix = "acic" if covariates == "ihdp" else f"acic_{covariates[:2]}"
    return Dataset(
        name=f"{prefix}_s{setting}",
        X=Z,
        treatment=treatment,
        outcome=outcome,
        cost=np.ones(n),
        propensity=None,  # IHDP-style: logged feedback comes from the surfaces
        has_ground_truth_effect=True,
        mu0=mu0,
        mu1=mu1,
    )

"""Loaders for the real and semi-synthetic benchmark datasets.

Marketing RCTs (Criteo, Hillstrom, Lenta, X5) come via scikit-uplift; their
treatment is randomized with a constant marginal probability, which we record
as the known propensity for the IPS oracle. They carry no per-unit cost, so we
assign unit cost (budget_k == fraction treated) — heterogeneous-cost stress is
the synthetic generator's job.

Semi-synthetic:
* IHDP — simulated response surfaces give exact (mu0, mu1) → ground-truth effect.
* Jobs (LaLonde/NSW) — NO individual counterfactuals; we keep only the
  randomized subset and evaluate via the known-propensity oracle
  (has_ground_truth_effect=False), per the WP1 estimand decision.

All network-dependent work is lazy so importing this module is cheap and
offline-safe; the loaders themselves are exercised by network-marked tests.
"""

from __future__ import annotations

import numpy as np

from allocation_ope_bench.data.base import Dataset
from allocation_ope_bench.data.download import cached_download

# Canonical CATE-benchmark replication files (Johansson/Shalit).
_IHDP_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
_JOBS_URL = "https://www.fredjo.com/files/jobs_DW_bin.new.10.train.npz"
# Twin births, as distributed with GANITE (Yoon et al., ICLR 2018).
_TWINS_URL = (
    "https://raw.githubusercontent.com/jsyoon0823/GANITE/master/data/Twin_data.csv.gz"
)


def _constant_propensity(treatment: np.ndarray) -> np.ndarray:
    """Known constant assignment rate of a marketing RCT (clipped off 0/1)."""
    p = float(np.clip(treatment.mean(), 1e-3, 1 - 1e-3))
    return np.full(treatment.shape[0], p)


def _numeric_features(df) -> tuple[np.ndarray, list[str]]:
    """Numeric feature matrix with median imputation of missing values.

    Some marketing RCTs (e.g. Lenta) carry NaNs in numeric columns; the sklearn
    meta-learner wrappers reject NaN, so we median-impute (a NaN-only column
    falls back to 0). Returns (X, feature_names)."""
    num = df.select_dtypes(include="number")
    X = np.asarray(num.values, dtype=float)
    if np.isnan(X).any():
        medians = np.nanmedian(X, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)  # all-NaN columns -> 0
        inds = np.where(np.isnan(X))
        X[inds] = np.take(medians, inds[1])
    return X, list(num.columns)


# ── scikit-uplift marketing RCTs ──────────────────────────────────────────────


def _from_sklift(name, fetch_fn, treat_label, control_label=None, **fetch_kwargs):
    bunch = fetch_fn(return_X_y_t=False, **fetch_kwargs)
    X, feature_names = _numeric_features(bunch.data)
    treatment_raw = np.asarray(bunch.treatment)
    outcome = np.asarray(bunch.target, dtype=float)

    if control_label is not None:
        # Binary arms already; map labels to {0, 1}.
        treatment = (treatment_raw == treat_label).astype(int)
    else:
        # Multi-arm (e.g. Hillstrom): everything that is not "no treatment".
        treatment = (~np.isin(treatment_raw, treat_label)).astype(int)

    return Dataset(
        name=name,
        X=X,
        treatment=treatment,
        outcome=outcome,
        cost=np.ones(X.shape[0]),
        propensity=_constant_propensity(treatment),
        has_ground_truth_effect=False,
        feature_names=feature_names,
    )


def load_hillstrom(target_col: str = "visit") -> Dataset:
    from sklift.datasets import fetch_hillstrom

    # Arms: "Mens E-Mail", "Womens E-Mail", "No E-Mail"; treat = any e-mail.
    return _from_sklift(
        "hillstrom",
        fetch_hillstrom,
        treat_label=["No E-Mail"],  # the control label(s); treat = NOT in this
        control_label=None,
        target_col=target_col,
    )


def load_lenta() -> Dataset:
    from sklift.datasets import fetch_lenta

    return _from_sklift("lenta", fetch_lenta, treat_label="test", control_label="control")


def load_x5() -> Dataset:
    """X5 RetailHero uplift trial.

    ``fetch_x5().data`` is a nested bunch (clients / train cohort / 45M-row
    purchases). We build a compact client-level feature set by joining the
    ``clients`` table onto the experiment cohort — age, encoded gender, and
    redemption tenure — without touching the huge purchases table. (Richer
    purchase-aggregate features are left to a dedicated feature-engineering step.)
    """
    import pandas as pd
    from sklift.datasets import fetch_x5

    bunch = fetch_x5()
    clients = bunch.data["clients"].copy()
    train_ids = bunch.data["train"][["client_id"]]
    treat = (np.asarray(bunch.treatment) == "treatment").astype(int)
    target = np.asarray(bunch.target, dtype=float)
    cohort = train_ids.assign(_t=treat, _y=target).merge(clients, on="client_id", how="left")

    issue = pd.to_datetime(cohort["first_issue_date"], errors="coerce")
    redeem = pd.to_datetime(cohort["first_redeem_date"], errors="coerce")
    tenure_days = (redeem - issue).dt.total_seconds() / 86400.0
    gender_code = cohort["gender"].map({"F": 0.0, "M": 1.0}).fillna(2.0)

    feats = pd.DataFrame(
        {
            "age": pd.to_numeric(cohort["age"], errors="coerce"),
            "gender": gender_code,
            "tenure_days": tenure_days,
            "has_redeemed": redeem.notna().astype(float),
        }
    )
    X, feature_names = _numeric_features(feats)
    return Dataset(
        name="x5",
        X=X,
        treatment=cohort["_t"].to_numpy().astype(int),
        outcome=cohort["_y"].to_numpy(dtype=float),
        cost=np.ones(X.shape[0]),
        propensity=_constant_propensity(cohort["_t"].to_numpy()),
        has_ground_truth_effect=False,
        feature_names=feature_names,
    )


def load_criteo(target_col: str = "conversion") -> Dataset:
    from sklift.datasets import fetch_criteo

    bunch = fetch_criteo(target_col=target_col, treatment_col="treatment")
    X, feature_names = _numeric_features(bunch.data)
    treatment = np.asarray(bunch.treatment).astype(int)
    outcome = np.asarray(bunch.target, dtype=float)
    return Dataset(
        name="criteo",
        X=X,
        treatment=treatment,
        outcome=outcome,
        cost=np.ones(X.shape[0]),
        propensity=_constant_propensity(treatment),
        has_ground_truth_effect=False,
        feature_names=feature_names,
    )


# ── semi-synthetic ────────────────────────────────────────────────────────────


def load_twins(seed: int = 0) -> Dataset:
    """Twin births (~11.4k same-sex pairs under 2 kg) --- OBSERVED ground truth.

    Every other exact-value source in this benchmark (\\dataset{Synthetic},
    \\dataset{IHDP}, both ACIC-style suites) gets its ``mu0, mu1`` from a response
    surface *we specify or fit*, so an estimator is ultimately scored against a
    simulation. Here it is not. In a twin pair one sibling is lighter and one
    heavier and BOTH outcomes are recorded, so taking the pair as the unit,

        mu0 = survival of the lighter twin,  mu1 = survival of the heavier twin

    are data, not fitted quantities. Be precise about what that buys: the two
    twins are different individuals, so this is a matched-pair contrast rather
    than one infant observed under both conditions, and it proxies a causal
    effect of birth weight only under co-twin exchangeability given the shared
    covariates. What matters here is narrower and does hold --- the reference
    value is fixed by the data and not by any surface we chose, so scoring
    against it is not circular.

    Because mu is exactly 0/1 and we set the factual outcome to the realized
    potential outcome, the surface-sampling path's calibrated noise is
    sigma = 0: logged rewards are the actual recorded outcomes and the retained
    counterfactual is the actual co-twin outcome.

    Scale caveat for allocation: survival is 0.823 (lighter) against 0.839
    (heavier), so the ATE is +1.61pp and the ENTIRE achievable range of
    V(pi) across all budgeted policies is [0.823, 0.839]. Relative RMSE
    normalizes by |V| ~ 0.83, so it compresses every estimator toward zero on
    this dataset for a reason unrelated to estimator skill --- the gross-value
    metric's documented tendency to reward baseline predictability, in its most
    extreme form. Read the value-range-normalized figures alongside it.

    Encoding: the source codes the outcome as days-at-death in [0, 360] with
    9999 for "survived the first year"; we map to a survival indicator so that
    the treatment (being heavier) carries positive value.
    """
    import pandas as pd

    path = cached_download(_TWINS_URL, "twins.csv.gz")
    df = pd.read_csv(path)
    cols = list(df.columns)
    X = df[cols[:-2]].to_numpy(dtype=float)
    mu0 = (df[cols[-2]].to_numpy() == 9999).astype(float)
    mu1 = (df[cols[-1]].to_numpy() == 9999).astype(float)

    # A factual arm is needed only so the surface path can calibrate its reward
    # noise; assigning at random and reading off the corresponding REALIZED
    # potential outcome makes that residual exactly zero.
    rng = np.random.default_rng(seed)
    treatment = (rng.random(X.shape[0]) < 0.5).astype(int)
    outcome = np.where(treatment == 1, mu1, mu0)

    return Dataset(
        name="twins",
        X=X,
        treatment=treatment,
        outcome=outcome,
        cost=np.ones(X.shape[0]),
        propensity=None,  # surface path, as for IHDP -- mu0/mu1 are exact
        has_ground_truth_effect=True,
        mu0=mu0,
        mu1=mu1,
        feature_names=[c.strip("'’") for c in cols[:-2]],
    )


def load_ihdp(replication: int = 0) -> Dataset:
    """IHDP with simulated response surfaces → exact (mu0, mu1)."""
    path = cached_download(_IHDP_URL, "ihdp_npci_1-100.train.npz")
    arr = np.load(path)
    r = replication
    X = np.asarray(arr["x"][:, :, r], dtype=float)
    treatment = np.asarray(arr["t"][:, r]).astype(int)
    outcome = np.asarray(arr["yf"][:, r], dtype=float)
    mu0 = np.asarray(arr["mu0"][:, r], dtype=float)
    mu1 = np.asarray(arr["mu1"][:, r], dtype=float)
    return Dataset(
        name="ihdp",
        X=X,
        treatment=treatment,
        outcome=outcome,
        cost=np.ones(X.shape[0]),
        propensity=None,
        has_ground_truth_effect=True,
        mu0=mu0,
        mu1=mu1,
    )


def load_jobs() -> Dataset:
    """Jobs (LaLonde/NSW): randomized subset only, no individual effects."""
    path = cached_download(_JOBS_URL, "jobs_DW_bin.new.10.train.npz")
    arr = np.load(path)
    r = 0
    X_all = np.asarray(arr["x"][:, :, r], dtype=float)
    t_all = np.asarray(arr["t"][:, r]).astype(int)
    y_all = np.asarray(arr["yf"][:, r], dtype=float)
    e_all = np.asarray(arr["e"][:, r]).astype(int)  # 1 = randomized-trial member

    mask = e_all == 1
    treatment = t_all[mask]
    return Dataset(
        name="jobs",
        X=X_all[mask],
        treatment=treatment,
        outcome=y_all[mask],
        cost=np.ones(int(mask.sum())),
        propensity=_constant_propensity(treatment),
        has_ground_truth_effect=False,
    )

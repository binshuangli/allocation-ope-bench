"""WP2 — allocation policy variants + logged-data construction."""

import numpy as np
import pytest

from allocation_ope_bench.data import Dataset, make_synthetic, train_eval_split
from allocation_ope_bench.policies import AllocationPolicy, make_logged_data


def _fit_policy(variant, **kw):
    ds = make_synthetic(n=4000, seed=0)
    tr, ev = train_eval_split(ds, seed=0)
    pol = AllocationPolicy(variant=variant, seed=0, **kw).fit(ds.subset(tr))
    return pol, ds.subset(ev)


def test_deterministic_prob_is_binary_and_respects_budget():
    pol, ev = _fit_policy("deterministic")
    p = pol.action_prob(ev.X, budget_k=0.3, feature_names=ev.feature_names)
    assert set(np.unique(p).tolist()).issubset({0.0, 1.0})
    assert abs(p.mean() - 0.3) < 0.02  # unit cost => fraction treated ~= budget


def test_softmax_prob_in_unit_interval_and_expected_spend():
    pol, ev = _fit_policy("softmax", temperature=1.0)
    p = pol.action_prob(ev.X, budget_k=0.3, feature_names=ev.feature_names)
    assert np.all((p >= 0) & (p <= 1))
    assert 0 < p.min() < p.max() < 1  # smoothed => genuine overlap
    assert abs(p.mean() - 0.3) < 0.03  # expected spend calibrated to budget


def test_epsilon_prob_mixes_hard_rule():
    pol, ev = _fit_policy("epsilon", epsilon=0.2)
    p = pol.action_prob(ev.X, budget_k=0.3, feature_names=ev.feature_names)
    # values are either 0.2*0.3 (withheld) or 0.8 + 0.2*0.3 (treated)
    assert np.all(p >= 0.2 * 0.3 - 1e-9)
    assert np.all(p <= 0.8 + 0.2 * 0.3 + 1e-9)


def test_logging_temperature_zero_keeps_rct():
    ds = make_synthetic(n=3000, seed=1)
    logged = make_logged_data(ds, np.zeros(ds.n), temperature=0.0, seed=1)
    assert logged.n == ds.n  # accept-all when logging == RCT
    assert np.allclose(logged.pscore, np.where(ds.treatment == 1, ds.propensity, 1 - ds.propensity))


def test_logging_temperature_controls_overlap():
    ds = make_synthetic(n=5000, seed=2)
    score = ds.mu1 - ds.mu0
    # Convention: HIGH temperature => smoother logging => good overlap; LOW =>
    # sharper => poor overlap. Sharper logging rejects more units (lower ESS),
    # so the retained sample shrinks — the WP4 overlap stress knob.
    smooth = make_logged_data(ds, score, temperature=5.0, seed=2)
    sharp = make_logged_data(ds, score, temperature=0.3, seed=2)
    assert sharp.n < smooth.n


# ── Semi-synthetic surface sampling (IHDP-style: known mu0/mu1, no propensity) ──


def _strip_propensity(ds: Dataset) -> Dataset:
    """Mimic IHDP: known potential-outcome means but no RCT propensity."""
    return Dataset(
        name="surface_only",
        X=ds.X,
        treatment=ds.treatment,
        outcome=ds.outcome,
        cost=ds.cost,
        propensity=None,
        has_ground_truth_effect=True,
        mu0=ds.mu0,
        mu1=ds.mu1,
    )


def test_surface_logging_used_when_no_propensity():
    ds = _strip_propensity(make_synthetic(n=3000, seed=4))
    score = ds.mu1 - ds.mu0
    logged = make_logged_data(ds, score, temperature=2.0, seed=4)
    # Surface path keeps ALL units (no rejection sampling).
    assert logged.n == ds.n
    # pscore lies strictly in (0, 1].
    assert logged.pscore.min() > 0 and logged.pscore.max() <= 1.0
    # logging_prob_treat populated for diagnostics.
    assert logged.logging_prob_treat is not None


def test_surface_logging_reward_matches_surface_scale():
    ds = _strip_propensity(make_synthetic(n=8000, seed=5, noise_scale=1.0))
    score = ds.mu1 - ds.mu0
    logged = make_logged_data(ds, score, temperature=2.0, seed=5)
    # Mean logged reward should track the average response surface under the
    # sampled actions (within noise) — i.e. consistent with the exact oracle.
    expected = np.where(logged.action == 1, ds.mu1, ds.mu0)
    # Reward = surface + zero-mean noise, so mean(reward - expected) ~ 0.
    assert abs(float(np.mean(logged.reward - expected))) < 0.15


def test_no_propensity_no_effect_rejected_at_construction():
    # The Dataset invariant (base.py) already forbids the only case where
    # make_logged_data could not build logged feedback: no propensity AND no
    # ground-truth effect. So such a dataset can never even be constructed.
    ds = make_synthetic(n=500, seed=6)
    with pytest.raises(ValueError, match="propensity"):
        Dataset(
            name="bad",
            X=ds.X,
            treatment=ds.treatment,
            outcome=ds.outcome,
            cost=ds.cost,
            propensity=None,
            has_ground_truth_effect=False,
        )


@pytest.mark.parametrize("uplift_model", ["random", "t_learner", "s_learner"])
def test_policy_score_is_a_function_of_x(uplift_model):
    """Scoring must be deterministic in x, not a fresh draw per call.

    ``AllocationPolicy.action_prob`` calls ``score`` internally, and the runner
    calls ``score`` again for the logger. A candidate whose score changed between
    calls would be logged under one policy, targeted as another, and scored
    against the truth of a third. The random baseline once did exactly that.
    """
    ds = make_synthetic(n=4000, seed=0)
    tr, ev = train_eval_split(ds, seed=0)
    pol = AllocationPolicy(uplift_model=uplift_model, variant="deterministic", seed=0).fit(
        ds.subset(tr)
    )
    ev_ds = ds.subset(ev)
    s1 = pol.score(ev_ds.X, feature_names=ev_ds.feature_names)
    pol.action_prob(ev_ds.X, budget_k=0.3, feature_names=ev_ds.feature_names)  # advances any rng
    s2 = pol.score(ev_ds.X, feature_names=ev_ds.feature_names)
    np.testing.assert_array_equal(s1, s2)

    # ... and row-wise, so a subset of the contexts scores identically (the logged
    # sample is a subset of the evaluation split).
    sub = np.arange(0, ev_ds.n, 3)
    s_sub = pol.score(ev_ds.X[sub], feature_names=ev_ds.feature_names)
    np.testing.assert_allclose(s_sub, s1[sub], rtol=1e-12, atol=1e-12)


def test_random_baseline_score_is_uniform_and_uninformative():
    ds = make_synthetic(n=4000, seed=0)
    tr, ev = train_eval_split(ds, seed=0)
    pol = AllocationPolicy(uplift_model="random", variant="deterministic", seed=0).fit(
        ds.subset(tr)
    )
    ev_ds = ds.subset(ev)
    s = pol.score(ev_ds.X, feature_names=ev_ds.feature_names)
    assert 0.0 <= s.min() and s.max() < 1.0
    assert abs(s.mean() - 0.5) < 0.02
    # uninformative about the covariates it is hashed from
    assert abs(np.corrcoef(s, ev_ds.X[:, 0])[0, 1]) < 0.06

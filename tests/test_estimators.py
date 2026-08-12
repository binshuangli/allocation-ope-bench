"""WP2 — all six estimators through one interface; CIs; RCT sanity vs ground truth."""

import numpy as np
import pytest

from allocation_ope_bench.data import make_synthetic, train_eval_split, true_allocation_value
from allocation_ope_bench.estimators import (
    OutcomeModel,
    get_ope_estimator,
    list_estimators,
    needs_policy_kwargs,
)
from allocation_ope_bench.policies import AllocationPolicy, make_logged_data


def _setup(budget_k=0.3, n=8000, seed=0):
    ds = make_synthetic(n=n, seed=seed, logging="rct", rct_propensity=0.5)
    tr, ev = train_eval_split(ds, seed=seed)
    eval_ds = ds.subset(ev)
    policy = AllocationPolicy(variant="deterministic", seed=seed).fit(ds.subset(tr))
    # temperature=0 => logged is the full eval split (good overlap) for the sanity check.
    logged = make_logged_data(
        eval_ds,
        policy.score(eval_ds.X, feature_names=eval_ds.feature_names),
        temperature=0.0,
        seed=seed,
    )
    target_prob = policy.action_prob(logged.context, budget_k, feature_names=eval_ds.feature_names)
    return eval_ds, policy, logged, target_prob, budget_k


def test_registry_has_six_standard_estimators():
    # The six standard estimators run through the (logged, target_prob) interface.
    assert {"dm", "ips", "snips", "dr", "switch_dr", "bips"}.issubset(set(list_estimators()))


def test_all_estimators_run_through_one_interface_with_ci():
    eval_ds, policy, logged, target_prob, budget_k = _setup()
    scores = policy.score(logged.context, feature_names=eval_ds.feature_names)
    shared_model = OutcomeModel(seed=0).fit(logged)  # shared base model
    for name in list_estimators():
        est = get_ope_estimator(name)
        # Policy-dependent estimators (Guo) need the scores + budget to re-solve.
        extra = {"scores": scores, "budget_k": budget_k} if needs_policy_kwargs(name) else {}
        res = est.estimate(
            logged, target_prob, outcome_model=shared_model, n_bootstrap=200, seed=0, **extra
        )
        assert res.estimator == name
        assert np.isfinite(res.value)
        assert np.isfinite(res.ci_low) and np.isfinite(res.ci_high)
        assert res.ci_low <= res.value <= res.ci_high


def test_estimators_recover_ground_truth_on_clean_rct():
    """With good overlap (RCT logging), IPS/SNIPS/DR should sit near the truth."""
    eval_ds, policy, logged, target_prob, budget_k = _setup(n=12000)
    truth = true_allocation_value(
        eval_ds, policy.score(eval_ds.X, feature_names=eval_ds.feature_names), budget_k
    )
    shared_model = OutcomeModel(seed=0).fit(logged)
    for name in ("ips", "snips", "dr"):
        res = get_ope_estimator(name).estimate(
            logged, target_prob, outcome_model=shared_model, seed=0
        )
        assert res.value == pytest.approx(truth, abs=0.2), f"{name}: {res.value} vs {truth}"


def test_dm_silently_confident_under_tight_budget_poor_overlap():
    """Documents the headline failure: under sharp logging + tight budget, IPS
    loses effective sample size while DM still returns a (possibly wrong) number."""
    ds = make_synthetic(n=8000, seed=3, logging="rct", rct_propensity=0.5)
    tr, ev = train_eval_split(ds, seed=3)
    eval_ds = ds.subset(ev)
    policy = AllocationPolicy(variant="deterministic", seed=3).fit(ds.subset(tr))
    score = policy.score(eval_ds.X, feature_names=eval_ds.feature_names)
    logged = make_logged_data(eval_ds, score, temperature=0.2, seed=3)  # low temp => poor overlap
    target_prob = policy.action_prob(logged.context, 0.1, feature_names=eval_ds.feature_names)

    ips = get_ope_estimator("ips").estimate(logged, target_prob, seed=3)
    dm = get_ope_estimator("dm").estimate(logged, target_prob, seed=3)
    # Both return finite numbers; the benchmark's job (WP3+) is to expose which
    # one is trustworthy via diagnostics — here we just assert they run.
    assert np.isfinite(ips.value) and np.isfinite(dm.value)


def test_shrinkage_dr_recovers_plain_dr_at_infinite_lambda():
    """lam -> inf must reproduce plain DR bit-for-bit (w_lam -> w), and lam = 0 must
    reproduce DM (correction removed)."""
    from allocation_ope_bench.estimators.doubly_robust import DoublyRobust
    from allocation_ope_bench.estimators.direct import DirectMethod
    from allocation_ope_bench.estimators.shrinkage import ShrinkageDR

    from allocation_ope_bench.estimators.base import OutcomeModel

    _, _, logged, tprob, _ = _setup()
    om = OutcomeModel(seed=0).fit(logged)
    dr = DoublyRobust().estimate(logged, tprob, outcome_model=om, n_bootstrap=10, seed=0)
    inf = ShrinkageDR(lam=float("inf")).estimate(logged, tprob, outcome_model=om, n_bootstrap=10, seed=0)
    assert inf.value == dr.value
    dm = DirectMethod().estimate(logged, tprob, outcome_model=om, n_bootstrap=10, seed=0)
    zero = ShrinkageDR(lam=0.0).estimate(logged, tprob, outcome_model=om, n_bootstrap=10, seed=0)
    assert abs(zero.value - dm.value) < 1e-12
    auto = ShrinkageDR().estimate(logged, tprob, outcome_model=om, n_bootstrap=10, seed=0)
    assert np.isfinite(auto.value)


def test_out_of_fold_outcome_model_is_actually_out_of_fold():
    """The OOF nuisance must not see a unit's own outcome, and must refuse other rows.

    Guards the RQ1 honest-nuisance check: if `predict` silently fell back to the
    full-sample fit, the comparison it exists to make would be vacuous. In-sample
    factual error is optimistic by construction, so OOF error must be strictly larger.
    """
    from allocation_ope_bench.estimators.base import OutcomeModel, OutOfFoldOutcomeModel

    _, _, logged, _, _ = _setup()
    ins = OutcomeModel(seed=0).fit(logged)
    oof = OutOfFoldOutcomeModel(seed=0).fit(logged)

    def factual_rmse(model):
        mu = np.where(
            logged.action == 1,
            model.predict(logged.context, 1),
            model.predict(logged.context, 0),
        )
        return float(np.sqrt(((logged.reward - mu) ** 2).mean()))

    assert factual_rmse(oof) > factual_rmse(ins)
    # Predictions differ per row, i.e. folds really were held out.
    assert not np.allclose(oof.predict(logged.context, 1), ins.predict(logged.context, 1))
    with pytest.raises(ValueError):
        oof.predict(logged.context[:5], 1)

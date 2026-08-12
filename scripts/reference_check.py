"""Numerical cross-check of our estimators against Open Bandit Pipeline.

Every estimator in this benchmark is our own implementation behind a shared
interface -- the right call for shared nuisances, but a benchmark paper owes the
reader evidence that its IPS/SNIPS/DM/DR agree with a reference implementation
on identical inputs. This script builds one representative configuration
(synthetic, seed 42, T-learner target, tau=2.0, k=0.3), maps the logged data to
OBP's format (two actions, deterministic target as a one-hot action
distribution, the SAME fitted mu-hat fed to both sides), and asserts agreement.

IPS/SNIPS/DM are closed-form given the data, so agreement must be exact to
floating point. DR agrees exactly when both sides receive the same mu-hat.
Discrepancy beyond 1e-8 fails loudly.

Run: python scripts/reference_check.py        (or: make reference-check)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

TOL = 1e-8


def main() -> int:
    from omegaconf import OmegaConf

    from allocation_ope_bench.data import train_eval_split
    from allocation_ope_bench.estimators.base import OutcomeModel
    from allocation_ope_bench.estimators.direct import DirectMethod
    from allocation_ope_bench.estimators.doubly_robust import DoublyRobust
    from allocation_ope_bench.estimators.ips import IPS, SNIPS
    from allocation_ope_bench.experiments.runner import _build_dataset
    from allocation_ope_bench.policies import AllocationPolicy, make_logged_data

    from obp.ope import (
        DirectMethod as ObpDM,
        DoublyRobust as ObpDR,
        InverseProbabilityWeighting as ObpIPS,
        SelfNormalizedInverseProbabilityWeighting as ObpSNIPS,
    )

    seed, tau, bk = 42, 2.0, 0.3
    dataset = _build_dataset(OmegaConf.create({"name": "synthetic"}), seed, None)
    tr, ev = train_eval_split(dataset, eval_frac=0.5, seed=seed)
    train_ds, eval_ds = dataset.subset(tr), dataset.subset(ev)
    pol = AllocationPolicy(uplift_model="t_learner", variant="deterministic",
                           seed=seed).fit(train_ds)
    score = pol.score(eval_ds.X, feature_names=eval_ds.feature_names)
    logged = make_logged_data(eval_ds, score, temperature=tau, seed=seed)
    om = OutcomeModel("lightgbm", seed=seed).fit(logged)
    tprob = pol.action_prob(logged.context, bk, feature_names=eval_ds.feature_names)

    ours = {
        "ips": IPS().estimate(logged, tprob, outcome_model=om, n_bootstrap=2, seed=seed).value,
        "snips": SNIPS().estimate(logged, tprob, outcome_model=om, n_bootstrap=2, seed=seed).value,
        "dm": DirectMethod().estimate(logged, tprob, outcome_model=om, n_bootstrap=2, seed=seed).value,
        "dr": DoublyRobust().estimate(logged, tprob, outcome_model=om, n_bootstrap=2, seed=seed).value,
    }

    # ---- map to OBP's format: 2 actions (0=control, 1=treat), len_list=1 ----
    n = logged.n
    action = logged.action.astype(int)
    reward = logged.reward.astype(float)
    pscore = logged.pscore.astype(float)          # pi_b(a_i | x_i), matching OBP
    z = (tprob > 0.5).astype(float)               # deterministic top-k assignment
    action_dist = np.zeros((n, 2, 1))
    action_dist[:, 1, 0] = z
    action_dist[:, 0, 0] = 1.0 - z
    mu_hat = np.zeros((n, 2, 1))                  # the SAME fitted mu-hat
    mu_hat[:, 0, 0] = om.predict(logged.context, 0)
    mu_hat[:, 1, 0] = om.predict(logged.context, 1)

    kw = dict(reward=reward, action=action, action_dist=action_dist,
              pscore=pscore, estimated_rewards_by_reg_model=mu_hat,
              position=np.zeros(n, dtype=int))
    theirs = {
        "ips": ObpIPS().estimate_policy_value(**kw),
        "snips": ObpSNIPS().estimate_policy_value(**kw),
        "dm": ObpDM().estimate_policy_value(**kw),
        "dr": ObpDR().estimate_policy_value(**kw),
    }

    import obp
    print(f"reference: Open Bandit Pipeline {obp.__version__}"
          f"  |  config: synthetic seed={seed} t_learner tau={tau} k={bk} n={n}")
    bad = []
    for name in ("ips", "snips", "dm", "dr"):
        d = abs(ours[name] - theirs[name])
        flag = "ok  " if d < TOL else "FAIL"
        print(f"  {flag} {name:6s} ours {ours[name]:.12f}   obp {theirs[name]:.12f}"
              f"   |diff| {d:.2e}")
        if d >= TOL:
            bad.append(name)
    if bad:
        print(f"\nFAILED: {bad} disagree with the reference implementation.")
        return 1
    print("\nAll four estimators agree with OBP to <1e-8 on identical inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Are the tail-control conclusions facts about OPE, or artifacts of the 0.02 floor?

Every other experiment floors the logging propensity at ``eps = 0.02``, so importance
weights never exceed 50. Three of the benchmark's conclusions live downstream of that
choice --- clipping never helps, shrinkage-DR helps only marginally, and every documented
failure is a bounded-weight phenomenon --- and a reviewer can fairly ask whether they are
statements about allocation OPE or about the floor. Real logs routinely carry propensities
well below 0.02, and tail control exists precisely for that regime.

This experiment sweeps the floor itself, ``eps in {0.02, 0.005, 0.001, 0.0002}`` (weight
ceilings 50, 200, 1000, 5000) crossed with temperatures down to ``tau = 0.05`` -- both
are needed, since at the main sweep's temperatures the logistic's own tail binds before
``eps`` does -- on the exact-value datasets under the self-aligned and
candidate-independent loggers, and re-runs IPS, clipped IPS at three thresholds, DR and
shrinkage-DR at each. The question is not whether error grows as the floor falls --- it
must --- but whether the *ranking* changes: does clipping stop being useless once the tail
is genuinely heavy, and does the paper's "tail control does not substitute for overlap"
lesson survive outside its bounded-weight sandbox?

The target policy, the truth labels, the fitted candidates, the seeds and the splits are
identical across floors, so the floor is the only thing that moves.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from allocation_ope_bench.data import train_eval_split, true_allocation_value
from allocation_ope_bench.estimators.base import OutcomeModel
from allocation_ope_bench.estimators.doubly_robust import DoublyRobust
from allocation_ope_bench.estimators.ips import IPS, ClippedIPS
from allocation_ope_bench.estimators.shrinkage import ShrinkageDR
from allocation_ope_bench.experiments.git_utils import get_git_hash
from allocation_ope_bench.experiments.logger_alignment import (
    _assert_scores_are_functions_of_x,
    _independent_score,
)
from allocation_ope_bench.experiments.runner import _build_dataset
from allocation_ope_bench.metrics import compute_diagnostics

log = logging.getLogger(__name__)

FLOORS = (0.02, 0.005, 0.001, 0.0002)
REGIMES = ("self_aligned", "independent")
# The floor alone cannot create a heavy tail: at the main sweep's temperatures the
# logistic's own tail binds before eps does (weights saturate near 140 for eps <= 0.005).
# Sharp logging is what pushes pi_b onto the floor, so this experiment uses its own
# temperature grid extending well below the main sweep's to reach the regime where
# tail control is supposed to matter.
TEMPS = (0.05, 0.1, 0.5, 2.0)


def _estimators():
    return {
        "ips": IPS(),
        "clipped_ips_m5": ClippedIPS(5.0),
        "clipped_ips_m50": ClippedIPS(50.0),
        "clipped_ips_m500": ClippedIPS(500.0),
        "dr": DoublyRobust(),
        "shrinkage_dr": ShrinkageDR(),
    }


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from allocation_ope_bench.policies import AllocationPolicy, make_logged_data

    git_hash = get_git_hash()
    max_n = cfg.get("max_n", None)
    max_n = int(max_n) if max_n is not None else None
    cand_names = [c for c in cfg.experiment.candidate_policies if c != "random"]
    budgets = [float(b) for b in cfg.experiment.budgets if float(b) < 1.0]
    temps = list(TEMPS)
    rows: list[dict] = []

    for ds_cfg in cfg.datasets:
        for seed in range(int(cfg.seed), int(cfg.seed) + int(cfg.experiment.n_seeds)):
            dataset = _build_dataset(ds_cfg, seed, max_n)
            tr, ev = train_eval_split(dataset, eval_frac=0.5, seed=seed)
            train_ds, eval_ds = dataset.subset(tr), dataset.subset(ev)
            policies, scores = {}, {}
            for cname in cand_names:
                try:
                    pol = AllocationPolicy(
                        uplift_model=cname, variant="deterministic", seed=seed
                    ).fit(train_ds)
                except Exception as exc:  # noqa: BLE001
                    log.warning("skip %s: %s", cname, exc)
                    continue
                policies[cname] = pol
                scores[cname] = pol.score(eval_ds.X, feature_names=eval_ds.feature_names)
            if not policies:
                continue
            _assert_scores_are_functions_of_x(policies, eval_ds)
            s_ind = _independent_score(eval_ds, seed)

            for cname, pol in policies.items():
                logger_scores = {"self_aligned": scores[cname], "independent": s_ind}
                for regime in REGIMES:
                    for temp in temps:
                        for floor in FLOORS:
                            logged = make_logged_data(
                                eval_ds, logger_scores[regime], temperature=temp,
                                seed=seed, clip=floor,
                            )
                            om = OutcomeModel("lightgbm", seed=seed).fit(logged)
                            for bk in budgets:
                                true_val = true_allocation_value(eval_ds, scores[cname], bk)
                                tprob = pol.action_prob(
                                    logged.context, bk, feature_names=eval_ds.feature_names
                                )
                                diag = compute_diagnostics(logged, tprob, bk)
                                for ename, est in _estimators().items():
                                    try:
                                        res = est.estimate(
                                            logged, tprob, outcome_model=om,
                                            n_bootstrap=2, seed=seed,
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        log.warning("%s failed: %s", ename, exc)
                                        continue
                                    rows.append(
                                        dict(
                                            git_hash=git_hash,
                                            dataset=dataset.name,
                                            seed=seed,
                                            candidate_policy=cname,
                                            logger_regime=regime,
                                            overlap_temperature=temp,
                                            budget_k=bk,
                                            floor=floor,
                                            estimator=ename,
                                            value_hat=res.value,
                                            true_value=true_val,
                                            rel_bias=(res.value - true_val)
                                            / max(abs(true_val), 1e-8),
                                            max_weight=diag["max_weight"],
                                            ess_fraction=diag["ess_fraction"],
                                        )
                                    )
        log.info("done %s (%d rows)", ds_cfg.name, len(rows))

    df = pd.DataFrame(rows)
    out = Path(cfg.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "floor_sensitivity.parquet", index=False)

    K = ["dataset", "candidate_policy", "logger_regime", "overlap_temperature",
         "budget_k", "floor"]
    cells = (
        df.groupby(K + ["estimator"])
        .apply(
            lambda g: np.sqrt(((g.value_hat - g.true_value) ** 2).mean())
            / max(abs(g.true_value.mean()), 1e-8),
            include_groups=False,
        )
        .rename("rel_rmse")
        .reset_index()
    )
    print("\n=== observed max weight by floor (the tail actually gets heavy?) ===")
    print(df.pivot_table(index="floor", columns="logger_regime", values="max_weight",
                         aggfunc="median").round(1).to_string())
    print("\n=== median relative RMSE by floor x estimator ===")
    print(cells.pivot_table(index="floor", columns="estimator", values="rel_rmse")
          .round(4).to_string())

    wide = cells.pivot_table(index=K, columns="estimator", values="rel_rmse").reset_index()
    print("\n=== paired vs raw IPS by floor (negative => clipping HELPS) ===")
    for floor in FLOORS:
        sub = wide[wide.floor == floor]
        line = [f"  eps={floor:<7}"]
        for e in ("clipped_ips_m5", "clipped_ips_m50", "clipped_ips_m500", "shrinkage_dr"):
            base = "dr" if e == "shrinkage_dr" else "ips"
            d = (sub[e] - sub[base]).dropna()
            if len(d):
                line.append(f"{e.replace('clipped_ips_','clip'):>10s} {d.median():+.4f} "
                            f"({(d < 0).mean():.0%} help)")
        print("  ".join(line))
    log.info("Wrote %d rows -> %s", len(df), out / "floor_sensitivity.parquet")


if __name__ == "__main__":
    main()

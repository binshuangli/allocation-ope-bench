"""How much of DM's undercoverage is the frozen nuisance?

Every interval in this benchmark is a percentile bootstrap over per-unit contributions
with ``mu-hat`` held fixed, so it is *conditional on the fitted nuisance*. Appendix
coverage shows DM at ~0.89 against a nominal 0.95, and the paper attributes the shortfall
to exactly this: resampling the units without refitting the outcome model cannot express
the model's own estimation variance. That is an explanation, not a measurement.

This experiment measures it. On the exact-value datasets it computes, for each cell, both

* the CONDITIONAL interval (the benchmark's default: resample contributions, mu-hat fixed), and
* the REFIT-AWARE interval (resample the logged rows, refit mu-hat on each resample, and
  recompute the estimate end to end),

and scores both against the known true value. If the frozen nuisance is the cause, the
refit-aware interval should be wider and cover closer to nominal, and the gap should be
largest for DM (which is all nuisance) and smaller for DR (whose correction term is
weighted by observed data).

Refitting inside the bootstrap is expensive, so this runs on a deliberately reduced grid
and a smaller bootstrap count; it is a targeted measurement, not a replacement for the
main coverage table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from allocation_ope_bench.data import train_eval_split, true_allocation_value
from allocation_ope_bench.estimators.base import LoggedData, OutcomeModel
from allocation_ope_bench.estimators.direct import DirectMethod
from allocation_ope_bench.estimators.doubly_robust import DoublyRobust
from allocation_ope_bench.experiments.git_utils import get_git_hash
from allocation_ope_bench.experiments.runner import _build_dataset

log = logging.getLogger(__name__)

N_BOOT = 100
TEMPS = (0.5, 2.0)
BUDGETS = (0.1, 0.3)


def _refit_interval(logged, tprob, est_name, seed, n_boot=N_BOOT):
    """Percentile interval that refits mu-hat on every bootstrap resample."""
    rng = np.random.default_rng(seed)
    n = logged.n
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        lg = LoggedData(
            context=logged.context[idx], action=logged.action[idx],
            reward=logged.reward[idx], pscore=logged.pscore[idx],
            logging_prob_treat=None if logged.logging_prob_treat is None
            else logged.logging_prob_treat[idx],
        )
        om = OutcomeModel("lightgbm", seed=seed).fit(lg)   # refit inside the resample
        est = DirectMethod() if est_name == "dm" else DoublyRobust()
        vals.append(est.estimate(lg, tprob[idx], outcome_model=om, n_bootstrap=2, seed=seed).value)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from allocation_ope_bench.policies import AllocationPolicy, make_logged_data

    git_hash = get_git_hash()
    cand_names = [c for c in cfg.experiment.candidate_policies if c != "random"]
    rows: list[dict] = []

    for ds_cfg in cfg.datasets:
        for seed in range(int(cfg.seed), int(cfg.seed) + int(cfg.experiment.n_seeds)):
            dataset = _build_dataset(ds_cfg, seed, None)
            tr, ev = train_eval_split(dataset, eval_frac=0.5, seed=seed)
            train_ds, eval_ds = dataset.subset(tr), dataset.subset(ev)
            for cname in cand_names:
                try:
                    pol = AllocationPolicy(
                        uplift_model=cname, variant="deterministic", seed=seed
                    ).fit(train_ds)
                except Exception as exc:  # noqa: BLE001
                    log.warning("skip %s: %s", cname, exc)
                    continue
                s = pol.score(eval_ds.X, feature_names=eval_ds.feature_names)
                for temp in TEMPS:
                    logged = make_logged_data(eval_ds, s, temperature=temp, seed=seed)
                    om = OutcomeModel("lightgbm", seed=seed).fit(logged)
                    for bk in BUDGETS:
                        true_val = true_allocation_value(eval_ds, s, bk)
                        tprob = pol.action_prob(
                            logged.context, bk, feature_names=eval_ds.feature_names
                        )
                        for ename, est in (("dm", DirectMethod()), ("dr", DoublyRobust())):
                            cond = est.estimate(
                                logged, tprob, outcome_model=om, n_bootstrap=N_BOOT, seed=seed
                            )
                            try:
                                lo, hi = _refit_interval(logged, tprob, ename, seed)
                            except Exception as exc:  # noqa: BLE001
                                log.warning("refit failed: %s", exc)
                                continue
                            rows.append(dict(
                                git_hash=git_hash, dataset=dataset.name, seed=seed,
                                candidate_policy=cname, overlap_temperature=temp,
                                budget_k=bk, estimator=ename, true_value=true_val,
                                value_hat=cond.value,
                                cond_lo=cond.ci_low, cond_hi=cond.ci_high,
                                refit_lo=lo, refit_hi=hi,
                            ))
        log.info("done %s (%d rows)", ds_cfg.name, len(rows))

    df = pd.DataFrame(rows)
    out = Path(cfg.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "refit_intervals.parquet", index=False)

    df["cond_cov"] = (df.cond_lo <= df.true_value) & (df.true_value <= df.cond_hi)
    df["refit_cov"] = (df.refit_lo <= df.true_value) & (df.true_value <= df.refit_hi)
    df["cond_w"] = (df.cond_hi - df.cond_lo) / df.true_value.abs()
    df["refit_w"] = (df.refit_hi - df.refit_lo) / df.true_value.abs()
    print("\n=== coverage: conditional vs refit-aware (nominal 0.95) ===")
    print(df.groupby("estimator")[["cond_cov", "refit_cov"]].mean().round(3).to_string())
    print("\n=== median interval width / |V| ===")
    print(df.groupby("estimator")[["cond_w", "refit_w"]].median().round(3).to_string())
    print(f"\nn cells = {len(df)}")
    log.info("Wrote %d rows -> %s", len(df), out / "refit_intervals.parquet")


if __name__ == "__main__":
    main()

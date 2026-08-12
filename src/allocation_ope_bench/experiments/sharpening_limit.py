"""Score-aligned vs action-aligned logging under extreme sharpening.

An external review caught a false proposition in an earlier draft: we claimed that
sharpening a score-aligned logger cannot collapse overlap (ESS/n -> k + F(0)). The
correct large-sample limit is

    ESS/n  ->  [ c/(1-eps) + (1-c)/eps ]^{-1},        c = k + F(0),

because the mean-to-cutoff band's floor-probability draws carry weight 1/eps and an
O(1/eps) second-moment contribution that the false proof discarded. Sharpening a
SCORE-aligned logger therefore eventually collapses overlap -- the collapse simply begins
just below the temperature range the main sweep tests. A genuinely ACTION-aligned logger
(logistic centered at the target's top-k cutoff rather than the score mean) has no
mismatch band, and sharpening drives ESS/n toward 1-eps.

This experiment verifies both statements on the benchmark's own data: the two logger
centerings crossed with temperatures extending well below the main grid
(tau in {0.05, 0.1, 0.25, 0.5, 2, 5}), recording ESS and IPS error. Only IPS and the
diagnostics are needed, so it is cheap. Note the action-aligned logger depends on the
budget, so its log is built per (candidate, tau, budget) rather than per (candidate, tau).
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from allocation_ope_bench.data import train_eval_split, true_allocation_value
from allocation_ope_bench.estimators.ips import IPS
from allocation_ope_bench.experiments.git_utils import get_git_hash
from allocation_ope_bench.experiments.logger_alignment import _assert_scores_are_functions_of_x
from allocation_ope_bench.experiments.runner import _build_dataset
from allocation_ope_bench.metrics import compute_diagnostics

log = logging.getLogger(__name__)

TEMPS = (0.05, 0.1, 0.25, 0.5, 2.0, 5.0)
EPS = 0.02


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from allocation_ope_bench.policies import AllocationPolicy, make_logged_data

    git_hash = get_git_hash()
    max_n = cfg.get("max_n", None)
    max_n = int(max_n) if max_n is not None else None
    cand_names = [c for c in cfg.experiment.candidate_policies if c != "random"]
    budgets = [float(b) for b in cfg.experiment.budgets if float(b) < 1.0]
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

            for cname, pol in policies.items():
                s = scores[cname]
                for tau in TEMPS:
                    # score-aligned: one log serves all budgets (as in the main sweep)
                    logged_sc = make_logged_data(eval_ds, s, temperature=tau, seed=seed)
                    for bk in budgets:
                        tprob = pol.action_prob(
                            logged_sc.context, bk, feature_names=eval_ds.feature_names
                        )
                        centering_cases = [("score_aligned", logged_sc, tprob)]
                        # action-aligned: logistic centered at this budget's cutoff
                        cut = float(np.quantile(s, 1.0 - bk))
                        logged_ac = make_logged_data(
                            eval_ds, s, temperature=tau, seed=seed, center=cut
                        )
                        tprob_ac = pol.action_prob(
                            logged_ac.context, bk, feature_names=eval_ds.feature_names
                        )
                        centering_cases.append(("action_aligned", logged_ac, tprob_ac))
                        true_val = true_allocation_value(eval_ds, s, bk)
                        for regime, logged, tp in centering_cases:
                            diag = compute_diagnostics(logged, tp, bk)
                            try:
                                res = IPS().estimate(logged, tp, n_bootstrap=2, seed=seed)
                            except Exception as exc:  # noqa: BLE001
                                log.warning("ips failed: %s", exc)
                                continue
                            rows.append(
                                dict(
                                    git_hash=git_hash,
                                    dataset=dataset.name,
                                    seed=seed,
                                    candidate_policy=cname,
                                    regime=regime,
                                    tau=tau,
                                    budget_k=bk,
                                    ess_fraction=diag["ess_fraction"],
                                    max_weight=diag["max_weight"],
                                    n_logged=int(logged.n),
                                    value_hat=res.value,
                                    true_value=true_val,
                                    rel_bias=(res.value - true_val)
                                    / max(abs(true_val), 1e-8),
                                )
                            )
        log.info("done %s (%d rows)", ds_cfg.name, len(rows))

    df = pd.DataFrame(rows)
    out = Path(cfg.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "sharpening_limit.parquet", index=False)

    print("\n=== median ESS fraction by regime x tau ===")
    print(df.pivot_table(index="regime", columns="tau", values="ess_fraction",
                         aggfunc="median").round(3).to_string())
    print("\n=== median |IPS rel bias| by regime x tau ===")
    df["ab"] = df.rel_bias.abs()
    print(df.pivot_table(index="regime", columns="tau", values="ab",
                         aggfunc="median").round(3).to_string())
    k1 = df[(df.budget_k == 0.1)]
    print("\n=== k=0.1 median ESS by regime x tau (the corrected limit's sharpest case) ===")
    print(k1.pivot_table(index="regime", columns="tau", values="ess_fraction",
                         aggfunc="median").round(3).to_string())
    c = 0.1 + 0.5
    print(f"\ncorrected score-aligned limit at k=0.1 (Gaussian-ish scores): "
          f"{1/(c/(1-EPS)+(1-c)/EPS):.3f};  action-aligned limit: {1-EPS:.3f}")
    log.info("Wrote %d rows -> %s", len(df), out / "sharpening_limit.parquet")


if __name__ == "__main__":
    main()

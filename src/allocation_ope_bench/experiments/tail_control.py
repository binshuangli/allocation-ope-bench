"""Does tail control (weight clipping, shrinkage-DR) help where the tail actually exists?

The clipping study of the main paper ran under SELF-ALIGNED logging, where the propensity
floor keeps the median per-cell max weight near 5 and clipping has nothing to remove. Its
own text names the untested case as the interesting one: the candidate-independent logger,
where max weights reach the floor ceiling of 50. This experiment closes that hole, and
evaluates the cited-but-previously-unrun shrinkage-DR (Su et al., 2020) at the same time.

Grid: the exact-value datasets (synthetic, IHDP, one ACIC nonlinear surface), all three
logger-alignment regimes, the full temperature x budget grid of the main sweep. Estimators:
IPS, clipped IPS at M in {5, 10, 50}, DR, and shrinkage-DR with its lambda tuned on the
logged sample by the same estimated-MSE style used for Switch-DR and mIPS. Policies are
fitted once per (dataset, seed); the score-stability assertion from the alignment
experiment guards the invariant that makes this safe.
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
    REGIMES,
    _assert_scores_are_functions_of_x,
    _independent_score,
)
from allocation_ope_bench.experiments.runner import _build_dataset
from allocation_ope_bench.metrics import compute_diagnostics

log = logging.getLogger(__name__)


def _estimators():
    return {
        "ips": IPS(),
        "clipped_ips_m5": ClippedIPS(5.0),
        "clipped_ips_m10": ClippedIPS(10.0),
        "clipped_ips_m50": ClippedIPS(50.0),
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
    budgets = [float(b) for b in cfg.experiment.budgets]
    temps = [float(t) for t in cfg.experiment.overlap_temperatures]
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
                    log.warning("skip %s on %s: %s", cname, ds_cfg.name, exc)
                    continue
                policies[cname] = pol
                scores[cname] = pol.score(eval_ds.X, feature_names=eval_ds.feature_names)
            if len(policies) < 2:
                continue
            _assert_scores_are_functions_of_x(policies, eval_ds)
            s_ind = _independent_score(eval_ds, seed)

            for cname, pol in policies.items():
                other = next(c for c in policies if c != cname)
                logger_scores = {
                    "self_aligned": scores[cname],
                    "misaligned": scores[other],
                    "independent": s_ind,
                }
                for regime in REGIMES:
                    for temp in temps:
                        logged = make_logged_data(
                            eval_ds, logger_scores[regime], temperature=temp, seed=seed
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
                                        estimator=ename,
                                        value_hat=res.value,
                                        true_value=true_val,
                                        rel_bias=(res.value - true_val)
                                        / max(abs(true_val), 1e-8),
                                        max_weight=diag["max_weight"],
                                        selected_lambda=getattr(est, "selected_lambda", None),
                                    )
                                )
        log.info("done %s (%d rows)", ds_cfg.name, len(rows))

    df = pd.DataFrame(rows)
    out = Path(cfg.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "tail_control.parquet", index=False)

    K = ["dataset", "candidate_policy", "logger_regime", "overlap_temperature", "budget_k"]
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
    print("\n=== median relative RMSE by regime x estimator ===")
    print(
        cells.pivot_table(index="logger_regime", columns="estimator", values="rel_rmse")
        .round(4).to_string()
    )
    ind = cells[cells.logger_regime == "independent"]
    piv = ind.pivot_table(index=[c for c in K if c != "logger_regime"],
                          columns="estimator", values="rel_rmse")
    print("\n=== INDEPENDENT logger: paired vs raw IPS (negative = clipping helps) ===")
    for e in ("clipped_ips_m5", "clipped_ips_m10", "clipped_ips_m50"):
        d = (piv[e] - piv["ips"]).dropna()
        print(f"  {e:16s} mean {d.mean():+.4f}  median {d.median():+.4f}  helps in {(d<0).mean():.0%} of cells")
    print("\n=== paired shrinkage_dr - dr by regime ===")
    piv2 = cells.pivot_table(index=K, columns="estimator", values="rel_rmse").reset_index()
    for r in REGIMES:
        d = piv2[piv2.logger_regime == r]
        dd = (d["shrinkage_dr"] - d["dr"]).dropna()
        print(f"  {r:13s} mean {dd.mean():+.4f}  median {dd.median():+.4f}  helps in {(dd<0).mean():.0%}")
    lam = df[df.estimator == "shrinkage_dr"].selected_lambda.astype(float)
    print("\nshrinkage lambda: inf share %.2f, finite median %.1f"
          % (np.isinf(lam).mean(), np.nanmedian(lam[~np.isinf(lam)])))
    print("clipping M=5 binds (max weight > 5) in %.0f%% of configurations"
          % (100 * (df[df.estimator == 'ips'].max_weight > 5).mean()))
    log.info("Wrote %d rows -> %s", len(df), out / "tail_control.parquet")


if __name__ == "__main__":
    main()

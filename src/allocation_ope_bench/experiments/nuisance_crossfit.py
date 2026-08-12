"""RQ1 robustness: does the model-based advantage depend on in-sample nuisance fitting?

In the main sweep the outcome model ``mu-hat`` is fit on the logged sample that the
estimators are then evaluated on. ``IPS`` uses no nuisance at all, so a reader can
reasonably ask whether the model-based-vs-weighting gap is partly an artifact of that
asymmetry. There is a second, opposite concern: in-sample fitting shrinks the DR
residuals ``y - mu-hat``, which mechanically pulls ``DR`` toward ``DM`` and could
manufacture the "DM and DR are indistinguishable" null reported in RQ1.

This experiment tests both by re-running the whole model-based family --- DM, DR,
Switch-DR and shrinkage-DR --- under BOTH nuisance modes on the same cells: the main
sweep's in-sample ``mu-hat`` and a five-fold out-of-fold ``mu-hat`` that never sees a
unit's own outcome (``OutOfFoldOutcomeModel``). IPS, which uses no nuisance at all, is
the fixed reference point, and frozen-policy cross-fitted DR is kept for continuity with
the earlier version of this check. The estimand is identical in both modes (the fixed
in-sample policy), so the paired difference isolates the nuisance-fitting asymmetry.
It is a robustness check on the family-level conclusion, not a benchmark estimator.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from allocation_ope_bench.data import train_eval_split, true_allocation_value
from allocation_ope_bench.estimators import get_ope_estimator
from allocation_ope_bench.estimators.base import OutcomeModel, OutOfFoldOutcomeModel
from allocation_ope_bench.experiments.git_utils import get_git_hash
from allocation_ope_bench.experiments.runner import _build_dataset

log = logging.getLogger(__name__)

ESTIMATORS = ("dm", "dr", "ips", "cross_fitted_dr", "clipped_ips")
# Model-based estimators re-run under both nuisance modes (IPS needs no nuisance).
FAMILY = ("dm", "dr", "switch_dr", "shrinkage_dr")


def _estimate(name, logged, target_prob, om, seed):
    est = get_ope_estimator(name)
    if name == "cross_fitted_dr":
        # Frozen-policy mode: passing target_prob keeps DR's fixed-policy estimand
        # and cross-fits ONLY the nuisance.
        return est.estimate(logged, target_prob, seed=seed).value
    if name in ("ips", "clipped_ips"):
        return est.estimate(logged, target_prob, seed=seed).value
    return est.estimate(logged, target_prob, outcome_model=om, seed=seed).value


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from allocation_ope_bench.policies import AllocationPolicy, make_logged_data

    git_hash = get_git_hash()
    max_n = cfg.get("max_n", None)
    max_n = int(max_n) if max_n is not None else None
    policies = list(cfg.experiment.candidate_policies)
    policies = [p for p in policies if p != "random"]  # a random score is not a target here
    rows = []

    for ds_cfg in cfg.datasets:
        for seed in range(int(cfg.seed), int(cfg.seed) + int(cfg.experiment.n_seeds)):
            dataset = _build_dataset(ds_cfg, seed, max_n)
            tr, ev = train_eval_split(dataset, eval_frac=0.5, seed=seed)
            train_ds, eval_ds = dataset.subset(tr), dataset.subset(ev)
            for pname in policies:
                pol = AllocationPolicy(uplift_model=pname, variant="deterministic", seed=seed)
                pol.fit(train_ds)
                scores = pol.score(eval_ds.X, feature_names=eval_ds.feature_names)
                for temp in cfg.experiment.overlap_temperatures:
                    logged = make_logged_data(eval_ds, scores, temperature=float(temp), seed=seed)
                    om = OutcomeModel("lightgbm", seed=seed).fit(logged)
                    om_oof = OutOfFoldOutcomeModel("lightgbm", seed=seed).fit(logged)
                    for bk in cfg.experiment.budgets:
                        if bk >= 1.0:
                            continue
                        true_val = true_allocation_value(eval_ds, scores, float(bk))
                        tprob = pol.action_prob(
                            logged.context, float(bk), feature_names=eval_ds.feature_names
                        )
                        # (estimator, nuisance mode) grid: IPS-family once, model-based twice.
                        in_sample = list(ESTIMATORS) + [
                            e for e in FAMILY if e not in ESTIMATORS
                        ]
                        runs = [(e, "in_sample", om) for e in in_sample]
                        runs += [(e, "out_of_fold", om_oof) for e in FAMILY]
                        for ename, mode, model in runs:
                            try:
                                v = _estimate(ename, logged, tprob, model, seed)
                            except Exception as exc:  # noqa: BLE001
                                log.warning("%s failed on %s: %s", ename, ds_cfg.name, exc)
                                continue
                            rows.append(
                                dict(
                                    git_hash=git_hash,
                                    dataset=dataset.name,
                                    seed=seed,
                                    candidate_policy=pname,
                                    overlap_temperature=float(temp),
                                    budget_k=float(bk),
                                    estimator=ename,
                                    nuisance=mode,
                                    value_hat=v,
                                    true_value=true_val,
                                )
                            )
        log.info("done %s (%d rows)", ds_cfg.name, len(rows))

    df = pd.DataFrame(rows)
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "nuisance_crossfit.parquet")

    by = ["dataset", "estimator", "nuisance", "candidate_policy", "overlap_temperature", "budget_k"]
    cells = (
        df.groupby(by)
        .apply(
            lambda g: np.sqrt(((g.value_hat - g.true_value) ** 2).mean())
            / max(abs(g.true_value.mean()), 1e-8),
            include_groups=False,
        )
        .rename("rel_rmse")
        .reset_index()
    )
    print("\n=== median relative RMSE by estimator x nuisance mode ===")
    print(cells.pivot_table(index="estimator", columns="nuisance",
                            values="rel_rmse", aggfunc="median").round(4).to_string())

    cfg_key = ["dataset", "candidate_policy", "overlap_temperature", "budget_k"]
    wide = cells.pivot_table(index=cfg_key, columns=["estimator", "nuisance"], values="rel_rmse")
    rng = np.random.default_rng(0)

    def paired(a, b, label):
        d = (wide[a] - wide[b]).dropna().to_numpy()
        if not d.size:
            return
        bt = np.array([rng.choice(d, d.size, True).mean() for _ in range(5000)])
        lo, hi = np.percentile(bt, [2.5, 97.5])
        flag = "EXCLUDES 0" if (lo < 0) == (hi < 0) else "straddles 0"
        print(f"  {label:52s} mean {d.mean():+.4f} median {np.median(d):+.4f} "
              f"[{lo:+.4f},{hi:+.4f}] {flag}")

    print("\n=== cost of honest nuisances (positive => out-of-fold WORSE) ===")
    for e in FAMILY:
        paired((e, "out_of_fold"), (e, "in_sample"), f"{e}: out-of-fold - in-sample")

    print("\n=== THE RQ1 FAMILY CLAIM under honest nuisances (negative => beats IPS) ===")
    for e in FAMILY:
        paired((e, "out_of_fold"), ("ips", "in_sample"), f"{e} (out-of-fold) - IPS")
    for e in FAMILY:
        d = (wide[(e, "out_of_fold")] - wide[("ips", "in_sample")]).dropna()
        print(f"  {e:14s} beats IPS in {(d < 0).mean():.0%} of {len(d)} configurations")

    print("\n=== does DM still tie DR once mu-hat is honest? ===")
    paired(("dm", "out_of_fold"), ("dr", "out_of_fold"), "DM - DR (both out-of-fold)")
    paired(("dm", "in_sample"), ("dr", "in_sample"), "DM - DR (both in-sample)")

    log.info("Wrote %d rows -> %s", len(df), out_dir / "nuisance_crossfit.parquet")


if __name__ == "__main__":
    main()

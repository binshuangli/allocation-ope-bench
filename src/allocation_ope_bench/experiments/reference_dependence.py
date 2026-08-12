"""Reference-dependence check: shared vs disjoint RCT reference split (review #3).

On the RCT datasets the reported error is relative to a Horvitz--Thompson
reference estimate, not exact truth. In the main runs the rejection-sampled
logged data and that reference are built from the SAME randomized evaluation
split, so the estimate and its reference share sampling noise. This experiment
tests whether that dependence changes the estimator ranking or effect sizes, by
recomputing everything under a fully DISJOINT three-way split:

    train (policy) | logged (feedback + estimate) | reference (HT value)

For each cell we score the same estimators against (a) the shared reference (HT
on the logged split) and (b) the disjoint reference (HT on independent units),
and compare per-estimator relative RMSE. If the ordering and magnitudes are
stable, the shared-split dependence is not driving the conclusions.

    python -m allocation_ope_bench.experiments.reference_dependence \
        n_jobs=4 "datasets=[{name: hillstrom},{name: lenta}]"
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allocation_ope_bench.data import true_allocation_value
from allocation_ope_bench.estimators import fixed_target_estimators, get_ope_estimator
from allocation_ope_bench.estimators.base import OutcomeModel
from allocation_ope_bench.estimators.registry import needs_policy_kwargs
from allocation_ope_bench.experiments.git_utils import get_git_hash
from allocation_ope_bench.experiments.runner import _build_dataset, _fit_candidate
from allocation_ope_bench.policies import make_logged_data

log = logging.getLogger(__name__)


def _three_way_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disjoint train / logged / reference indices (40/30/30)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    a, b = int(0.40 * n), int(0.70 * n)
    return perm[:a], perm[a:b], perm[b:]


def _estimate_one(logged, target_prob, scores_logged, shared_om, budget_k, seed, ename):
    est = get_ope_estimator(ename)
    if needs_policy_kwargs(ename):
        if ename == "cross_fitted_dr":
            return est.estimate(logged, budget_k=budget_k, seed=seed).value
        if ename == "perturbation_dr":
            return est.estimate(
                logged,
                target_prob,
                outcome_model=shared_om,
                scores=scores_logged,
                budget_k=budget_k,
                seed=seed,
            ).value
        return est.estimate(logged, target_prob, seed=seed).value
    return est.estimate(logged, target_prob, outcome_model=shared_om, seed=seed).value


def _cell(dataset, seed, temp, budget_k, git_hash) -> list[dict]:
    tr, lg, rf = _three_way_split(dataset.n, seed)
    logged_ds = dataset.subset(lg)
    ref_ds = dataset.subset(rf)
    rows: list[dict] = []
    for cname in ("t_learner", "s_learner", "random"):
        try:
            policy = _fit_candidate(cname, dataset.subset(tr), seed)
        except Exception as exc:  # noqa: BLE001
            log.warning("policy %s failed: %s", cname, exc)
            continue
        scores_eval = policy.score(logged_ds.X, feature_names=logged_ds.feature_names)
        logged = make_logged_data(logged_ds, scores_eval, temperature=temp, seed=seed)
        target_prob = policy.action_prob(
            logged.context, budget_k, feature_names=logged_ds.feature_names
        )
        scores_logged = policy.score(logged.context, feature_names=logged_ds.feature_names)
        shared_om = OutcomeModel(seed=seed).fit(logged)

        # Two references for the SAME estimate: shared (logged split) vs disjoint.
        v_shared = true_allocation_value(logged_ds, scores_eval, budget_k)
        scores_ref = policy.score(ref_ds.X, feature_names=ref_ds.feature_names)
        v_disjoint = true_allocation_value(ref_ds, scores_ref, budget_k)

        for ename in fixed_target_estimators():
            try:
                vhat = _estimate_one(
                    logged, target_prob, scores_logged, shared_om, budget_k, seed, ename
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("est %s failed: %s", ename, exc)
                continue
            rows.append(
                {
                    "git_hash": git_hash,
                    "dataset": dataset.name,
                    "seed": seed,
                    "overlap_temperature": temp,
                    "budget_k": budget_k,
                    "candidate_policy": cname,
                    "estimator": ename,
                    "value_hat": vhat,
                    "ref_shared": v_shared,
                    "ref_disjoint": v_disjoint,
                }
            )
    return rows


def _rel_rmse(g: pd.DataFrame, ref_col: str) -> float:
    err = g["value_hat"] - g[ref_col]
    denom = abs(g[ref_col].mean())
    return float(np.sqrt(np.mean(err**2)) / denom) if denom > 1e-8 else float("nan")


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    log.info("Reference-dependence config:\n%s", OmegaConf.to_yaml(cfg))
    git_hash = get_git_hash()
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    exp = cfg.experiment
    budgets = list(exp.budgets)
    temps = list(exp.get("overlap_temperatures", [2.0]))
    n_seeds = int(exp.n_seeds)
    max_n = int(cfg.get("max_n")) if cfg.get("max_n", None) is not None else None

    all_rows: list[dict] = []
    for ds_cfg in cfg.datasets:
        for so in range(n_seeds):
            seed = int(cfg.seed) + so
            ds = _build_dataset(ds_cfg, seed=seed, max_n=max_n)
            for t in temps:
                for b in budgets:
                    all_rows.extend(_cell(ds, seed, float(t), float(b), git_hash))

    df = pd.DataFrame(all_rows)
    out = results_dir / "reference_dependence.parquet"
    df.to_parquet(out, index=False)

    # Per (dataset, estimator): rel-RMSE under each reference, over configs.
    by = ["dataset", "estimator", "overlap_temperature", "budget_k", "candidate_policy"]
    recs = []
    for keys, g in df.groupby(by):
        recs.append(
            {
                **dict(zip(by, keys)),
                "rel_rmse_shared": _rel_rmse(g, "ref_shared"),
                "rel_rmse_disjoint": _rel_rmse(g, "ref_disjoint"),
            }
        )
    cells = pd.DataFrame(recs)
    summ = (
        cells.groupby(["dataset", "estimator"])[["rel_rmse_shared", "rel_rmse_disjoint"]]
        .median()
        .round(4)
    )
    print("\n=== Median relative RMSE: shared vs disjoint reference ===")
    print(summ.to_string())
    summ.to_csv(results_dir / "reference_dependence_summary.csv")
    log.info("Wrote %d rows -> %s", len(df), out)


if __name__ == "__main__":
    run()

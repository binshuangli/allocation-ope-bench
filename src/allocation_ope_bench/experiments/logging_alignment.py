"""Misaligned-logging robustness check (reviewer: is the overlap knob confounded?).

In the main benchmark the logging policy is a temperature-smoothed version of the
SAME candidate being evaluated (self-aligned logging): the temperature sweep varies
the sharpness of an aligned logger. But OPE is often requested precisely when the
logging policy is MISALIGNED with the new candidate. This experiment separates the two
axes by crossing the logging and target policies:

    aligned:    log under policy P, evaluate policy P   (the main-benchmark regime)
    misaligned: log under policy P, evaluate policy Q != P

For each cell we build logged feedback from the LOGGING policy's score, then score the
fixed-target estimators against the TARGET policy's true allocation value. Comparing
aligned vs misaligned per estimator shows whether the RQ1/RQ2 conclusions survive
logging misalignment.

    python -m allocation_ope_bench.experiments.logging_alignment \
        experiment=align_run n_jobs=4 max_n=50000 \
        "datasets=[{name: synthetic},{name: ihdp},{name: hillstrom},{name: lenta}]"
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allocation_ope_bench.data import train_eval_split, true_allocation_value
from allocation_ope_bench.estimators import fixed_target_estimators, get_ope_estimator
from allocation_ope_bench.estimators.base import OutcomeModel
from allocation_ope_bench.estimators.registry import needs_policy_kwargs
from allocation_ope_bench.experiments.git_utils import get_git_hash
from allocation_ope_bench.experiments.runner import _build_dataset, _fit_candidate
from allocation_ope_bench.policies import make_logged_data

log = logging.getLogger(__name__)

_CANDIDATES = ("t_learner", "s_learner")


def _cell(dataset, seed, temp, budget_k, git_hash) -> list[dict]:
    train_idx, eval_idx = train_eval_split(dataset, eval_frac=0.5, seed=seed)
    eval_ds = dataset.subset(eval_idx)
    train_ds = dataset.subset(train_idx)

    policies = {}
    for c in _CANDIDATES:
        try:
            policies[c] = _fit_candidate(c, train_ds, seed)
        except Exception as exc:  # noqa: BLE001
            log.warning("policy %s failed: %s", c, exc)
    if len(policies) < 2:
        return []

    rows: list[dict] = []
    for log_name, log_pol in policies.items():
        scores_log = log_pol.score(eval_ds.X, feature_names=eval_ds.feature_names)
        logged = make_logged_data(eval_ds, scores_log, temperature=temp, seed=seed)
        shared_om = OutcomeModel(seed=seed).fit(logged)
        for tgt_name, tgt_pol in policies.items():
            scores_tgt = tgt_pol.score(eval_ds.X, feature_names=eval_ds.feature_names)
            true_val = true_allocation_value(eval_ds, scores_tgt, budget_k)
            target_prob = tgt_pol.action_prob(
                logged.context, budget_k, feature_names=eval_ds.feature_names
            )
            scores_tgt_logged = tgt_pol.score(logged.context, feature_names=eval_ds.feature_names)
            for ename in fixed_target_estimators():
                est = get_ope_estimator(ename)
                try:
                    if needs_policy_kwargs(ename):
                        if ename == "cross_fitted_dr":
                            continue  # cross-fit derives its own policy; N/A here
                        if ename == "perturbation_dr":
                            res = est.estimate(
                                logged,
                                target_prob,
                                outcome_model=shared_om,
                                scores=scores_tgt_logged,
                                budget_k=budget_k,
                                seed=seed,
                            )
                        else:
                            res = est.estimate(logged, target_prob, seed=seed)
                    else:
                        res = est.estimate(logged, target_prob, outcome_model=shared_om, seed=seed)
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
                        "logging_policy": log_name,
                        "target_policy": tgt_name,
                        "alignment": "aligned" if log_name == tgt_name else "misaligned",
                        "estimator": ename,
                        "value_hat": res.value,
                        "true_value": true_val,
                        "abs_bias": abs(res.value - true_val),
                        "rel_bias": (res.value - true_val) / max(abs(true_val), 1e-8),
                    }
                )
    return rows


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    log.info("Logging-alignment config:\n%s", OmegaConf.to_yaml(cfg))
    git_hash = get_git_hash()
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    exp = cfg.experiment
    budgets = list(exp.budgets)
    temps = list(exp.get("overlap_temperatures", [2.0]))
    n_seeds = int(exp.n_seeds)
    max_n = int(cfg.get("max_n")) if cfg.get("max_n", None) is not None else None

    rows: list[dict] = []
    for ds_cfg in cfg.datasets:
        for so in range(n_seeds):
            seed = int(cfg.seed) + so
            ds = _build_dataset(ds_cfg, seed=seed, max_n=max_n)
            for t in temps:
                for b in budgets:
                    rows.extend(_cell(ds, seed, float(t), float(b), git_hash))

    df = pd.DataFrame(rows)
    out = results_dir / "logging_alignment.parquet"
    df.to_parquet(out, index=False)

    # Median relative RMSE per (dataset, estimator, alignment).
    by = [
        "dataset",
        "estimator",
        "alignment",
        "overlap_temperature",
        "budget_k",
        "logging_policy",
        "target_policy",
    ]

    def _rmse(g):
        rmse = np.sqrt(np.mean((g.value_hat - g.true_value) ** 2))
        return float(rmse / max(abs(g.true_value.mean()), 1e-8))

    cells = df.groupby(by).apply(_rmse, include_groups=False).rename("rel_rmse").reset_index()
    summ = cells.groupby(["estimator", "alignment"])["rel_rmse"].median().unstack().round(4)
    print("\n=== Median relative RMSE: aligned vs misaligned logging (pooled) ===")
    print(summ.to_string())
    summ.to_csv(results_dir / "logging_alignment_summary.csv")
    log.info("Wrote %d rows -> %s", len(df), out)


if __name__ == "__main__":
    run()

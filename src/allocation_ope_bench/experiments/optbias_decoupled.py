"""Policy/nuisance INDEPENDENCE ablation for the optimizer's-curse result.

In the main RQ3 experiment the target policy is derived from the very OutcomeModel that
DR then uses as its nuisance, so policy and nuisance share an inductive bias by
construction. A reviewer asked whether the sign of the frozen-policy cross-fitting
result depends on that coupling. Here the policy is still learned from an in-sample
LightGBM tau-hat (the curse must still be present), but the DR nuisance is a different
model class (ridge), so cross-fitting the nuisance no longer cross-fits anything the
policy was built from. If the sign survives, the finding is about policy-data
dependence rather than about model sharing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allocation_ope_bench.data import train_eval_split, true_allocation_value
from allocation_ope_bench.data.ground_truth import allocate_under_budget
from allocation_ope_bench.estimators import (
    get_ope_estimator,
    optimization_bias_estimators,
)
from allocation_ope_bench.estimators.base import OutcomeModel
from allocation_ope_bench.experiments.git_utils import get_git_hash
from allocation_ope_bench.experiments.runner import _build_dataset
from allocation_ope_bench.policies import make_logged_data

log = logging.getLogger(__name__)


def _opt_bias_cell(*, dataset, train_idx, eval_idx, seed, budget_k, git_hash) -> list[dict]:
    """One (seed, budget_k) optimizer's-curse cell across the three estimators."""
    eval_ds = dataset.subset(eval_idx)

    # temperature=0 => accept-all logging => logged aligns row-for-row with eval.
    # A neutral score is fine; the logging policy at T=0 ignores it.
    neutral = np.zeros(eval_ds.n)
    logged = make_logged_data(eval_ds, neutral, temperature=0.0, seed=seed)
    if logged.n != eval_ds.n:
        log.warning(
            "[%s] logged.n=%d != eval.n=%d at T=0; skipping cell.",
            dataset.name,
            logged.n,
            eval_ds.n,
        )
        return []

    # In-sample outcome model => the optimizer's curse.
    # Policy from an in-sample LightGBM tau-hat (curse present, as in the main run)...
    policy_model = OutcomeModel('lightgbm', seed=seed).fit(logged)
    scores = policy_model.predict(logged.context, 1) - policy_model.predict(logged.context, 0)
    # ...but the DR nuisance is a DIFFERENT model class, so the estimator's nuisance is
    # not the object the policy was built from.
    model = OutcomeModel('ridge', seed=seed).fit(logged)
    target = allocate_under_budget(scores, np.ones(logged.n), budget_k).astype(float)

    # Truth: true value of the in-sample-optimized policy (same policy for all 3).
    true_val = true_allocation_value(eval_ds, scores, budget_k)

    def _row(ename: str, value: float, lo: float, hi: float, truth: float) -> dict:
        return {
            "git_hash": git_hash,
            "regime": "optimization_bias",
            "dataset": dataset.name,
            "seed": seed,
            "budget_k": budget_k,
            "estimator": ename,
            "value_hat": value,
            "ci_low": lo,
            "ci_high": hi,
            "true_value": truth,
            "bias": value - truth,
            "abs_bias": abs(value - truth),
            "rel_bias": (value - truth) / max(abs(truth), 1e-8),
        }

    rows: list[dict] = []
    for ename in optimization_bias_estimators():
        est = get_ope_estimator(ename)
        try:
            if ename == "dr":
                res = est.estimate(logged, target, outcome_model=model, seed=seed)
            elif ename == "cross_fitted_dr":
                # FROZEN-POLICY mode: the in-sample policy is fixed; only the
                # outcome-model nuisance is cross-fitted. Same estimand as
                # plain DR (the value of `target`), so biases are comparable.
                res = est.estimate(logged, target, seed=seed)
            elif ename == "perturbation_dr":
                res = est.estimate(
                    logged,
                    target,
                    outcome_model=model,
                    scores=scores,
                    budget_k=budget_k,
                    seed=seed,
                )
            else:  # pragma: no cover - guarded by the registry set
                res = est.estimate(logged, target, outcome_model=model, seed=seed)
        except Exception as exc:  # noqa: BLE001
            log.warning("Estimator %s failed on %s: %s", ename, dataset.name, exc)
            continue
        rows.append(_row(ename, res.value, res.ci_low, res.ci_high, true_val))

    # FOLD-POLICY variant (secondary): the estimator re-derives the allocation
    # per fold from out-of-fold scores, so its estimand is the value of the
    # learning ALGORITHM, not of `target`. Score it against its own
    # fold-matched reference: the true value of each fold-specific policy on
    # its held-out fold, averaged with fold weights. Replicates the
    # estimator's internal folding exactly (same rng stream and split).
    try:
        from allocation_ope_bench.estimators.base import OutcomeModel as _OM

        rng = np.random.default_rng(seed)
        folds = np.array_split(rng.permutation(logged.n), 5)
        all_idx = np.arange(logged.n)
        phi = np.empty(logged.n)
        fold_truths, fold_sizes = [], []
        from allocation_ope_bench.estimators.doubly_robust import dr_contributions

        for fold in folds:
            train_idx = np.setdiff1d(all_idx, fold)
            om = _OM(seed=seed).fit(logged.subset(train_idx))
            s_f = om.predict(logged.context[fold], 1) - om.predict(logged.context[fold], 0)
            a_f = allocate_under_budget(s_f, np.ones(len(fold)), budget_k).astype(float)
            phi[fold] = dr_contributions(logged.subset(fold), a_f, om)
            fold_truths.append(true_allocation_value(eval_ds.subset(fold), s_f, budget_k))
            fold_sizes.append(len(fold))
        algo_value = float(phi.mean())
        algo_truth = float(np.average(fold_truths, weights=fold_sizes))
        from allocation_ope_bench.metrics.stats import bootstrap_ci

        _, lo, hi = bootstrap_ci(np.mean, phi, n_bootstrap=500, seed=seed)
        rows.append(_row("cross_fitted_dr_algo", algo_value, lo, hi, algo_truth))
    except Exception as exc:  # noqa: BLE001
        log.warning("Fold-policy variant failed on %s: %s", dataset.name, exc)

    return rows


def _opt_bias_cell_wrapper(args: dict) -> list[dict]:
    return _opt_bias_cell(**args)


def _run_dataset(*, ds_cfg, exp, base_seed, git_hash, n_jobs, max_n=None) -> list[dict]:
    ds_name = ds_cfg.name if hasattr(ds_cfg, "name") else str(ds_cfg)
    n_seeds = int(exp.n_seeds)
    budgets = list(exp.budgets)

    cell_args: list[dict] = []
    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        try:
            dataset = _build_dataset(ds_cfg, seed=seed, max_n=max_n)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to load %s (seed=%d): %s", ds_name, seed, exc)
            return []
        train_idx, eval_idx = train_eval_split(dataset, eval_frac=0.5, seed=seed)
        for budget_k in budgets:
            cell_args.append(
                dict(
                    dataset=dataset,
                    train_idx=train_idx,
                    eval_idx=eval_idx,
                    seed=seed,
                    budget_k=float(budget_k),
                    git_hash=git_hash,
                )
            )

    log.info("[%s] %d optimizer's-curse cells, n_jobs=%d", ds_name, len(cell_args), n_jobs)
    if n_jobs == 1:
        results = [_opt_bias_cell_wrapper(a) for a in cell_args]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_opt_bias_cell_wrapper)(a) for a in cell_args
        )
    out: list[dict] = []
    for r in results:
        out.extend(r)
    return out


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    log.info("Optimization-bias config:\n%s", OmegaConf.to_yaml(cfg))
    git_hash = get_git_hash()
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = int(cfg.get("n_jobs", 1))
    max_n = cfg.get("max_n", None)
    max_n = int(max_n) if max_n is not None else None

    all_rows: list[dict] = []
    for ds_cfg in cfg.datasets:
        ds_name = ds_cfg.name if hasattr(ds_cfg, "name") else str(ds_cfg)
        log.info("=== Optimization-bias dataset: %s ===", ds_name)
        rows = _run_dataset(
            ds_cfg=ds_cfg,
            exp=cfg.experiment,
            base_seed=int(cfg.seed),
            git_hash=git_hash,
            n_jobs=n_jobs,
            max_n=max_n,
        )
        all_rows.extend(rows)

    if not all_rows:
        log.error("No optimization-bias results collected.")
        return

    df = pd.DataFrame(all_rows)
    out_path = results_dir / "optimization_bias.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Wrote %d optimization-bias rows → %s", len(df), out_path)

    _print_summary(df)


def _print_summary(df: pd.DataFrame) -> None:
    print("\n=== Optimization-bias regime: signed bias & |bias| by estimator ===")
    summary = (
        df.groupby("estimator")
        .agg(
            mean_bias=("bias", "mean"),
            mean_abs_bias=("abs_bias", "mean"),
            std_abs_bias=("abs_bias", "std"),
            n=("abs_bias", "count"),
        )
        .reindex(optimization_bias_estimators())
    )
    print(summary.to_string())

    # De-biasing relative to plain DR (the optimistic baseline).
    if "dr" in summary.index:
        dr_bias = summary.loc["dr", "mean_abs_bias"]
        print("\n=== De-biasing vs plain DR (lower |bias| = better) ===")
        for est in summary.index:
            if est == "dr" or not np.isfinite(summary.loc[est, "mean_abs_bias"]):
                continue
            reduction = 100.0 * (1.0 - summary.loc[est, "mean_abs_bias"] / dr_bias)
            print(f"  {est:18s} removes {reduction:5.1f}% of plain-DR |bias|")


if __name__ == "__main__":
    run()

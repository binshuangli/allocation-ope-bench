"""WP4/WP5 — Hydra experiment runner.

Sweeps (datasets × overlap_temperature × seed × budget_k × candidate_policy × estimator)
and writes two tidy parquet files:

  results_dir/estimates.parquet  — one row per (cell × candidate_policy × estimator)
  results_dir/selection.parquet  — per-estimator selection metrics aggregated over
                                    candidate policies within each cell
  results_dir/anomalies.parquet  — rows flagged by the WP5 anomaly validator

Usage
-----
    # Smoke run (synthetic, 1 seed, 1 budget):
    python -m allocation_ope_bench.experiments.runner experiment=smoke

    # Medium run (~1 hour with n_jobs=4 on 4 cores):
    python -m allocation_ope_bench.experiments.runner experiment=medium_run n_jobs=4

    # Full run (all public datasets — see `make repro-full` for the dataset list):
    python -m allocation_ope_bench.experiments.runner experiment=full_run n_jobs=4 \\
        "datasets=[{name: synthetic},{name: hillstrom},{name: ihdp}]"

Parallelism
-----------
``n_jobs`` controls the number of joblib workers (default 1 = serial).
Each worker handles one (seed × overlap_temp × budget_k) cell — cells are
independent so parallelism is safe. Pass ``n_jobs=-1`` to use all cores.

Leakage guard
-------------
AllocationPolicy.fit() is ALWAYS called on the *train* split; eval split provides
the logged data and ground-truth oracle.  assert_no_feature_leakage() checks both
splits at startup.
"""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allocation_ope_bench.data import (
    Dataset,
    assert_no_feature_leakage,
    make_synthetic,
    train_eval_split,
    true_allocation_value,
)
from allocation_ope_bench.estimators import (
    fixed_target_estimators,
    get_ope_estimator,
    needs_policy_kwargs,
)
from allocation_ope_bench.estimators.base import OutcomeModel, OutOfFoldOutcomeModel
from allocation_ope_bench.experiments.git_utils import get_git_hash
from allocation_ope_bench.metrics import (
    compute_diagnostics,
    mean_sharpe_ratio,
    selection_regret,
)
from allocation_ope_bench.policies import AllocationPolicy, make_logged_data

log = logging.getLogger(__name__)


# ── Dataset factory ───────────────────────────────────────────────────────────


def _subsample(dataset: Dataset, max_n: int | None, seed: int) -> Dataset:
    """Randomly subsample a dataset to at most ``max_n`` rows (seeded).

    Large public RCTs (lenta ~700k, x5, criteo ~13M) are capped to keep the
    sweep tractable; a uniform random subsample preserves the known constant
    propensity, so the IPS oracle stays valid.
    """
    if max_n is None or dataset.n <= max_n:
        return dataset
    rng = np.random.default_rng(seed)
    idx = rng.choice(dataset.n, size=int(max_n), replace=False)
    idx.sort()
    return dataset.subset(idx)


def _build_dataset(ds_cfg: Any, seed: int, max_n: int | None = None) -> Dataset:
    """Build a dataset from a dataset config entry (DictConfig or dict).

    Real datasets are optionally capped to ``max_n`` rows (per-entry ``max_n``
    overrides the global cap; synthetic ignores it — its size is set by n_train).
    """
    if hasattr(ds_cfg, "name"):
        name = ds_cfg.name
    elif isinstance(ds_cfg, dict):
        name = ds_cfg["name"]
    else:
        name = str(ds_cfg)

    # Per-dataset max_n overrides the global cap when present.
    if hasattr(ds_cfg, "get") and ds_cfg.get("max_n", None) is not None:
        max_n = int(ds_cfg.get("max_n"))

    if name == "synthetic":
        n_train = int(ds_cfg.get("n_train", 5000)) if hasattr(ds_cfg, "get") else 5000
        n_test = int(ds_cfg.get("n_test", 2000)) if hasattr(ds_cfg, "get") else 2000
        n_features = int(ds_cfg.get("n_features", 10)) if hasattr(ds_cfg, "get") else 10
        effect_size = float(ds_cfg.get("effect_size", 1.0)) if hasattr(ds_cfg, "get") else 1.0
        treatment_rate = float(ds_cfg.get("treatment_rate", 0.5)) if hasattr(ds_cfg, "get") else 0.5
        return make_synthetic(
            n=n_train + n_test,
            n_features=n_features,
            seed=seed,
            effect_scale=effect_size,
            rct_propensity=treatment_rate,
            logging="rct",
            name=name,
        )

    if name == "acic":
        from allocation_ope_bench.data.acic_dgp import make_acic_ihdp

        setting = int(ds_cfg.get("setting", 1)) if hasattr(ds_cfg, "get") else 1
        covariates = str(ds_cfg.get("covariates", "ihdp")) if hasattr(ds_cfg, "get") else "ihdp"
        n_rows = ds_cfg.get("n_rows", None) if hasattr(ds_cfg, "get") else None
        n_rows = int(n_rows) if n_rows is not None else None
        return make_acic_ihdp(setting=setting, seed=seed, covariates=covariates, n_rows=n_rows)

    from allocation_ope_bench.data.loaders import (
        load_criteo,
        load_hillstrom,
        load_ihdp,
        load_jobs,
        load_lenta,
        load_twins,
        load_x5,
    )

    _LOADERS = {
        "hillstrom": load_hillstrom,
        "lenta": load_lenta,
        "x5": load_x5,
        "criteo": load_criteo,
        "ihdp": load_ihdp,
        "jobs": load_jobs,
        "twins": load_twins,
    }
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {sorted(_LOADERS)}")
    dataset = _LOADERS[name]()
    return _subsample(dataset, max_n, seed)


# ── Outcome-model-quality diagnostic ──────────────────────────────────────────


def _outcome_model_oof_rmse(logged, outcome_model: str, seed: int, n_folds: int = 5) -> dict:
    """Out-of-fold held-out RMSE of mu-hat on the logged sample.

    The model-based analogue of the ESS/support diagnostics: computable from
    logged data alone, it measures how well the outcome-model class actually
    predicts held-out logged rewards. Reported per arm and pooled, normalized
    by the logged-reward sd (so 1.0 ~ no better than predicting the mean).
    """
    n = logged.n
    y_sd = float(np.std(logged.reward))
    if n < 2 * n_folds or y_sd < 1e-12:
        return {"mu_rmse": float("nan"), "mu_rmse_t": float("nan"), "mu_rmse_c": float("nan")}
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    pred = np.empty(n)
    all_idx = np.arange(n)
    for fold in folds:
        train_idx = np.setdiff1d(all_idx, fold)
        om = OutcomeModel(outcome_model, seed=seed).fit(logged.subset(train_idx))
        a_f = logged.action[fold]
        Xf = logged.context[fold]
        p1 = om.predict(Xf, 1)
        p0 = om.predict(Xf, 0)
        pred[fold] = np.where(a_f == 1, p1, p0)
    resid = logged.reward - pred
    treated = logged.action == 1

    def _rmse(mask) -> float:
        return float(np.sqrt(np.mean(resid[mask] ** 2)) / y_sd) if mask.sum() >= 5 else float("nan")

    return {
        "mu_rmse": float(np.sqrt(np.mean(resid**2)) / y_sd),
        "mu_rmse_t": _rmse(treated),
        "mu_rmse_c": _rmse(~treated),
    }


# ── Candidate policy factory ───────────────────────────────────────────────────


def _fit_candidate(policy_name: str, train_ds: Dataset, seed: int) -> AllocationPolicy:
    return AllocationPolicy(uplift_model=policy_name, variant="deterministic", seed=seed).fit(
        train_ds
    )


# ── Per-cell estimation ────────────────────────────────────────────────────────


def _run_cell(
    *,
    dataset: Dataset,
    train_idx,
    eval_idx,
    seed: int,
    overlap_temperature: float,
    budget_k: float,
    candidate_names: list[str],
    estimator_names: list[str],
    git_hash: str,
    outcome_model: str = "lightgbm",
    nuisance: str = "in_sample",
    logger_center: str = "mean",
) -> tuple[list[dict], list[dict]]:
    """Return (estimate_rows, selection_rows) for one (seed, overlap_temp, budget_k) cell.

    ``outcome_model`` picks the shared mu-hat base learner; 'ridge' / 'stump'
    are the deliberately degraded options for the misspecification experiment.
    """
    train_ds = dataset.subset(train_idx)
    eval_ds = dataset.subset(eval_idx)

    # Fit candidate policies on train split ONLY — leakage guard.
    candidates: dict[str, AllocationPolicy] = {}
    for cname in candidate_names:
        try:
            candidates[cname] = _fit_candidate(cname, train_ds, seed)
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping candidate %s on %s: %s", cname, dataset.name, exc)

    if not candidates:
        log.warning("No candidates fitted for %s; skipping cell.", dataset.name)
        return [], []

    estimate_rows: list[dict] = []
    sel_true: dict[str, list[float]] = {e: [] for e in estimator_names}
    sel_hat: dict[str, list[float]] = {e: [] for e in estimator_names}

    for cname, policy in candidates.items():
        # Full-eval scores for true_allocation_value and make_logged_data.
        scores = policy.score(eval_ds.X, feature_names=eval_ds.feature_names)

        # Rejection sampling may drop units; logged.n <= eval_ds.X.shape[0].
        # ``logger_center='cutoff'`` centres the logistic at this budget's top-k score
        # boundary instead of the score mean. Appendix L names the mean-centred design as
        # a defect --- it makes tau incomparable across budgets and leaves the logger
        # aligned with the candidate's score *ranking* rather than its action boundary.
        # Under this option the logger is genuinely action-aligned, so tau means the same
        # thing at every budget. The log then depends on the budget, which is fine here
        # because a cell already fixes (seed, temperature, budget).
        _centre = (
            float(np.quantile(scores, 1.0 - budget_k)) if logger_center == "cutoff" else None
        )
        logged = make_logged_data(
            eval_ds, scores, temperature=overlap_temperature, seed=seed, center=_centre
        )

        # Ground truth on the full eval split (pre-rejection-sampling).
        true_val = true_allocation_value(eval_ds, scores, budget_k)

        # Target treat probabilities aligned with logged.context.
        target_prob = policy.action_prob(
            logged.context, budget_k, feature_names=eval_ds.feature_names
        )

        # Scores aligned to logged units (perturbation_dr needs this).
        scores_logged = policy.score(logged.context, feature_names=eval_ds.feature_names)

        # Shared outcome model — fit once, shared across standard estimators.
        # ``nuisance=out_of_fold`` serves each logged row a prediction from a fold that
        # excludes it, so the model-based estimators never see a unit's own outcome. This
        # is the honest-nuisance variant of the whole benchmark, not a reduced check.
        om_cls = OutOfFoldOutcomeModel if nuisance == "out_of_fold" else OutcomeModel
        shared_om = om_cls(outcome_model, seed=seed).fit(logged)

        diag = compute_diagnostics(logged, target_prob, budget_k)
        # Outcome-model-quality diagnostic (the DM/DR analogue of ESS for IPS):
        # out-of-fold held-out RMSE of mu-hat on the logged sample, per arm and
        # pooled, normalized by the logged-reward sd so it is comparable across
        # outcome scales. Computable without ground truth.
        diag.update(_outcome_model_oof_rmse(logged, outcome_model, seed))

        base_meta = {
            "git_hash": git_hash,
            "dataset": dataset.name,
            "outcome_model": outcome_model,
            "seed": seed,
            "overlap_temperature": overlap_temperature,
            "budget_k": budget_k,
            "candidate_policy": cname,
            "true_value": true_val,
            **{f"diag_{k}": v for k, v in diag.items()},
        }

        for ename in estimator_names:
            est = get_ope_estimator(ename)
            try:
                if needs_policy_kwargs(ename):
                    if ename == "cross_fitted_dr":
                        res = est.estimate(logged, budget_k=budget_k, seed=seed)
                    elif ename == "perturbation_dr":
                        res = est.estimate(
                            logged,
                            target_prob,
                            outcome_model=shared_om,
                            scores=scores_logged,
                            budget_k=budget_k,
                            seed=seed,
                        )
                    else:
                        res = est.estimate(logged, target_prob, seed=seed)
                else:
                    res = est.estimate(logged, target_prob, outcome_model=shared_om, seed=seed)

                row = {
                    **base_meta,
                    "estimator": ename,
                    "value_hat": res.value,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    "bias": res.value - true_val,
                    "abs_bias": abs(res.value - true_val),
                    "rel_bias": (res.value - true_val) / max(abs(true_val), 1e-8),
                    "selection_compatible": not needs_policy_kwargs(ename),
                }
                estimate_rows.append(row)

                if not needs_policy_kwargs(ename):
                    sel_true[ename].append(true_val)
                    sel_hat[ename].append(res.value)

            except Exception as exc:  # noqa: BLE001
                log.warning("Estimator %s failed on %s/%s: %s", ename, cname, dataset.name, exc)

    behavior_value = (
        float(np.mean([r["true_value"] for r in estimate_rows])) if estimate_rows else 0.0
    )

    selection_rows: list[dict] = []
    sel_meta = {
        "git_hash": git_hash,
        "dataset": dataset.name,
        "seed": seed,
        "overlap_temperature": overlap_temperature,
        "budget_k": budget_k,
        "n_candidates": len(candidates),
    }
    for ename in estimator_names:
        tv = sel_true[ename]
        hv = sel_hat[ename]
        if len(tv) < 2:
            continue
        tv_arr = np.array(tv)
        hv_arr = np.array(hv)
        sel = selection_regret(tv_arr, hv_arr)
        msr = mean_sharpe_ratio(tv_arr, hv_arr, behavior_value, k_min=2)
        cands = list(candidates.keys())
        selection_rows.append(
            {
                **sel_meta,
                "estimator": ename,
                "regret": sel["regret"],
                "regret_normalized": sel["regret_normalized"],
                "correct_selection": sel["correct"],
                "selected_policy": (
                    cands[sel["selected_policy"]]
                    if sel["selected_policy"] < len(cands)
                    else str(sel["selected_policy"])
                ),
                "best_policy": (
                    cands[sel["best_policy"]]
                    if sel["best_policy"] < len(cands)
                    else str(sel["best_policy"])
                ),
                "mean_sharpe_ratio_k2plus": msr,
            }
        )

    return estimate_rows, selection_rows


def _run_cell_wrapper(args: dict) -> tuple[list[dict], list[dict]]:
    """Top-level wrapper for joblib: unpacks args dict and calls _run_cell."""
    return _run_cell(**args)


# ── Per-dataset sweep ─────────────────────────────────────────────────────────


def _run_dataset(
    *,
    ds_cfg: Any,
    exp: DictConfig,
    base_seed: int,
    git_hash: str,
    n_jobs: int = 1,
    max_n: int | None = None,
    nuisance: str = "in_sample",
    logger_center: str = "mean",
) -> tuple[list[dict], list[dict]]:
    """All seeds × overlap temps × budgets for one dataset, with optional parallelism."""
    ds_name = ds_cfg.name if hasattr(ds_cfg, "name") else str(ds_cfg)
    # RQ1 fixed-target set (excludes cross_fitted_dr; it lives in the dedicated
    # optimization-bias experiment — see experiments/optimization_bias.py).
    estimator_names = fixed_target_estimators()
    candidate_names = list(exp.get("candidate_policies", ["t_learner", "s_learner", "random"]))
    outcome_model = str(exp.get("outcome_model", "lightgbm"))
    n_seeds = int(exp.n_seeds)
    budgets = list(exp.budgets)
    overlap_temps = list(exp.get("overlap_temperatures", [2.0]))

    # Load dataset once per seed (real datasets are idempotent given same data).
    cell_args: list[dict] = []

    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        try:
            dataset = _build_dataset(ds_cfg, seed=seed, max_n=max_n)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to load dataset %s (seed=%d): %s", ds_name, seed, exc)
            return [], []

        train_idx, eval_idx = train_eval_split(dataset, eval_frac=0.5, seed=seed)

        try:
            assert_no_feature_leakage(dataset.subset(train_idx))
            assert_no_feature_leakage(dataset.subset(eval_idx))
        except AssertionError as exc:
            log.error("Leakage detected in %s: %s — SKIPPING", ds_name, exc)
            return [], []

        for overlap_temp, budget_k in product(overlap_temps, budgets):
            cell_args.append(
                dict(
                    dataset=dataset,
                    train_idx=train_idx,
                    eval_idx=eval_idx,
                    seed=seed,
                    overlap_temperature=float(overlap_temp),
                    budget_k=float(budget_k),
                    candidate_names=candidate_names,
                    estimator_names=estimator_names,
                    git_hash=git_hash,
                    outcome_model=outcome_model,
                    nuisance=nuisance,
                    logger_center=logger_center,
                )
            )

    log.info(
        "[%s] Running %d cells (seeds=%d, overlaps=%d, budgets=%d) n_jobs=%d",
        ds_name,
        len(cell_args),
        n_seeds,
        len(overlap_temps),
        len(budgets),
        n_jobs,
    )

    if n_jobs == 1:
        results = []
        for i, args in enumerate(cell_args):
            log.info(
                "[%s] cell %d/%d: seed=%d overlap=%.1f budget=%.2f",
                ds_name,
                i + 1,
                len(cell_args),
                args["seed"],
                args["overlap_temperature"],
                args["budget_k"],
            )
            results.append(_run_cell_wrapper(args))
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_run_cell_wrapper)(args) for args in cell_args
        )

    all_est: list[dict] = []
    all_sel: list[dict] = []
    for est_rows, sel_rows in results:
        all_est.extend(est_rows)
        all_sel.extend(sel_rows)

    return all_est, all_sel


# ── Main entry point ───────────────────────────────────────────────────────────


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    git_hash = get_git_hash()
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets_cfg = cfg.datasets
    exp = cfg.experiment
    n_jobs = int(cfg.get("n_jobs", 1))
    max_n = cfg.get("max_n", None)
    max_n = int(max_n) if max_n is not None else None

    log.info(
        "Run: %d dataset(s), %d seed(s), %d budget(s), %d overlap temp(s), n_jobs=%d, max_n=%s",
        len(datasets_cfg),
        int(exp.n_seeds),
        len(exp.budgets),
        len(exp.get("overlap_temperatures", [2.0])),
        n_jobs,
        max_n,
    )

    all_estimates: list[dict] = []
    all_selection: list[dict] = []

    for ds_cfg in datasets_cfg:
        ds_name = ds_cfg.name if hasattr(ds_cfg, "name") else str(ds_cfg)
        log.info("=== Dataset: %s ===", ds_name)
        est_rows, sel_rows = _run_dataset(
            ds_cfg=ds_cfg,
            exp=exp,
            base_seed=int(cfg.seed),
            git_hash=git_hash,
            n_jobs=n_jobs,
            max_n=max_n,
            nuisance=str(cfg.get("nuisance", "in_sample")),
            logger_center=str(cfg.get("logger_center", "mean")),
        )
        all_estimates.extend(est_rows)
        all_selection.extend(sel_rows)
        log.info(
            "Dataset %s: %d estimate rows, %d selection rows",
            ds_name,
            len(est_rows),
            len(sel_rows),
        )

    if not all_estimates:
        log.error("No results collected — check warnings above.")
        return

    est_df = pd.DataFrame(all_estimates)
    sel_df = pd.DataFrame(all_selection)

    est_path = results_dir / "estimates.parquet"
    sel_path = results_dir / "selection.parquet"
    est_df.to_parquet(est_path, index=False)
    sel_df.to_parquet(sel_path, index=False)

    log.info("Wrote %d estimate rows → %s", len(est_df), est_path)
    log.info("Wrote %d selection rows → %s", len(sel_df), sel_path)

    # Validation / anomaly detection.
    from allocation_ope_bench.experiments.validate import (
        print_validation_report,
        validate_estimates,
    )

    flagged = validate_estimates(est_df)
    flagged_path = results_dir / "anomalies.parquet"
    flagged.to_parquet(flagged_path, index=False)
    print_validation_report(flagged, title="WP5 Anomaly Report")
    if not flagged.empty:
        log.warning("%d anomalous rows flagged — see %s", len(flagged), flagged_path)

    _print_summary(est_df, sel_df)


def _print_summary(est_df: pd.DataFrame, sel_df: pd.DataFrame) -> None:
    print("\n=== Estimate bias summary (abs_bias by dataset × estimator) ===")
    grp_cols = [c for c in ["dataset", "estimator"] if c in est_df.columns]
    summary = (
        est_df.groupby(grp_cols)["abs_bias"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_abs_bias", "std": "std_abs_bias", "count": "n"})
        .sort_values(["dataset", "mean_abs_bias"] if "dataset" in grp_cols else "mean_abs_bias")
    )
    print(summary.to_string())

    if not sel_df.empty:
        print("\n=== Selection accuracy (correct_selection rate by estimator) ===")
        sel_summary = (
            sel_df.groupby("estimator")["correct_selection"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "correct_rate", "count": "n"})
            .sort_values("correct_rate", ascending=False)
        )
        print(sel_summary.to_string())


if __name__ == "__main__":
    run()

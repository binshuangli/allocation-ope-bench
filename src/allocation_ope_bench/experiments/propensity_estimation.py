"""What happens to the RQ1 ordering and the RQ2 screen when propensities are ESTIMATED?

Every other experiment in this benchmark hands the weighting estimators the exact logging
propensity. That is the single most favorable assumption we make, and in real observational
logs propensity-estimation error is the first-order problem rather than a secondary one --
so the guidance the benchmark produces has, until now, described a setting practitioners
rarely occupy outright.

This experiment replaces the true ``pi_b`` with an estimate ``pi_b-hat`` fitted on the
logged sample itself, and re-runs both the accuracy comparison and the fragility screen
under it. Four propensity conditions form a quality ladder:

* ``true``      -- the exact propensity (control; reproduces the main results)
* ``lightgbm``  -- flexible, correctly specified in the sense that the logger is a smooth
                   function of the candidate score, which is a function of x
* ``logistic``  -- linear in x; the logger is a logistic in the *standardized score*, so
                   this is well specified only when that score is near-linear in x
* ``marginal``  -- the sample treat rate, ignoring x entirely; the floor of the ladder and
                   what a practitioner has when no covariates are recorded

Two design choices worth stating. Propensities are fitted OUT OF FOLD (5-fold) so the same
rows do not both fit and use the model -- fitting in-sample would flatter the estimate in
exactly the way this experiment exists to test. And the estimate is floored at the same
0.02 the generator uses, so the comparison isolates estimation error rather than
re-introducing the unbounded-weight regime the benchmark deliberately excludes.

Both ``pscore`` and ``logging_prob_treat`` are replaced, so the ESS and support-deficiency
diagnostics are computed from the estimated propensity too -- which is what a practitioner
would actually have. A screen that only works on propensities you do not know would be of
no use.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from allocation_ope_bench.data import train_eval_split, true_allocation_value
from allocation_ope_bench.estimators import get_ope_estimator
from allocation_ope_bench.estimators.base import OutcomeModel
from allocation_ope_bench.experiments.git_utils import get_git_hash
from allocation_ope_bench.experiments.logger_alignment import (
    REGIMES,
    _assert_scores_are_functions_of_x,
    _independent_score,
)
from allocation_ope_bench.experiments.runner import _build_dataset
from allocation_ope_bench.metrics import compute_diagnostics

log = logging.getLogger(__name__)

PSCORE_MODELS = ("true", "lightgbm", "logistic", "marginal")
ESTIMATORS = ("dm", "ips", "snips", "dr", "switch_dr")
FLOOR = 0.02


def _fit_propensity(kind: str, X: np.ndarray, a: np.ndarray, seed: int, n_folds: int = 5):
    """Out-of-fold estimate of pi_b(1 | x). Returns probabilities on the same rows."""
    n = len(a)
    if kind == "marginal":
        # Still out-of-fold: the held-out mean, so it cannot see its own rows.
        rng = np.random.default_rng(seed)
        folds = np.array_split(rng.permutation(n), n_folds)
        out = np.empty(n)
        for f in folds:
            mask = np.ones(n, bool)
            mask[f] = False
            out[f] = a[mask].mean() if mask.any() else a.mean()
        return out

    from sklearn.linear_model import LogisticRegression

    def make():
        if kind == "logistic":
            return LogisticRegression(max_iter=1000, C=1.0)
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=100, num_leaves=15, learning_rate=0.1, random_state=seed, verbose=-1
        )

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    out = np.empty(n)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        if len(np.unique(a[tr])) < 2:  # degenerate fold: fall back to the marginal
            out[f] = a[tr].mean() if len(tr) else a.mean()
            continue
        m = make().fit(X[tr], a[tr])
        out[f] = m.predict_proba(X[f])[:, 1]
    return out


def _with_estimated_pscore(logged, p_treat_hat: np.ndarray):
    """Rebuild a LoggedData whose propensity -- and diagnostics -- use the estimate."""
    p = np.clip(p_treat_hat, FLOOR, 1.0 - FLOOR)
    pscore = np.where(logged.action == 1, p, 1.0 - p)
    return dataclasses.replace(logged, pscore=pscore, logging_prob_treat=p)


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
                    log.warning("skipping %s on %s: %s", cname, ds_cfg.name, exc)
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
                        # fit each propensity model once per logged sample
                        variants = {}
                        for kind in PSCORE_MODELS:
                            if kind == "true":
                                variants[kind] = (logged, 0.0, float("nan"))
                                continue
                            try:
                                ph = _fit_propensity(kind, logged.context, logged.action, seed)
                            except Exception as exc:  # noqa: BLE001
                                log.warning("propensity %s failed: %s", kind, exc)
                                continue
                            err = float(np.mean(np.abs(ph - logged.logging_prob_treat)))
                            corr = (
                                float(np.corrcoef(ph, logged.logging_prob_treat)[0, 1])
                                if np.std(ph) > 1e-12
                                else float("nan")
                            )
                            variants[kind] = (_with_estimated_pscore(logged, ph), err, corr)

                        for bk in budgets:
                            true_val = true_allocation_value(eval_ds, scores[cname], bk)
                            tprob = pol.action_prob(
                                logged.context, bk, feature_names=eval_ds.feature_names
                            )
                            for kind, (lg, err, corr) in variants.items():
                                diag = compute_diagnostics(lg, tprob, bk)
                                base = {
                                    "git_hash": git_hash,
                                    "dataset": dataset.name,
                                    "seed": seed,
                                    "candidate_policy": cname,
                                    "logger_regime": regime,
                                    "overlap_temperature": temp,
                                    "budget_k": bk,
                                    "pscore_model": kind,
                                    "pscore_mae": err,
                                    "pscore_corr": corr,
                                    "true_value": true_val,
                                    "n_logged": int(lg.n),
                                    **{f"diag_{k}": v for k, v in diag.items()},
                                }
                                for ename in ESTIMATORS:
                                    try:
                                        res = get_ope_estimator(ename).estimate(
                                            lg, tprob, outcome_model=om, seed=seed
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        log.warning("%s failed: %s", ename, exc)
                                        continue
                                    rows.append(
                                        {
                                            **base,
                                            "estimator": ename,
                                            "value_hat": res.value,
                                            "rel_bias": (res.value - true_val)
                                            / max(abs(true_val), 1e-8),
                                        }
                                    )
        log.info("done %s (%d rows)", ds_cfg.name, len(rows))

    df = pd.DataFrame(rows)
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "estimates.parquet", index=False)

    print("\n=== propensity-model quality (mean |pi-hat - pi|, corr with truth) ===")
    q = df[df.pscore_model != "true"].groupby("pscore_model")[["pscore_mae", "pscore_corr"]].mean()
    print(q.round(4).to_string())

    key = ["dataset", "candidate_policy", "logger_regime", "overlap_temperature", "budget_k"]
    cells = (
        df.groupby(key + ["pscore_model", "estimator"])
        .apply(
            lambda g: np.sqrt(((g.value_hat - g.true_value) ** 2).mean())
            / max(abs(g.true_value.mean()), 1e-8),
            include_groups=False,
        )
        .rename("rel_rmse")
        .reset_index()
    )
    print("\n=== median relative RMSE by propensity model x estimator ===")
    print(
        cells.pivot_table(index="pscore_model", columns="estimator", values="rel_rmse")
        .reindex(list(PSCORE_MODELS))
        .round(4)
        .to_string()
    )
    print("\n=== IPS failure rate (|rel bias| > 10%) by propensity model x regime ===")
    ips = df[df.estimator == "ips"].copy()
    ips["u"] = ips.rel_bias.abs() > 0.10
    print(
        ips.pivot_table(index="pscore_model", columns="logger_regime", values="u")
        .reindex(list(PSCORE_MODELS))
        .round(3)
        .to_string()
    )

    from scipy.stats import spearmanr

    print("\n=== does the ESS screen still rank IPS error under estimated propensities? ===")
    for kind in PSCORE_MODELS:
        sub = ips[ips.pscore_model == kind]
        e = sub.groupby(key).rel_bias.apply(lambda x: x.abs().median()).rename("e")
        d = sub.groupby(key)[["diag_ess_fraction", "diag_support_deficiency"]].mean()
        m = pd.concat([e, d], axis=1).dropna()
        if len(m) < 10 or m.diag_ess_fraction.nunique() < 2:
            continue
        y = (m.e > 0.10).to_numpy()
        from allocation_ope_bench.analysis.trust_inference import _auc

        auc = _auc(y, -m.diag_ess_fraction.to_numpy()) if 0 < y.sum() < len(y) else float("nan")
        print(
            f"  {kind:9s} n={len(m):4d}  rho(ESS)={spearmanr(m.diag_ess_fraction, m.e).statistic:+.3f}"
            f"  AUC={auc:.3f}  unsafe={y.mean():.3f}"
        )
    log.info("Wrote %d rows -> %s", len(df), out_dir / "estimates.parquet")


if __name__ == "__main__":
    main()

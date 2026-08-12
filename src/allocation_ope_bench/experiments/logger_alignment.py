"""RQ2: overlap risk is driven by logger--target MISALIGNMENT, not logging sharpness.

The main sweep logs each candidate under a smoothed version of *its own* score
(``self_aligned``). That design cannot sweep overlap. A deterministic top-$k$ target is
best covered by a logger that mimics it, so sharpening a self-aligned logger (low
``temperature``) concentrates logging on exactly the units the target treats and overlap
*improves*. Measured on synthetic at k=0.2, holding the target fixed and varying only the
logger, median ESS fraction at tau=0.5 is 0.495 self-aligned, 0.435 misaligned and 0.170
under a candidate-independent logger -- and median max weight 6.8 / 17.2 / 31.5. The
temperature knob only bites in interaction with misalignment.

This experiment therefore makes alignment an explicit axis, crossed with temperature:

* ``self_aligned``  -- logger built from the candidate's own score (the main-sweep design)
* ``misaligned``    -- logger built from the *other* learned candidate's score, i.e. a
                       plausible incumbent that disagrees with the target
* ``independent``   -- logger built from a candidate-independent random score

Everything else matches the main runner: same dataset build, same 50/50 split, same seeds,
same rejection sampler, same shared LightGBM outcome model per logged sample, same
fixed-target estimator set, same per-row schema (so ``analysis.trust_inference`` and the
RQ2 aggregation consume it unchanged).

Policies are fitted once per (dataset, seed) rather than per cell. That is safe only
because every scoring rule is a deterministic function of x; the experiment asserts this
at startup rather than trusting it, since a stateful scorer previously broke exactly this
assumption (see ``models.baselines.RandomUpliftEstimator``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from allocation_ope_bench.data import train_eval_split, true_allocation_value
from allocation_ope_bench.estimators import fixed_target_estimators, get_ope_estimator
from allocation_ope_bench.estimators.base import OutcomeModel
from allocation_ope_bench.experiments.git_utils import get_git_hash
from allocation_ope_bench.experiments.runner import _build_dataset
from allocation_ope_bench.metrics import compute_diagnostics

log = logging.getLogger(__name__)

REGIMES = ("self_aligned", "misaligned", "independent")
# perturbation_dr targets a smoothed-policy estimand, so it is not comparable on the
# fixed-target error this experiment measures.
ESTIMATORS = tuple(e for e in fixed_target_estimators() if e != "perturbation_dr")


def _independent_score(eval_ds, seed: int) -> np.ndarray:
    """A candidate-independent logger score, drawn once per (dataset, seed).

    Offset the seed so this never coincides with the random *candidate*'s score --
    the logger must not be aligned with any candidate on the slate.
    """
    return np.random.default_rng(900_000 + seed).normal(size=eval_ds.n)


def _assert_scores_are_functions_of_x(policies, eval_ds) -> None:
    for name, pol in policies.items():
        a = pol.score(eval_ds.X, feature_names=eval_ds.feature_names)
        pol.action_prob(eval_ds.X, 0.3, feature_names=eval_ds.feature_names)
        b = pol.score(eval_ds.X, feature_names=eval_ds.feature_names)
        if not np.array_equal(a, b):
            raise RuntimeError(
                f"candidate {name!r} does not score as a function of x; hoisting the fit "
                "out of the cell loop would desynchronize logger, target and truth."
            )


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from allocation_ope_bench.policies import AllocationPolicy, make_logged_data

    git_hash = get_git_hash()
    max_n = cfg.get("max_n", None)
    max_n = int(max_n) if max_n is not None else None
    # Alignment is only meaningful between *learned* candidates; the random baseline is
    # the decoy for RQ4, not a target whose overlap we care about here.
    cand_names = [c for c in cfg.experiment.candidate_policies if c != "random"]
    budgets = [float(b) for b in cfg.experiment.budgets]
    temps = [float(t) for t in cfg.experiment.overlap_temperatures]
    logger_center = str(cfg.get("logger_center", "mean"))
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
                    log.warning("skipping candidate %s on %s: %s", cname, ds_cfg.name, exc)
                    continue
                policies[cname] = pol
                scores[cname] = pol.score(eval_ds.X, feature_names=eval_ds.feature_names)
            if len(policies) < 2:
                log.warning("need >=2 candidates for a misaligned logger; skipping %s", ds_cfg.name)
                continue
            _assert_scores_are_functions_of_x(policies, eval_ds)
            s_independent = _independent_score(eval_ds, seed)

            for cname, pol in policies.items():
                other = next(c for c in policies if c != cname)
                logger_scores = {
                    "self_aligned": scores[cname],
                    "misaligned": scores[other],
                    "independent": s_independent,
                }
                for regime in REGIMES:
                    for temp in temps:
                        # With mean-centring one log serves every budget. Centring at the
                        # top-k cutoff (Appendix L's corrected design) makes the logger
                        # budget-dependent, so the log must be rebuilt per budget.
                        _cut = logger_center == "cutoff"
                        logged = om = None
                        if not _cut:
                            logged = make_logged_data(
                                eval_ds, logger_scores[regime], temperature=temp, seed=seed
                            )
                            om = OutcomeModel("lightgbm", seed=seed).fit(logged)
                        for bk in budgets:
                            if _cut:
                                _c = float(np.quantile(logger_scores[regime], 1.0 - bk))
                                logged = make_logged_data(
                                    eval_ds, logger_scores[regime], temperature=temp,
                                    seed=seed, center=_c,
                                )
                                om = OutcomeModel("lightgbm", seed=seed).fit(logged)
                            true_val = true_allocation_value(eval_ds, scores[cname], bk)
                            tprob = pol.action_prob(
                                logged.context, bk, feature_names=eval_ds.feature_names
                            )
                            diag = compute_diagnostics(logged, tprob, bk)
                            base = {
                                "git_hash": git_hash,
                                "dataset": dataset.name,
                                "seed": seed,
                                "candidate_policy": cname,
                                "logger_regime": regime,
                                "logger_policy": other if regime == "misaligned" else regime,
                                "overlap_temperature": temp,
                                "budget_k": bk,
                                "true_value": true_val,
                                "n_logged": int(logged.n),
                                **{f"diag_{k}": v for k, v in diag.items()},
                            }
                            for ename in ESTIMATORS:
                                try:
                                    res = get_ope_estimator(ename).estimate(
                                        logged, tprob, outcome_model=om, seed=seed
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    log.warning("%s failed on %s: %s", ename, ds_cfg.name, exc)
                                    continue
                                rows.append(
                                    {
                                        **base,
                                        "estimator": ename,
                                        "value_hat": res.value,
                                        "ci_low": res.ci_low,
                                        "ci_high": res.ci_high,
                                        "rel_bias": (res.value - true_val)
                                        / max(abs(true_val), 1e-8),
                                    }
                                )
        log.info("done %s (%d rows)", ds_cfg.name, len(rows))

    df = pd.DataFrame(rows)
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "estimates.parquet", index=False)

    cellkey = ["dataset", "candidate_policy", "logger_regime", "overlap_temperature", "budget_k"]
    ips = df[df.estimator == "ips"].copy()
    ips["abs_rel_bias"] = ips.rel_bias.abs()
    print("\n=== median ESS fraction by regime x temperature ===")
    print(
        ips.pivot_table(
            index="logger_regime", columns="overlap_temperature",
            values="diag_ess_fraction", aggfunc="median",
        ).round(3).to_string()
    )
    print("\n=== median per-cell max weight by regime x temperature ===")
    print(
        ips.pivot_table(
            index="logger_regime", columns="overlap_temperature",
            values="diag_max_weight", aggfunc="median",
        ).round(2).to_string()
    )
    print("\n=== share of cells with IPS |relative bias| > 10% ===")
    print(
        ips.assign(u=ips.abs_rel_bias > 0.10)
        .pivot_table(index="logger_regime", columns="overlap_temperature", values="u")
        .round(3).to_string()
    )
    from scipy.stats import spearmanr

    cell_err = ips.groupby(cellkey).abs_rel_bias.median().rename("e")
    cell_dg = ips.groupby(cellkey)[
        ["diag_ess_fraction", "diag_support_deficiency", "diag_max_weight"]
    ].mean()
    m = pd.concat([cell_err, cell_dg], axis=1).dropna().reset_index()
    print(f"\n=== Spearman rho vs IPS |rel bias|, CELL level (n={len(m)}) ===")
    for lab, sub in [("pooled", m)] + [(f"  within {r}", m[m.logger_regime == r]) for r in REGIMES]:
        if sub.diag_ess_fraction.nunique() < 2:
            continue
        print(
            f"{lab:22s} n={len(sub):4d}  ESS {spearmanr(sub.diag_ess_fraction, sub.e).statistic:+.3f}"
            f"  supdef {spearmanr(sub.diag_support_deficiency, sub.e).statistic:+.3f}"
            f"  maxw {spearmanr(sub.diag_max_weight, sub.e).statistic:+.3f}"
        )
    log.info("Wrote %d rows -> %s", len(df), out_dir / "estimates.parquet")


if __name__ == "__main__":
    main()

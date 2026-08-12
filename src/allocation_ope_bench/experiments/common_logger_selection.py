"""RQ4 under a COMMON logger: policy selection on one shared logged dataset.

Why this exists
---------------
The main runner builds the logged sample from *each candidate's own* score, so a
candidate is evaluated on data collected under a behavior policy aligned with it.
Comparing those estimates across candidates therefore compares policy--logger
*pairs*, not policies: it conflates candidate quality with the candidate-specific
data-collection process. Real policy selection has one historical log and ranks all
candidates on it.

This experiment fixes that. Per cell we build ONE logged dataset from a designated
logger score and evaluate every candidate against those identical observations and
propensities. Two logger regimes are run:

* ``aligned_t``     -- logger built from the ``t_learner`` score (aligned with one
  candidate, misaligned with the others).
* ``independent``   -- logger built from a candidate-independent random score, so no
  candidate is favored by the data-collection process.

Selection quality is then computed exactly as in RQ4: which candidate each estimator
ranks highest, versus which candidate is genuinely best.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from allocation_ope_bench.data import train_eval_split, true_allocation_value
from allocation_ope_bench.estimators import (
    fixed_target_estimators,
    get_ope_estimator,
    needs_policy_kwargs,
)
from allocation_ope_bench.estimators.base import OutcomeModel
from allocation_ope_bench.experiments.git_utils import get_git_hash
from allocation_ope_bench.experiments.runner import _build_dataset, _fit_candidate
from allocation_ope_bench.metrics.selection import selection_regret
from allocation_ope_bench.policies import make_logged_data

log = logging.getLogger(__name__)

LOGGER_REGIMES = ("aligned_t", "independent")


def _logger_score(regime: str, candidates: dict, eval_ds, seed: int) -> np.ndarray:
    """The score defining the shared behavior policy for this cell."""
    if regime == "independent":
        # Candidate-independent logging: favors no candidate by construction.
        return np.random.default_rng(10_000 + seed).normal(size=eval_ds.n)
    if regime == "aligned_t":
        pol = candidates["t_learner"]
        return pol.score(eval_ds.X, feature_names=eval_ds.feature_names)
    raise ValueError(f"unknown logger regime {regime!r}")


def run_cell(
    *,
    dataset,
    eval_ds,
    candidates,
    cand_scores,
    seed,
    temperature,
    budget_k,
    estimator_names,
    regime,
    git_hash,
):
    if len(candidates) < 2:
        return []

    # ---- ONE shared logged dataset for every candidate in this cell ----
    s_log = _logger_score(regime, candidates, eval_ds, seed)
    logged = make_logged_data(eval_ds, s_log, temperature=temperature, seed=seed)
    shared_om = OutcomeModel("lightgbm", seed=seed).fit(logged)

    true_vals, est_vals = {}, {e: {} for e in estimator_names}
    for cname, pol in candidates.items():
        scores = cand_scores[cname]
        true_vals[cname] = true_allocation_value(eval_ds, scores, budget_k)
        tprob = pol.action_prob(logged.context, budget_k, feature_names=eval_ds.feature_names)
        for ename in estimator_names:
            try:
                res = get_ope_estimator(ename).estimate(
                    logged, tprob, outcome_model=shared_om, seed=seed
                )
                est_vals[ename][cname] = res.value
            except Exception as exc:  # noqa: BLE001
                log.warning("estimator %s failed (%s/%s): %s", ename, dataset.name, cname, exc)

    rows = []
    for ename, vals in est_vals.items():
        if len(vals) < 2:
            continue
        # Score with the SHARED metric so regret and its normalization are defined
        # exactly as in the main RQ4 protocol (regret / value-spread across the
        # slate). Computing it inline here previously normalized by |V| instead,
        # which made regret incomparable across logging designs.
        cands = list(vals.keys())
        sel = selection_regret(
            np.array([true_vals[c] for c in cands]),
            np.array([vals[c] for c in cands]),
        )
        rows.append(
            dict(
                git_hash=git_hash,
                dataset=dataset.name,
                seed=seed,
                logger_regime=regime,
                overlap_temperature=temperature,
                budget_k=budget_k,
                estimator=ename,
                n_candidates=len(vals),
                selected_policy=cands[sel["selected_policy"]],
                best_policy=cands[sel["best_policy"]],
                correct_selection=float(sel["correct"]),
                regret=float(sel["regret"]),
                regret_normalized=float(sel["regret_normalized"]),
            )
        )
    return rows


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    git_hash = get_git_hash()
    # Selection-compatible estimators only: perturbation-DR re-solves the
    # allocation from the candidate's own scores, so it cannot rank a fixed slate.
    estimator_names = [e for e in fixed_target_estimators() if not needs_policy_kwargs(e)]
    candidate_names = list(cfg.experiment.candidate_policies)
    rows = []
    max_n = cfg.get("max_n", None)
    max_n = int(max_n) if max_n is not None else None
    for ds_cfg in cfg.datasets:
        # Seeds MUST match the main runner (base_seed + offset, cfg.seed=42), or the
        # train/eval splits and fitted candidates differ and the comparison against
        # the per-candidate-logger RQ4 results is confounded by seed.
        for seed in range(int(cfg.seed), int(cfg.seed) + int(cfg.experiment.n_seeds)):
            dataset = _build_dataset(ds_cfg, seed, max_n)
            tr, ev = train_eval_split(dataset, eval_frac=0.5, seed=seed)
            train_ds, eval_ds = dataset.subset(tr), dataset.subset(ev)

            # Candidates depend only on (dataset, seed) -- fit once and reuse
            # across every logger regime / temperature / budget in this seed.
            candidates, cand_scores = {}, {}
            for cname in candidate_names:
                try:
                    pol = _fit_candidate(cname, train_ds, seed)
                    candidates[cname] = pol
                    cand_scores[cname] = pol.score(eval_ds.X, feature_names=eval_ds.feature_names)
                except Exception as exc:  # noqa: BLE001
                    log.warning("skip candidate %s on %s: %s", cname, ds_cfg.name, exc)

            for regime in LOGGER_REGIMES:
                for temp in cfg.experiment.overlap_temperatures:
                    for bk in cfg.experiment.budgets:
                        if bk >= 1.0:
                            continue  # degenerate: every candidate treats everyone
                        rows.extend(
                            run_cell(
                                dataset=dataset,
                                eval_ds=eval_ds,
                                candidates=candidates,
                                cand_scores=cand_scores,
                                seed=seed,
                                temperature=float(temp),
                                budget_k=float(bk),
                                estimator_names=estimator_names,
                                regime=regime,
                                git_hash=git_hash,
                            )
                        )
        log.info("done %s (%d rows so far)", ds_cfg.name, len(rows))

    df = pd.DataFrame(rows)
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "common_logger_selection.parquet"
    df.to_parquet(out)

    print("\n=== correct-selection rate by logger regime (common logger) ===")
    print(
        df.pivot_table(
            index="estimator", columns="logger_regime", values="correct_selection", aggfunc="mean"
        )
        .round(3)
        .to_string()
    )
    print("\n=== how often each estimator picks the random-score candidate ===")
    print(
        df.assign(picks_random=(df.selected_policy == "random").astype(float))
        .pivot_table(
            index="estimator", columns="logger_regime", values="picks_random", aggfunc="mean"
        )
        .round(3)
        .to_string()
    )
    print(f"\nrandom is truly best in {(df.best_policy == 'random').mean():.3f} of cells")
    log.info("Wrote %d rows -> %s", len(df), out)


if __name__ == "__main__":
    main()

"""Which hyper-parameters do the two tuned hybrids actually select?

``Switch-DR`` chooses ``tau`` and ``mIPS`` chooses ``alpha`` on the logged sample by
minimizing an estimated-MSE proxy (see ``estimators.doubly_robust`` /
``estimators.ips``). The main sweep records only the resulting value estimates, and
those estimates turn out to be near-duplicates of their untuned parents: ``mIPS``
tracks ``IPS`` to within 0.001 and ``Switch-DR`` is statistically indistinguishable
from ``DR``. This experiment records the *selected* hyper-parameter itself, so the
near-duplication can be reported as a property of the selection rules
(``tau = inf`` reduces Switch-DR to DR; ``alpha = 0`` reduces mIPS to IPS) rather
than left unexplained.

It replays the main runner's cell construction exactly -- same dataset build, same
50/50 split, same seeds, same rejection-sampled logs, same shared LightGBM outcome
model -- and differs only in what is computed per cell: the two tuners, with the
bootstrap disabled since the point estimate is irrelevant here.

The candidate policy is refitted inside the (temperature, budget) loop rather than
hoisted, because ``runner._run_cell`` refits per cell and the ``random`` baseline holds
a *stateful* RNG that ``action_prob`` advances on every call (see
``models.baselines.RandomUpliftEstimator``). Hoisting the fit would silently desynchronize
this experiment's random-candidate cells from the main sweep's.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from allocation_ope_bench.data import train_eval_split
from allocation_ope_bench.estimators.base import OutcomeModel
from allocation_ope_bench.estimators.doubly_robust import SwitchDR
from allocation_ope_bench.estimators.ips import BIPS, _importance_weight
from allocation_ope_bench.experiments.git_utils import get_git_hash
from allocation_ope_bench.experiments.runner import _build_dataset

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from allocation_ope_bench.policies import AllocationPolicy, make_logged_data

    git_hash = get_git_hash()
    max_n = cfg.get("max_n", None)
    max_n = int(max_n) if max_n is not None else None
    rows = []

    for ds_cfg in cfg.datasets:
        for seed in range(int(cfg.seed), int(cfg.seed) + int(cfg.experiment.n_seeds)):
            dataset = _build_dataset(ds_cfg, seed, max_n)
            tr, ev = train_eval_split(dataset, eval_frac=0.5, seed=seed)
            train_ds, eval_ds = dataset.subset(tr), dataset.subset(ev)
            for pname in cfg.experiment.candidate_policies:
                for temp in cfg.experiment.overlap_temperatures:
                    for bk in cfg.experiment.budgets:
                        # Refit per cell, exactly as runner._run_cell does.
                        try:
                            pol = AllocationPolicy(
                                uplift_model=pname, variant="deterministic", seed=seed
                            ).fit(train_ds)
                        except Exception as exc:  # noqa: BLE001
                            log.warning("skipping %s on %s: %s", pname, ds_cfg.name, exc)
                            break
                        scores = pol.score(eval_ds.X, feature_names=eval_ds.feature_names)
                        logged = make_logged_data(
                            eval_ds, scores, temperature=float(temp), seed=seed
                        )
                        om = OutcomeModel("lightgbm", seed=seed).fit(logged)
                        tprob = pol.action_prob(
                            logged.context, float(bk), feature_names=eval_ds.feature_names
                        )
                        sdr, mips = SwitchDR(), BIPS()
                        sdr.estimate(logged, tprob, outcome_model=om, n_bootstrap=2, seed=seed)
                        mips.estimate(logged, tprob, n_bootstrap=2, seed=seed)
                        w = np.asarray(_importance_weight(logged, tprob))
                        rows.append(
                            dict(
                                git_hash=git_hash,
                                dataset=dataset.name,
                                seed=seed,
                                candidate_policy=pname,
                                overlap_temperature=float(temp),
                                budget_k=float(bk),
                                selected_tau=sdr.selected_tau,
                                selected_alpha=mips.selected_alpha,
                                max_weight=float(w.max()),
                                n_logged=int(logged.n),
                            )
                        )
        log.info("done %s (%d rows)", ds_cfg.name, len(rows))

    df = pd.DataFrame(rows)
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "tuner_selection.parquet")

    n = len(df)
    print(f"\n=== tuner selections over {n} cells ===")
    print("\nSwitch-DR selected tau:")
    print((df.selected_tau.value_counts(dropna=False).sort_index() / n).round(4).to_string())
    print(f"  share at tau = inf (Switch-DR == DR): {np.isinf(df.selected_tau).mean():.3f}")
    print(f"  share with tau >= max weight (no unit switched): "
          f"{(df.selected_tau >= df.max_weight).mean():.3f}")
    print("\nmIPS selected alpha:")
    print((df.selected_alpha.value_counts(dropna=False).sort_index() / n).round(4).to_string())
    print(f"  share at alpha = 0 (mIPS == IPS): {(df.selected_alpha == 0.0).mean():.3f}")
    print("\nBy overlap temperature (share degenerate):")
    print(
        df.assign(
            tau_inf=np.isinf(df.selected_tau),
            tau_inactive=df.selected_tau >= df.max_weight,
            alpha_0=df.selected_alpha == 0.0,
        )
        .groupby("overlap_temperature")[["tau_inf", "tau_inactive", "alpha_0"]]
        .mean()
        .round(3)
        .to_string()
    )
    log.info("Wrote %d rows -> %s", n, out_dir / "tuner_selection.parquet")


if __name__ == "__main__":
    main()

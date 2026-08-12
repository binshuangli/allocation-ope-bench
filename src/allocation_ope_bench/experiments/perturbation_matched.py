"""Is perturbation-DR accurate on its own estimand?

Perturbation-DR targets a smoothed allocation, E_eps[V(pi_{s+eps})], not the fixed
top-k allocation the rest of the benchmark scores, so ranking it against V(pi_e) in
the main tables conflates estimator error with an estimand gap. Appendix I promised
the estimand-coherent check as future work: score the estimate against its own
matched reference

    V_pert,ref = (1/M) sum_m V_exact(z_m),

where the z_m replicate the estimator's exact perturbation draws --- same
``default_rng(seed)`` stream, same ``scale = 0.5 * sd(scores)``, same M=25 --- and
each V_exact(z_m) is computed from the known per-unit arm means mu1/mu0 on the
exact-value datasets. If the estimator is accurate here, its poor showing against
the fixed-allocation truth is an estimand gap, not an estimation failure.

Runs the main grid on the exact-value datasets: synthetic + IHDP, both candidate
learners, tau in {0.5, 2, 5}, budgets {0.1, 0.2, 0.3, 0.5, 0.7}, seeds 42-51.
Summary convention matches the benchmark elsewhere: per-cell median of |relative
error| over seeds, then median over the 60 cells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from allocation_ope_bench.data import train_eval_split
from allocation_ope_bench.data.ground_truth import allocate_under_budget
from allocation_ope_bench.estimators.base import OutcomeModel
from allocation_ope_bench.estimators.perturbation_dr import PerturbationSmoothedDR
from allocation_ope_bench.experiments.runner import _build_dataset


def main() -> None:
    from allocation_ope_bench.policies import AllocationPolicy, make_logged_data

    rows: list[dict] = []
    for ds_name in ("synthetic", "ihdp"):
        ds_cfg = OmegaConf.create({"name": ds_name})
        for seed in range(42, 52):
            dataset = _build_dataset(ds_cfg, seed, None)
            tr, ev = train_eval_split(dataset, eval_frac=0.5, seed=seed)
            train_ds, eval_ds = dataset.subset(tr), dataset.subset(ev)
            for cname in ("t_learner", "s_learner"):
                pol = AllocationPolicy(
                    uplift_model=cname, variant="deterministic", seed=seed
                ).fit(train_ds)
                scores_eval = pol.score(eval_ds.X, feature_names=eval_ds.feature_names)
                for tau in (0.5, 2.0, 5.0):
                    logged = make_logged_data(
                        eval_ds, scores_eval, temperature=tau, seed=seed
                    )
                    om = OutcomeModel("lightgbm", seed=seed).fit(logged)
                    s_logged = pol.score(
                        logged.context, feature_names=eval_ds.feature_names
                    )
                    # exact per-unit arm means for the logged units, matched by row bytes
                    mu_map = {
                        r.tobytes(): (m1, m0)
                        for r, m1, m0 in zip(eval_ds.X, eval_ds.mu1, eval_ds.mu0)
                    }
                    mus = np.array([mu_map[r.tobytes()] for r in logged.context])
                    mu1_l, mu0_l = mus[:, 0], mus[:, 1]
                    costs = np.ones(logged.n)
                    for bk in (0.1, 0.2, 0.3, 0.5, 0.7):
                        est = PerturbationSmoothedDR()
                        res = est.estimate(
                            logged, None, outcome_model=om, n_bootstrap=2,
                            seed=seed, scores=s_logged, budget_k=bk,
                        )
                        # matched reference: replicate the estimator's own rng stream
                        rng = np.random.default_rng(seed)
                        s_sd = s_logged.std() + 1e-12
                        refs = []
                        for _ in range(25):
                            noise = rng.normal(scale=0.5 * s_sd, size=logged.n)
                            z = allocate_under_budget(s_logged + noise, costs, bk)
                            refs.append(float(np.where(z == 1, mu1_l, mu0_l).mean()))
                        vref = float(np.mean(refs))
                        rows.append(dict(
                            dataset=ds_name, seed=seed, cand=cname, tau=tau, bk=bk,
                            est=res.value, ref=vref,
                            rel=(res.value - vref) / max(abs(vref), 1e-8),
                        ))

    df = pd.DataFrame(rows)
    out = Path("results/perturbation_matched")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "matched.parquet")
    cells = (
        df.groupby(["dataset", "cand", "tau", "bk"])
        .rel.apply(lambda x: x.abs().median())
        .rename("e")
        .reset_index()
    )
    print("cells:", len(cells), " median |rel err|:", round(cells.e.median(), 4))
    print("by tau:", cells.groupby("tau").e.median().round(4).to_dict())
    print("fail>10%:", f"{(cells.e > 0.10).mean():.1%}")


if __name__ == "__main__":
    main()

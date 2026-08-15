# allocation-ope-bench

**When Can You Trust Offline Evaluation of Equal-Cost Top-k Allocation?**
**A Controlled, Reproducible Benchmark and Practitioner's Guide**

A reproducible benchmark that stress-tests six fixed-policy off-policy evaluation
(OPE) estimators (plus a smoothed-policy sensitivity estimator) under the
conditions that arise in real budget-constrained allocation: poor logging
overlap, a deterministic top-k rule that exposes weak overlap, and an optimizer's
curse when the policy is fit on the same data it is evaluated on.

> **Paper:** *When Can You Trust Offline Evaluation of Equal-Cost Top-k Allocation?
> A Controlled, Reproducible Benchmark and Practitioner's Guide* — Li (2026).
> Preprint: [arXiv:2608.12489](https://arxiv.org/abs/2608.12489).

> **Status:** the manuscript's numbers are regenerated from `results/` by
> `make analyze`, and `make check-numbers` re-derives ~75 headline values from
> those parquets and asserts they appear verbatim in the manuscript source, so
> drift fails the build rather than shipping.

---

## Quick start

```bash
# 1. Install (Python 3.11+)
pip install -e ".[dev]"
pre-commit install

# 2. Smoke check (~1 min)
make smoke

# 3. Unit tests
make test
```

## Reproduce the paper results

```bash
# Full RQ1–2 sweep (hours on 4 cores; needs ~2 GB disk for parquets)
make repro-full

# Optimizer's-curse experiment (RQ3–4)
make repro-optbias

# Analysis: figures + LaTeX tables + digest
make analyze
```

The analysis pipeline writes:
- `results/full_run/figures/` — 5 publication-quality PDFs + PNGs
- `results/full_run/tables/` — 3 LaTeX `booktabs` tables
- `results/full_run/analysis_digest.md` — numerical summary of all findings

`results/` is gitignored. The paper PDF is at [`paper/main.pdf`](paper/main.pdf).

---

## Datasets

All datasets are public. The benchmark downloads or loads them automatically
via their respective libraries.

| Dataset | Source | Regime | n (used) |
|---|---|---|---|
| Synthetic | generated | Synthetic | 7,000 |
| IHDP | Hill (2011) via `causalml` | Semi-synth. (continuous) | 672 |
| Jobs | LaLonde (1986) via `causalml` | Semi-synth. (RCT) | 578 |
| Hillstrom | Hillstrom (2008) via `sklift` | Marketing RCT | 50,000 (cap) |
| Lenta | Lenta (public) via `sklift` | Marketing RCT | 50,000 (cap) |

Large marketing RCTs are uniformly subsampled to 50,000 rows, which preserves
the constant propensity.

---

## Estimators

| Estimator | Key reference |
|---|---|
| DM (Direct Method) | Dudík et al. (2011) |
| IPS | Horvitz & Thompson (1952) |
| SNIPS (self-normalised IPS) | Swaminathan & Joachims (2015) |
| DR (Doubly Robust) | Dudík et al. (2011) |
| Switch-DR | Wang et al. (2017) |
| mixture-propensity IPS (mIPS; config key `bips`) | Swaminathan & Joachims (2015) |
| Perturbation DR | — |
| Cross-fitted DR (RQ3 only) | Chernozhukov et al. (2018) / DML |

---

## Key findings

Each is stated at the strength the evidence supports; the paper's Section 4.1
lists what the benchmark *cannot* establish.

1. **Overlap is governed by logger–target *action* alignment, not by logging
   sharpness.** For a deterministic top-k target the weight is
   `1/pi_b(a_target(x)|x)`, so what matters is the logger's probability of the
   *target's actions*. Over the tested range, sharpening a logger built from the
   target's own score barely moves overlap; action-level disagreement collapses
   it (IPS failure 8.3% / 13.3% / 31.7% across three logger regimes). Effective
   sample size ranks this risk **across logging environments** (ROC-AUC 0.85
   in-sample; 0.83 and 0.91 on two held-out families) — but only weakly *within*
   a single fixed log (median Spearman rho = -0.11), and its cut points do not
   transfer.

2. **The optimizer's curse is not fixed by cross-fitting the outcome nuisance.**
   When the rule is fit on the data used to evaluate it, cross-fitting the
   nuisance alone leaves the reuse bias in place and makes it *worse* (-16% to
   -36% across eight known-effect regimes). Honest policy-level splitting
   reduces bias magnitude by 58–92%, but by targeting the learning procedure's
   value — a change of estimand, not a de-biasing of the full-sample policy.
   The sign is tested, not derived: a normal-means toy leaves total DR optimism
   invariant to the nuisance, so the effect requires covariate-indexed structure.

3. **Propensity-estimation error is the largest degradation measured, and can
   invert the diagnostic.** Replacing the exact propensity with an out-of-fold
   estimate raises IPS failure from 6.3% to 37–63% of cells — dwarfing the
   2.8–11.7% from changing the logger regime — and a poor propensity model does
   not merely weaken the ESS screen but inverts it. A credible propensity model
   is the screen's prerequisite.

4. **Estimator accuracy: model-based methods lead where truth is exact.** On
   exact-value datasets DM (0.029) and the DR family (0.030) beat the IPS family
   (0.057) by roughly 2x. On randomized-trial-reference datasets the spread is
   0.003, which we do *not* read as a ranking — a noisy reference adds a common
   error floor. Exact-value and HT-reference results are never pooled.

5. **Policy selection depends on the candidate slate and the metric more than on
   the logging design.** We detect no logging-design effect that survives
   dataset-level clustering; the slate effect is larger than anything we could
   resolve. One robust pathology: IPS over-selects the easiest-to-evaluate
   candidate (up to 2.8x its true-best rate), with DM showing the same tendency
   at design-dependent strength.

6. **Validated against a non-simulated reference.** Every exact-value surface in
   the benchmark is simulated, so the mechanisms are additionally checked on
   Twins, whose evaluation reference is read from recorded paired outcomes: the
   RQ1 ordering (at a larger margin), the alignment mechanism, and the RQ3 sign
   all replicate, while the calibrated cut points do not.

---

## Project structure

```
src/allocation_ope_bench/
  data/         # dataset loaders + ground-truth oracle (true_allocation_value)
  policies/     # budget-constrained allocation + rejection-sampling logging
  estimators/   # OPE estimator registry (fixed_target / optimization_bias sets)
  metrics/      # relative RMSE, selection metrics, CI helpers
  experiments/  # Hydra runner (runner.py), optimizer's-curse (optimization_bias.py),
                #   anomaly validation (validate.py)
  analysis/     # aggregate.py, figures.py, tables.py, run.py (CLI)
conf/           # Hydra config tree (config.yaml, dataset overrides)
paper/          # LaTeX manuscript (main.tex, refs.bib, figures/, tables/)
tests/          # 101 offline unit tests
```

---

## Reproducibility notes

- Python ≥ 3.11, all dependencies pinned in `pyproject.toml`.
- Hydra configs under `conf/` are the single source of truth for sweep
  parameters (seeds, budgets, overlap temperatures).
- Each output parquet embeds a `git_hash` column for provenance.
- `make repro-full && make repro-optbias && make analyze` regenerates all paper
  artifacts deterministically.

### Erratum (August 2026)

An earlier draft claimed that sharpening a score-aligned logger cannot collapse
overlap (ESS/n → k + F(0)). That proposition was wrong: the mean-to-cutoff
band's floor-probability draws carry an O(1/ε) second-moment contribution, and
the correct limit is ESS/n → [c/(1−ε) + (1−c)/ε]⁻¹ ≈ ε/(1−c). The paper now
states the corrected limit, and `make repro-sharpening-limit` verifies both the
score-aligned collapse and the action-aligned logger's opposite behavior on the
benchmark's own data (see the docstring of
`src/allocation_ope_bench/experiments/sharpening_limit.py`).

---

## Citation

```bibtex
@article{li2026allocationope,
  title        = {When Can You Trust Offline Evaluation of Equal-Cost Top-$k$ Allocation?
                  {A} Controlled, Reproducible Benchmark and Practitioner's Guide},
  author       = {Li, Binshuang},
  journal      = {arXiv preprint arXiv:2608.12489},
  year         = {2026},
  eprint       = {2608.12489},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
}
```

---

## License

[MIT](LICENSE) — public data and personal research only. No employer data,
code, or proprietary insight is referenced anywhere in this repository.

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
> Preprint: arXiv (identifier added on posting).

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

1. **Overlap is the dominant driver of importance-weighting error.** IPS-family
   estimators degrade sharply under the poor overlap that deterministic
   allocation induces; DM and DR-family are robust. ESS fraction and support
   deficiency predict IPS error (Spearman |ρ| ≈ 0.43–0.48); a fragility screen
   validated out-of-DGP flags fragile evaluations before ground truth.

2. **The optimizer's curse is conditional, and not fixed by nuisance
   cross-fitting.** Plain DR is materially biased only on continuous known-effect
   data (synthetic, IHDP); nuisance-only cross-fitting does *not* remove it (it
   can worsen it), while honest policy-level splitting cuts the bias magnitude
   69–92% by evaluating the learning procedure. On binary RCTs no material bias
   is detected. Report per-dataset, never pooled.

3. **DM wins on selection.** DM selects the reference-best policy ~53–69% of the
   time vs ~27–29% for IPS (exact-value vs RCT-reference datasets), with the
   lowest normalized regret. Policy selection with OPE is substantially harder
   than point estimation.

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
  title   = {When Can You Trust Offline Evaluation of Equal-Cost Top-$k$ Allocation?
             {A} Controlled, Reproducible Benchmark and Practitioner's Guide},
  author  = {Li, Binshuang},
  year    = {2026},
}
```

---

## License

[MIT](LICENSE) — public data and personal research only. No employer data,
code, or proprietary insight is referenced anywhere in this repository.

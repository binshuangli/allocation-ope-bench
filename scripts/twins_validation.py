"""Does the benchmark's exact-value conclusion survive on OBSERVED ground truth?

Compares the Twins run against the published exact-value findings.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

K = ["dataset", "candidate_policy", "logger_regime", "overlap_temperature", "budget_k"]
FAM = {"dm": "DM", "dr": "DR", "switch_dr": "DR", "snips": "IPS", "ips": "IPS", "bips": "IPS"}


def rel_rmse(df):
    """Paper's convention: RMSE over seeds / |mean true value|, then median over configs."""
    g = df.groupby(["dataset", "candidate_policy", "overlap_temperature",
                    "budget_k", "estimator"])
    out = g.apply(lambda x: np.sqrt(((x.value_hat - x.true_value) ** 2).mean())
                  / max(abs(x.true_value.mean()), 1e-12), include_groups=False)
    return out.rename("rel_rmse").reset_index()


print("=" * 70)
print("RQ1 --- estimator accuracy on OBSERVED ground truth (Twins)")
print("=" * 70)
d = pd.read_parquet("results/twins_run/estimates.parquet")
d = d[d.estimator != "perturbation_dr"]
d = d[d.budget_k < 1.0]  # k=1 excluded, as throughout
r = rel_rmse(d)
med = r.groupby("estimator").rel_rmse.median().sort_values()
print("\nmedian relative RMSE (Twins):")
for e, v in med.items():
    print(f"   {e:14s} {v:.4f}   [{FAM.get(e,'?')} family]")

fam = r.assign(fam=r.estimator.map(FAM)).groupby("fam").rel_rmse.median()
print("\nby family (Twins):", {k: round(v, 4) for k, v in fam.items()})
print("published exact-value means: DM 0.029, DR 0.030, IPS family 0.057 (~2x gap)")
if "IPS" in fam and "DM" in fam:
    print(f"--> Twins IPS/DM ratio = {fam['IPS']/fam['DM']:.2f}x"
          f"   (published ~{0.057/0.029:.2f}x)")

# Value-range context: relative RMSE divides by |V| ~ 0.83 while the achievable
# spread of V is only ~0.10, so the metric compresses here.
print(f"\ntrue-value range in run: [{d.true_value.min():.4f}, {d.true_value.max():.4f}]"
      f"  mean |V| = {d.true_value.abs().mean():.4f}")
sp = d.true_value.max() - d.true_value.min()
print(f"error as share of achievable spread ({sp:.4f}):")
for e, v in med.items():
    print(f"   {e:14s} {v * d.true_value.abs().mean() / sp * 100:5.1f}% of spread")

print()
print("=" * 70)
print("RQ2 --- alignment vs sharpness on Twins")
print("=" * 70)
a = pd.read_parquet("results/logger_alignment_twins/estimates.parquet")
i = a[a.estimator == "ips"].copy()
i["ab"] = i.rel_bias.abs()
m = i.groupby(K).agg(e=("ab", "median"), ess=("diag_ess_fraction", "mean"),
                     sd=("diag_support_deficiency", "mean")).reset_index()

print("\nmedian ESS fraction by regime x temperature:")
piv = m.pivot_table(index="logger_regime", columns="overlap_temperature",
                    values="ess", aggfunc="median")
print(piv.round(3).to_string())
print("\n(published pattern: self-aligned FLAT across tau; misaligned/independent COLLAPSE at low tau)")

print("\nIPS failure rate (share of cells with median |rel bias| > 10%) at tau=0.5:")
t = m[m.overlap_temperature == 0.5]
trip = []
for rg in ("self_aligned", "misaligned", "independent"):
    s = t[t.logger_regime == rg]
    v = (s.e > 0.10).mean() * 100 if len(s) else float("nan")
    trip.append(v)
    print(f"   {rg:16s} {v:5.1f}%   (n={len(s)})")
print(f"   published five-dataset triple: 8.3% / 13.3% / 31.7%")

print("\nESS-vs-error association (the RQ2 screen):")
rho = spearmanr(m.ess, m.e).statistic
print(f"   pooled rho(ESS, |rel bias|) = {rho:+.3f}   (published pooled -0.41)")
for rg in ("self_aligned", "misaligned", "independent"):
    s = m[m.logger_regime == rg]
    if len(s) > 3:
        print(f"   {rg:16s} rho = {spearmanr(s.ess, s.e).statistic:+.3f}")

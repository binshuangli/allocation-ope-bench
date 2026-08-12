"""Assert that headline numbers in the manuscripts still match the released parquets.

Two independent review rounds found stale numbers that survived a claimed
"recomputed everything" pass, because the recomputation lived in a shell session and
not in the repo. This closes that: each entry below states a quantity, recomputes it
from ``results/``, formats it exactly as the paper prints it, and asserts the string
appears in both .tex sources. A number that drifts fails the build instead of shipping.

Run: python scripts/check_numbers.py            (or: make check-numbers)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
TEX = [ROOT / "paper" / "main.tex", ROOT / "paper_compact_readable" / "main.tex"]
K = ["dataset", "candidate_policy", "logger_regime", "overlap_temperature", "budget_k"]


def _auc(y, s):
    from allocation_ope_bench.analysis.trust_inference import _auc as a
    return a(y, s)


def _ladder():
    d = pd.read_parquet(ROOT / "results/logger_alignment/estimates.parquet")
    i = d[d.estimator == "ips"].copy()
    i["ab"] = i.rel_bias.abs()
    return i.groupby(K).agg(
        e=("ab", "median"), ess=("diag_ess_fraction", "mean"),
        sd=("diag_support_deficiency", "mean"), mw=("diag_max_weight", "mean"),
    ).reset_index()


def checks():
    out = []
    m = _ladder()

    # RQ2 conditioning ladder: pooled + the three logger regimes.
    y = (m.e > 0.10).to_numpy()
    out.append(("ladder pooled rho", f"${spearmanr(m.ess, m.e).statistic:+.2f}$"))
    out.append(("ladder pooled AUC", f"${_auc(y, -m.ess.to_numpy()):.2f}$"))
    for r, lab in (("self_aligned", "self"), ("misaligned", "misal"), ("independent", "indep")):
        s = m[m.logger_regime == r]
        ys = (s.e > 0.10).to_numpy()
        out.append((f"ladder {lab} rho", f"${spearmanr(s.ess, s.e).statistic:+.2f}$"))
        out.append((f"ladder {lab} unsafe", f"${ys.mean() * 100:.1f}\\%$"))

    # Per-dataset and LODO ranges quoted in Sec 5.2 and the ladder table.
    wd = [spearmanr(g.ess, g.e).statistic for _, g in m.groupby("dataset")]
    out.append(("within-dataset rho range", f"$-{abs(max(wd)):.2f}$ to $-{abs(min(wd)):.2f}$"))
    lo = [spearmanr(m[m.dataset != x].ess, m[m.dataset != x].e).statistic
          for x in m.dataset.unique()]
    out.append(("LODO range", f"$-{abs(max(lo)):.2f}$ to $-{abs(min(lo)):.2f}$"))

    # RQ2 headline failure triple at the sharpest tested temperature.
    t = m[m.overlap_temperature == 0.5]
    trip = [f"{(t[t.logger_regime == r].e > 0.10).mean() * 100:.1f}"
            for r in ("self_aligned", "misaligned", "independent")]
    out.append(("failure triple tau=0.5", f"${trip[0]}\\% \\to {trip[1]}\\% \\to {trip[2]}\\%$"))

    # Corrected (cutoff-centred) logger: the confirmation run.
    c = pd.read_parquet(ROOT / "results/logger_alignment_cutoff/estimates.parquet")
    ci = c[c.estimator == "ips"].copy()
    ci["ab"] = ci.rel_bias.abs()
    cm = ci.groupby(K).agg(e=("ab", "median"), ess=("diag_ess_fraction", "mean")).reset_index()
    ct = cm[cm.overlap_temperature == 0.5]
    ctrip = [f"{(ct[ct.logger_regime == r].e > 0.10).mean() * 100:.1f}"
             for r in ("self_aligned", "misaligned", "independent")]
    out.append(("cutoff triple", f"${ctrip[0]}\\%$ / ${ctrip[1]}\\%$ / ${ctrip[2]}\\%$"))
    sa = ci[ci.logger_regime == "self_aligned"].groupby("overlap_temperature")
    ess = sa.diag_ess_fraction.median()
    out.append(("cutoff ESS reversal", f"${ess[5.0]:.3f} \\to {ess[0.5]:.3f}$"))
    # The screen's association on the CORRECTED sweep -- the answer to "the design
    # keeping more failures also makes your diagnostic look more useful".
    cm2 = ci.groupby(K).agg(e=("ab", "median"), ess=("diag_ess_fraction", "mean")).reset_index()
    out.append(("cutoff pooled rho", f"$\\rho(\\text{{ESS}})=-{abs(spearmanr(cm2.ess, cm2.e).statistic):.2f}$"))
    ind2 = cm2[cm2.logger_regime == "independent"]
    out.append(("cutoff indep rho", f"$-{abs(spearmanr(ind2.ess, ind2.e).statistic):.2f}$"))
    yc = (cm2.e > 0.10).to_numpy()
    out.append(("cutoff pooled AUC", f"${_auc(yc, -cm2.ess.to_numpy()):.3f}$"))
    yi = (ind2.e > 0.10).to_numpy()
    out.append(("cutoff within-indep AUC", f"${_auc(yi, -ind2.ess.to_numpy()):.3f}$"))
    out.append(("cutoff failure concentration",
                f"${int(yi.sum())}$ of ${int(yc.sum())}$ failing cells"))

    # Decoupled-ablation synthetic exception (+1%): the one dataset where nuisance
    # cross-fitting is a wash rather than harmful, explained via the ridge
    # misspecification the omquality ladder establishes.
    dec = ROOT / "results/optbias_decoupled/optimization_bias.parquet"
    if dec.exists():
        dd2 = pd.read_parquet(dec)
        gg2 = dd2.groupby(["dataset", "estimator"]).abs_bias.mean().unstack()
        xr = (gg2["dr"] - gg2["cross_fitted_dr"]) / gg2["dr"] * 100
        out.append(("decoupled synthetic exception", f"${xr['synthetic']:+.0f}\\%$"))

    # Misspecification-ladder seed intervals (analysis-only, from the seed-level
    # rows already in results/misspec_run/): the DM-IPS crossing at the stump rung.
    ms = ROOT / "results/misspec_run/stump/estimates.parquet"
    ml = ROOT / "results/misspec_run/lightgbm/estimates.parquet"
    if ms.exists() and ml.exists():
        CFGm = ["dataset", "candidate_policy", "overlap_temperature", "budget_k",
                "estimator"]
        def _mats(path):
            dd = pd.read_parquet(path)
            dd = dd[dd.estimator.isin(["dm", "dr", "ips", "snips"])
                    & (dd.dataset == "synthetic")].copy()
            dd["err2"] = (dd.value_hat - dd.true_value) ** 2
            e2 = dd.pivot_table(index=CFGm, columns="seed", values="err2")
            tv = dd.pivot_table(index=CFGm, columns="seed", values="true_value")
            return (e2.to_numpy(), tv.to_numpy(),
                    e2.index.get_level_values("estimator").to_numpy())
        e2s, tvs, ests = _mats(ms)
        nsd = e2s.shape[1]
        rngm = np.random.default_rng(0)
        gaps = []
        for _ in range(2000):
            colsm = rngm.integers(0, nsd, nsd)
            rrm = (np.sqrt(e2s[:, colsm].mean(axis=1))
                   / np.maximum(np.abs(tvs[:, colsm].mean(axis=1)), 1e-12))
            gaps.append(np.median(rrm[ests == "dm"]) - np.median(rrm[ests == "ips"]))
        gaps = np.array(gaps)
        lo_g, hi_g = np.percentile(gaps, [2.5, 97.5])
        out.append(("ladder crossing CI", f"$[{lo_g:+.3f},\\,{hi_g:+.3f}]$"))
        out.append(("ladder crossing share",
                    f"${np.mean(gaps > 0) * 100:.0f}\\%$ of $2{{,}}000$ seed resamples"))

    # The delta grid behind the held-out screen validation (Table delta-grid): the
    # claim "we report the full delta grid" is now backed by a table; derive its
    # informative AUC cells so they cannot drift.
    for tag, rel in (("ihdpcov", "results/logger_alignment_acic/estimates.parquet"),
                     ("hillcov", "results/logger_alignment_acic_hillstrom/estimates.parquet")):
        pth = ROOT / rel
        if not pth.exists():
            continue
        dg = pd.read_parquet(pth)
        ig = dg[dg.estimator == "ips"].copy()
        ig["ab"] = ig.rel_bias.abs()
        mg = ig.groupby(K).agg(e=("ab", "median"),
                               ess=("diag_ess_fraction", "mean")).reset_index()
        for delta in (0.01, 0.02, 0.05, 0.10):
            yg = (mg.e > delta).to_numpy()
            if 0 < yg.sum() < len(yg):
                out.append((f"delta-grid {tag} @{delta:.0%}",
                            f"{_auc(yg, -mg.ess.to_numpy()):.2f}"))

    # Per-regime AUCs and the regime x tau summary -- the rows the first harness skipped,
    # and where the stale 5-seed prose survived alongside a corrected table.
    for r, lab in (("self_aligned", "self"), ("misaligned", "misal"), ("independent", "indep")):
        s = m[m.logger_regime == r]
        ys = (s.e > 0.10).to_numpy()
        out.append((f"ladder {lab} AUC", f"${_auc(ys, -s.ess.to_numpy()):.2f}$"))
    rs = [spearmanr(g.ess, g.e).statistic for _, g in m.groupby(["logger_regime",
                                                                "overlap_temperature"])]
    out.append(("regime x tau", f"${np.median(rs):+.2f}$"))
    out.append(("regime x tau sign count", f"${sum(r < 0 for r in rs)}/9$"))

    # Cluster-bootstrap intervals (Appendix). These were the last surviving 5-seed
    # artifact precisely because no check covered them.
    m2 = m.copy()
    m2["dp"] = m2.dataset + "|" + m2.candidate_policy
    rng = np.random.default_rng(0)
    for col, lab in (("ess", "ESS fraction"), ("sd", "support deficiency"),
                     ("mw", "max weight")):
        if col not in m2:
            continue
        for key, tag in (("dataset", "5cl"), ("dp", "10cl")):
            ks = m2[key].unique()
            bt = []
            for _ in range(4000):
                pick = rng.choice(ks, len(ks), True)
                s = pd.concat([m2[m2[key] == p] for p in pick])
                r = spearmanr(s[col], s.e).statistic
                if np.isfinite(r):
                    bt.append(r)
            lo, hi = np.percentile(bt, [2.5, 97.5])
            out.append((f"cluster CI {lab} ({tag})", f"$[{lo:+.2f},\\,{hi:+.2f}]$"))

    # Appendix I: perturbation-DR scored against its own matched (smoothed-policy)
    # reference. The paper claims accuracy on its own estimand; assert the numbers
    # and the "no cell exceeds 10%" claim itself.
    pm = ROOT / "results/perturbation_matched/matched.parquet"
    if pm.exists():
        p = pd.read_parquet(pm)
        cells = (p.groupby(["dataset", "cand", "tau", "bk"])
                 .rel.apply(lambda x: x.abs().median()))
        assert (cells > 0.10).sum() == 0, "pert-matched: a cell exceeds 10% rel err"
        out.append(("pert matched cells", f"${len(cells)}$ cells"))
        out.append(("pert matched median", f"${cells.median() * 100:.1f}\\%$"))
        bt2 = cells.groupby(level="tau").median() * 100
        out.append(("pert matched by tau",
                    f"${bt2[0.5]:.1f}\\%/{bt2[2.0]:.1f}\\%/{bt2[5.0]:.1f}\\%$"))

    # Incremental-value renormalization (Table 1's incremental columns). The prose
    # previously carried 5-seed values from a lost shell session -- unreproducible
    # under six conventions -- precisely because nothing asserted them. Convention
    # pinned here: RMSE over seeds / |mean over seeds of V - V(treat-none)|, per
    # config, median over configs, all budgets; V(treat-none) = mean mu0 on the
    # seed's eval split.
    fr = ROOT / "results/full_run/estimates.parquet"
    if fr.exists():
        from omegaconf import OmegaConf

        from allocation_ope_bench.data import train_eval_split
        from allocation_ope_bench.experiments.runner import _build_dataset

        vbase = {}
        for ds in ("synthetic", "ihdp"):
            for seed in range(42, 52):
                dd = _build_dataset(OmegaConf.create({"name": ds}), seed, None)
                _, ev = train_eval_split(dd, eval_frac=0.5, seed=seed)
                vbase[(ds, seed)] = float(dd.subset(ev).mu0.mean())
        fd = pd.read_parquet(fr)
        fd = fd[(fd.estimator != "perturbation_dr")
                & fd.dataset.isin(["synthetic", "ihdp"])].copy()
        fd["v0"] = fd.apply(lambda r: vbase[(r.dataset, r.seed)], axis=1)
        CFGi = ["dataset", "candidate_policy", "overlap_temperature", "budget_k"]
        gi = fd.groupby(CFGi + ["estimator"])
        ri = gi.apply(lambda x: np.sqrt(((x.value_hat - x.true_value) ** 2).mean())
                      / max(abs((x.true_value - x.v0).mean()), 1e-12),
                      include_groups=False)
        mi = (ri.rename("e").reset_index()
              .pivot_table(index="estimator", columns="dataset", values="e",
                           aggfunc="median"))
        out.append(("incr synthetic DM/DR",
                    f"${mi.loc['dm', 'synthetic']:.3f}$ vs.\\ ${mi.loc['dr', 'synthetic']:.3f}$"))
        lo_mb = min(mi.loc[e, d] for e in ("dm", "dr") for d in mi.columns)
        hi_mb = max(mi.loc[e, d] for e in ("dm", "dr") for d in mi.columns)
        lo_w = min(mi.loc[e, d] for e in ("ips", "bips") for d in mi.columns)
        hi_w = max(mi.loc[e, d] for e in ("ips", "bips") for d in mi.columns)
        out.append(("incr DM/DR range", f"${lo_mb:.3f}$--${hi_mb:.3f}$"))
        out.append(("incr IPS-family range", f"${lo_w:.3f}$--${hi_w:.3f}$"))
        for est, tex in (("dm", "0.035"), ("ips", "0.142")):
            assert f"{mi.loc[est, 'ihdp']:.3f}" == tex, (est, mi.loc[est, 'ihdp'])

    # The two numbers that survived three review rounds because nothing asserted
    # them: the scoped nuisance-cross-fit win rates and the RQ4 cluster intervals.
    nc = ROOT / "results/nuisance_crossfit/nuisance_crossfit.parquet"
    if nc.exists():
        n = pd.read_parquet(nc)
        # IPS uses no outcome model, so it is stored once (in the in-sample block) and
        # is numerically identical under out-of-fold nuisances -- the paper says so.
        # The comparison is therefore OOF model-based vs that single IPS column.
        IX = ["dataset", "seed", "candidate_policy", "overlap_temperature", "budget_k"]
        mb = n[n.nuisance == "out_of_fold"]
        ips = n[(n.nuisance == "in_sample") & (n.estimator == "ips")]
        CFG = ["dataset", "candidate_policy", "overlap_temperature", "budget_k"]

        def _relrmse(df):
            # the paper's convention: RMSE over seeds / |mean true value|, per config
            g = df.groupby(CFG + ["estimator"])
            return g.apply(
                lambda x: np.sqrt(((x.value_hat - x.true_value) ** 2).mean())
                / max(abs(x.true_value.mean()), 1e-12), include_groups=False
            ).rename("e").reset_index().pivot_table(
                index=CFG, columns="estimator", values="e")

        cfg = _relrmse(mb)
        cfg["ips"] = _relrmse(ips)["ips"]
        cfg = cfg.dropna()
        out.append(("nuis-xfit n configs", f"${len(cfg)}$ configurations"))
        # The two manuscripts phrase this differently (the compact gives a range, the
        # full version gives DM and the others separately), so assert the computed
        # values rather than one shared string -- drift still fails the build.
        rates = {e: (cfg[e] < cfg["ips"]).mean() * 100
                 for e in ("dm", "dr", "switch_dr") if e in cfg}
        assert round(rates["dm"]) == 89, f"DM win rate moved to {rates['dm']:.1f}"
        assert all(round(v) == 100 for k, v in rates.items() if k != "dm"), rates
        out.append(("nuis-xfit DM rate", f"${round(rates['dm'])}\\%$"))

    sel = ROOT / "results/full_run/selection.parquet"
    if sel.exists():
        d = pd.read_parquet(sel)
        d = d[(d.budget_k < 1.0) & (d.n_candidates == 3)]
        wv = d.pivot_table(index=["dataset", "seed", "budget_k", "overlap_temperature"],
                           columns="estimator", values="correct_selection", aggfunc="first")
        g = (wv["dm"].astype(float) - wv["ips"].astype(float)).dropna()
        out.append(("RQ4 per-cand gap", f"$+{g.mean():.3f}$"))
        # the paper's claim is that dataset-level clustering excludes zero NOWHERE on
        # this slate; assert that rather than a printed digit, since it is the claim.
        ix = g.index.to_frame()
        ks = ix.dataset.unique()
        cl = {k: g[ix.dataset == k].values for k in ks}
        rng = np.random.default_rng(0)
        bt = [np.concatenate([cl[p] for p in rng.choice(ks, len(ks), True)]).mean()
              for _ in range(4000)]
        assert np.percentile(bt, 2.5) <= 0, "RQ4 dataset-level CI now excludes zero"

    # Appendix: Twins external validation on OBSERVED ground truth. These are the
    # only figures in the paper not scored against a surface we fit, so they get
    # the same drift protection as the rest.
    tw = ROOT / "results/twins_run/estimates.parquet"
    if tw.exists():
        t = pd.read_parquet(tw)
        t = t[t.estimator != "perturbation_dr"]
        g = t.groupby(["dataset", "candidate_policy", "overlap_temperature",
                       "budget_k", "estimator"])
        rr = g.apply(lambda x: np.sqrt(((x.value_hat - x.true_value) ** 2).mean())
                     / max(abs(x.true_value.mean()), 1e-12), include_groups=False)
        med = rr.rename("e").reset_index().groupby("estimator").e.median()
        fam = {"dm": "DM", "dr": "DR", "switch_dr": "DR",
               "snips": "IPS", "ips": "IPS", "bips": "IPS"}
        fm = med.groupby(med.index.map(fam)).median()
        out.append(("twins DM", f"${med['dm']:.4f}$"))
        out.append(("twins DR family", f"${fm['DR']:.4f}$"))
        out.append(("twins SNIPS", f"${med['snips']:.4f}$"))
        out.append(("twins IPS family", f"${fm['IPS']:.4f}$"))
        out.append(("twins model-based margin", f"${fm['IPS'] / fm['DM']:.1f}\\times$"))

    ta = ROOT / "results/logger_alignment_twins/estimates.parquet"
    if ta.exists():
        a = pd.read_parquet(ta)
        ai = a[a.estimator == "ips"].copy()
        ai["ab"] = ai.rel_bias.abs()
        # the "never reaches the 10% criterion" claim, asserted not asserted-about
        assert ai.ab.max() < 0.10, f"twins now reaches the failure threshold: {ai.ab.max()}"
        out.append(("twins max |rel bias|", f"${ai.ab.max():.3f}$"))
        am = ai.groupby(K).agg(e=("ab", "median"), ess=("diag_ess_fraction", "mean"),
                               sd=("diag_support_deficiency", "mean"),
                               mw=("diag_max_weight", "mean")).reset_index()
        sa = am[am.logger_regime == "self_aligned"].groupby("overlap_temperature").ess.median()
        out.append(("twins self-aligned ESS",
                    f"${sa[0.5]:.3f}$, ${sa[2.0]:.3f}$, ${sa[5.0]:.3f}$"))
        for rg, lab in (("misaligned", "misal"), ("independent", "indep")):
            s = am[(am.logger_regime == rg) & (am.overlap_temperature == 0.5)]
            out.append((f"twins {lab} ESS tau0.5", f"${s.ess.median():.3f}$"))
        out.append(("twins rho triple",
                    f"$-{abs(spearmanr(am.ess, am.e).statistic):.2f}$ / "
                    f"$+{spearmanr(am.sd, am.e).statistic:.2f}$ / "
                    f"$+{spearmanr(am.mw, am.e).statistic:.2f}$"))

    to = ROOT / "results/twins_optbias/optimization_bias.parquet"
    if to.exists():
        o = pd.read_parquet(to)
        gg = o.groupby("estimator").agg(b=("bias", "mean"), ab=("abs_bias", "mean"),
                                        tv=("true_value", "mean"))
        base = gg.loc["dr", "ab"]
        out.append(("twins DR optimism",
                    f"${gg.loc['dr', 'b'] / abs(gg.loc['dr', 'tv']) * 100:.1f}\\%$"))
        out.append(("twins nuisance x-fit",
                    f"${(base - gg.loc['cross_fitted_dr', 'ab']) / base * 100:.1f}\\%$"))
        out.append(("twins honest split",
                    f"${(base - gg.loc['cross_fitted_dr_algo', 'ab']) / base * 100:.1f}\\%$"))
    return out


BANNED = [
    # Superseded 5-seed values. Presence anywhere in either manuscript is a failure:
    # the harness must catch stale prose coexisting with a corrected table.
    ("ladder rho independent (5-seed)", "$-0.55$"),
    ("incremental prose (5-seed, lost session)", "$0.149$ vs.\\ $0.151$"),
    ("ladder unsafe independent (5-seed)", "$17\\%$ of cells unsafe"),
    ("ladder rho self (5-seed)", "$-0.25$ self-aligned"),
    ("ladder rho misaligned (5-seed)", "$-0.08$ misaligned"),
    ("regime x tau (5-seed)", "median $-0.00$"),
    ("within-dataset range (5-seed)", "$-0.41$ to $-0.70$"),
    ("budget AUC range (5-seed)", "$0.84$--$0.96$"),
    ("hardening rho (5-seed)", "$-0.44$ / $+0.32$"),
    ("median logged n (5-seed)", "$2{,}063$"),
    ("config count (5-seed)", "$1{,}080$"),
]


def counts():
    """Assert every printed configuration count against its parquet."""
    spec = [
        ("results/acic_run/estimates.parquet", "$2{,}160$"),
        ("results/acic_hillstrom_run/estimates.parquet", "$2{,}160$"),
        ("results/logger_alignment_acic/estimates.parquet", "$4{,}320$"),
        ("results/logger_alignment_acic_hillstrom/estimates.parquet", "$4{,}320$"),
        ("results/full_run/estimates.parquet", "$2{,}700$"),
    ]
    out = []
    for rel, printed in spec:
        p = ROOT / rel
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        n = len(d) // d.estimator.nunique()
        s = f"${n:,}$".replace(",", "{,}")
        out.append((f"configs {Path(rel).parent.name}", s, printed))
    return out


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    srcs = {p: p.read_text() for p in TEX}
    bad = []
    for name, computed, _ in counts():
        missing = [str(p.relative_to(ROOT)) for p, txt in srcs.items() if computed not in txt]
        flag = "ok  " if not missing else "FAIL"
        print(f"  {flag} {name:28s} {computed}"
              + (f"   missing from: {missing}" if missing else ""))
        if missing:
            bad.append((name, computed, missing))
    for name, s in checks():
        missing = [str(p.relative_to(ROOT)) for p, txt in srcs.items() if s not in txt]
        flag = "ok  " if not missing else "FAIL"
        print(f"  {flag} {name:28s} {s}" + (f"   missing from: {missing}" if missing else ""))
        if missing:
            bad.append((name, s, missing))
    for name, s in BANNED:
        present = [str(p.relative_to(ROOT)) for p, txt in srcs.items() if s in txt]
        flag = "ok  " if not present else "FAIL"
        print(f"  {flag} banned {name:21s} {s}" + (f"   STILL PRESENT in {present}" if present else ""))
        if present:
            bad.append((name, s, present))
    # Numbers that appear in one manuscript but not the other are usually a value that
    # was corrected in one file and not the other -- the failure mode that produced the
    # 25%-vs-26% clipping disagreement.
    import re
    a, b = (re.findall(r"\$[-+]?\d+\.\d+\$", s) for s in srcs.values())
    only = (set(a) ^ set(b)) - {"$0.62$", "$0.82$", "$0.92$", "$0.85$", "$0.70$", "$0.78$"}
    if only:
        print(f"\n  note  {len(only)} decimals appear in only one manuscript (expected: the"
              f" full version carries detail the compact omits; scan if a value was"
              f" corrected in one file only)")

    if bad:
        print(f"\nFAILED: {len(bad)} value(s) stale, missing or superseded.")
        return 1
    print("\nAll recomputed values found in both manuscripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""WP6 — LaTeX (booktabs) tables for the paper, from the aggregated frames.

Each ``table_*`` returns a LaTeX string and (optionally) writes a .tex file. Tables
mirror the figures: RQ1 accuracy, RQ3 optimization de-biasing, RQ4 selection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from allocation_ope_bench.analysis import aggregate as agg

# Display relabelling: "bips" is a mixture-propensity IPS, not a covariate-balancing
# method — show it under the clearer name to avoid confusion (review #7).
_DISPLAY = {"bips": "mIPS"}


def _disp(est: str) -> str:
    return _DISPLAY.get(est, est).replace("_", r"\_")


def _fmt(x: float, nd: int = 3) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.{nd}f}"


def _write(tex: str, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.tex"
    p.write_text(tex)
    return p


def table_rq1_accuracy(estimates: pd.DataFrame, out_dir: Path | None = None) -> str:
    """Relative-RMSE table (rows=estimator, cols=dataset); best per column bolded."""
    tab = agg.rq1_accuracy_table(estimates)
    mean_cols = [c for c in ("mean_exact", "mean_rct") if c in tab.columns]
    datasets = [c for c in tab.columns if c not in mean_cols]

    mean_hdr = {"mean_exact": "mean (exact)", "mean_rct": "mean (RCT-ref)"}
    header = " & ".join(["estimator", *datasets, *[mean_hdr[c] for c in mean_cols]]) + r" \\"
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{RQ1 estimator accuracy: median relative RMSE "
        r"(RMSE$/|V_{\mathrm{ref}}|$, normalized by the configuration-level reference "
        r"value --- exact where known, else the HT reference estimate), lower is "
        r"better. Best per column in bold. The cross-dataset averages are reported "
        r"\emph{separately} for exact-value and HT-reference datasets rather than "
        r"pooled, since the two measure error against different kinds of reference.}",
        r"\label{tab:rq1-accuracy}",
        r"\begin{tabular}{l" + "r" * (len(datasets) + len(mean_cols)) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for est in tab.index:
        cells = []
        for col in [*datasets, *mean_cols]:
            v = _fmt(tab.loc[est, col])
            # Bold every estimator tied for the column minimum at displayed
            # precision, not only the first (idxmin) — otherwise a tie looks like
            # a strict win.
            if v == _fmt(tab[col].min()):
                v = r"\textbf{" + v + "}"
            cells.append(v)
        lines.append(" & ".join([_disp(est), *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    if out_dir is not None:
        _write(tex, out_dir, "tab_rq1_accuracy")
    return tex


def table_acic_hardening(estimates: pd.DataFrame, out_dir: Path | None = None) -> str:
    """Known-effect hardening: rel-RMSE per estimator across the six
    ACIC-2017-style DGP settings on real IHDP covariates (S1..S6 columns)."""
    tab = agg.rq1_accuracy_table(estimates)
    mean_cols = [c for c in ("mean_exact", "mean_rct") if c in tab.columns]
    datasets = [c for c in tab.columns if c not in mean_cols]
    pretty = {ds: ds.replace("acic_s", "S") for ds in datasets}

    mean_hdr = {"mean_exact": "mean (exact)", "mean_rct": "mean (RCT-ref)"}
    header = (
        " & ".join(
            ["estimator", *[pretty[ds] for ds in datasets], *[mean_hdr[c] for c in mean_cols]]
        )
        + r" \\"
    )
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Known-effect hardening: median relative RMSE on six "
        r"ACIC-2017-style DGP settings over the real \dataset{IHDP} covariates "
        r"(S1/S2 linear, S3/S4 nonlinear, S5/S6 step-subgroup surfaces; odd $=$ low "
        r"noise, even $=$ high). Best per column in bold.}",
        r"\label{tab:acic-hardening}",
        r"\begin{tabular}{l" + "r" * (len(datasets) + len(mean_cols)) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for est in tab.index:
        cells = []
        for col in [*datasets, *mean_cols]:
            v = _fmt(tab.loc[est, col])
            # Bold every estimator tied for the column minimum at displayed
            # precision, not only the first (idxmin) — otherwise a tie looks like
            # a strict win.
            if v == _fmt(tab[col].min()):
                v = r"\textbf{" + v + "}"
            cells.append(v)
        lines.append(" & ".join([_disp(est), *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    if out_dir is not None:
        _write(tex, out_dir, "tab_acic_hardening")
    return tex


def table_rq3_debias(
    optbias: pd.DataFrame, out_dir: Path | None = None, acic_optbias: pd.DataFrame | None = None
) -> str:
    """Per-dataset optimizer's-curse de-biasing table (signed bias + % removed)."""
    summary = agg.rq3_debias_summary(optbias)
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{RQ3 optimizer's curse, per dataset. Plain-DR $|$bias$|$ (in-sample "
        r"policy value) and the $|$bias$|$ change from nuisance-only cross-fitting "
        r"(same fixed-policy estimand as DR); and the bias reduction under honest "
        r"\emph{algorithm} evaluation (a different estimand --- the learning "
        r"procedure's value). ``material optimistic bias'' is an explicit criterion: plain "
        r"DR's mean \emph{signed} bias exceeds $2\%$ of $|V_{\mathrm{ref}}|$, i.e.\ DR is "
        r"materially \emph{optimistic} (third column). It is signed, not absolute, on purpose --- "
        r"\dataset{Jobs} ($-0.2\%$) and \dataset{Lenta} ($-3.8\%$) carry non-trivial "
        r"$|$bias$|$ but are not optimistic, so there is no upward reuse bias to remove; "
        r"\dataset{IHDP} ($+6.9\%$) and \dataset{synthetic} ($+23.6\%$) are flagged. "
        r"DR $|$bias$|$ is shown both "
        r"raw and as a fraction of $|V_{\mathrm{ref}}|$, since raw magnitudes are not "
        r"comparable across datasets with different outcome scales.}",
        r"\label{tab:rq3-debias}",
        r"\small",
        r"\begin{tabular}{lcrrrrr}",
        r"\toprule",
        r"dataset & material optimistic & DR bias$/|V|$ & DR $|$bias$|$ & DR $|$bias$|/|V|$ "
        r"& nuis.\ x-fit & honest-algo. \\",
        r" & bias? & (signed) & & & \% removed & \% removed \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            " & ".join(
                [
                    r["dataset"].replace("_", r"\_"),
                    "yes" if r["curse_present"] else "no",
                    _fmt(r.get("dr_signed_rel_bias", float("nan"))),
                    _fmt(r["dr_abs_bias"]),
                    _fmt(r.get("dr_rel_bias", float("nan"))),
                    _fmt(r.get("cross_fitted_dr_pct_removed", float("nan")), 1),
                    _fmt(r.get("cross_fitted_dr_algo_pct_removed", float("nan")), 1),
                ]
            )
            + r" \\"
        )
    if acic_optbias is not None:
        # The six ACIC-style known-effect DGPs supply the 58% floor of the headline
        # 58-92% range; without them the range has no displayed home.
        lines.append(r"\midrule")
        lines.append(
            r"\multicolumn{7}{l}{\emph{six ACIC-style known-effect DGPs on real "
            r"\dataset{IHDP} covariates (Section~\ref{sec:robustness}):}} \\"
        )
        for _, r in agg.rq3_debias_summary(acic_optbias).iterrows():
            lines.append(
                " & ".join(
                    [
                        r["dataset"].replace("_", r"\_"),
                        "yes" if r["curse_present"] else "no",
                        _fmt(r.get("dr_signed_rel_bias", float("nan"))),
                        _fmt(r["dr_abs_bias"]),
                        _fmt(r.get("dr_rel_bias", float("nan"))),
                        _fmt(r.get("cross_fitted_dr_pct_removed", float("nan")), 1),
                        _fmt(r.get("cross_fitted_dr_algo_pct_removed", float("nan")), 1),
                    ]
                )
                + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    if out_dir is not None:
        _write(tex, out_dir, "tab_rq3_debias")
    return tex


def table_rq4_selection(selection: pd.DataFrame, out_dir: Path | None = None) -> str:
    """Selection-quality table: correct rate, normalized regret, SharpeRatio@k>=2."""
    tab = agg.rq4_selection_table(selection)
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{RQ4 policy selection: rate of selecting the reference-best policy "
        r"(the truly best on exact-value datasets, the HT-reference-best on RCT "
        r"datasets; higher better), mean normalized regret (lower better), and "
        r"SharpeRatio@$k{\geq}2$ (higher better). \textbf{This table uses the "
        r"per-candidate (self-aligned) logging design}, in which each candidate is "
        r"scored on a log built from its own score, so it compares policy--logger pairs "
        r"rather than policies. Section~\ref{sec:rq4} shows the model-based-over-IPS "
        r"margin is nonetheless close to what a shared log gives (about $1.2\times$); "
        r"see Table~\ref{tab:rq4-logger} for the common-logger results, which we regard "
        r"as the primary selection evidence.}",
        r"\label{tab:rq4-selection}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"estimator & correct rate & norm. regret & Sharpe@$k{\geq}2$ \\",
        r"\midrule",
    ]
    for _, r in tab.iterrows():
        lines.append(
            " & ".join(
                [
                    _disp(r["estimator"]),
                    _fmt(r["correct_rate"]),
                    _fmt(r["mean_regret_norm"]),
                    _fmt(r["mean_sharpe_k2plus"]),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    if out_dir is not None:
        _write(tex, out_dir, "tab_rq4_selection")
    return tex


def table_rq4_by_reftype(selection: pd.DataFrame, out_dir: Path | None = None) -> str:
    """Correct-selection rate split by exact-value vs RCT-reference datasets."""
    tab = agg.rq4_selection_by_reference_type(selection)
    for col in ("exact", "rct"):
        if col not in tab.columns:
            tab[col] = float("nan")
    tab = tab.reindex([e for e in agg.PRIMARY_ESTIMATORS if e in tab.index])
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{RQ4 selection quality split by reference type: rate of selecting "
        r"the best candidate. On exact-value datasets (\dataset{synthetic}, "
        r"\dataset{IHDP}) this is the \emph{truly} best policy; on RCT-reference "
        r"datasets (\dataset{Jobs}, \dataset{Hillstrom}, \dataset{Lenta}) it is the "
        r"policy ranked best by the noisy HT reference. Three-candidate slate; "
        r"uniform-random baseline $=1/3\approx0.33$. \textbf{Per-candidate "
        r"(self-aligned) logging design} --- see Table~\ref{tab:rq4-logger}.}",
        r"\label{tab:rq4-reftype}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"estimator & exact-value (truly best) & RCT-reference (reference-best) \\",
        r"\midrule",
    ]
    for est in tab.index:
        lines.append(
            f"{_disp(est)} & {_fmt(tab.loc[est, 'exact'])} & {_fmt(tab.loc[est, 'rct'])}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    if out_dir is not None:
        _write(tex, out_dir, "tab_rq4_reftype")
    return tex


def make_all_tables(results: dict, out_dir: Path) -> list[Path]:
    paths = [_write(table_rq1_accuracy(results["estimates"]), out_dir, "tab_rq1_accuracy")]
    if "optimization_bias" in results:
        paths.append(
            _write(table_rq3_debias(results["optimization_bias"]), out_dir, "tab_rq3_debias")
        )
    if "selection" in results:
        paths.append(
            _write(table_rq4_selection(results["selection"]), out_dir, "tab_rq4_selection")
        )
        paths.append(_write(table_rq4_by_reftype(results["selection"]), out_dir, "tab_rq4_reftype"))
    return paths

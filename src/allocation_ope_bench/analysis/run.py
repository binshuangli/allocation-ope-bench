"""WP6 — analysis CLI: read result parquets, write all figures + tables + a digest.

Usage
-----
    python -m allocation_ope_bench.analysis.run --results-dir results/full_run
    # writes:
    #   <results-dir>/figures/*.pdf,*.png
    #   <results-dir>/tables/*.tex
    #   <results-dir>/analysis_digest.md   (human-readable headline summary)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from allocation_ope_bench.analysis import aggregate as agg
from allocation_ope_bench.analysis import figures, tables


def _df_to_md(df: pd.DataFrame, index: bool = True) -> str:
    """Minimal GitHub-flavored markdown table (avoids the optional `tabulate` dep)."""
    df = df.copy()
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(out)


def write_digest(results: dict, out_path: Path) -> Path:
    """A short markdown digest of the headline numbers (sanity check + paper notes)."""
    est = results["estimates"]
    lines = ["# Analysis digest\n"]

    lines.append("## RQ1 — accuracy (median relative RMSE)\n")
    acc = agg.rq1_accuracy_table(est)
    lines.append(_df_to_md(acc.round(3)))
    _mc = "mean_exact" if "mean_exact" in acc.columns else "mean_rct"
    best = acc[_mc].idxmin()
    lines.append(f"\nBest mean-across-datasets accuracy: **{best}**\n")

    lines.append("\n## RQ2 — trust: do logged-data diagnostics predict IPS error?\n")
    corr = agg.rq2_diagnostic_correlation(est, estimator="ips")
    lines.append(_df_to_md(corr.round(3), index=False))
    lines.append(
        "\nSpearman ρ of each diagnostic vs IPS |relative bias|. Expect ρ<0 for ESS "
        "fraction (more support → less error) and ρ>0 for support deficiency / max "
        "weight (more risk → more error).\n"
    )

    if "optimization_bias" in results:
        lines.append("\n## RQ3 — optimizer's curse (per dataset)\n")
        deb = agg.rq3_debias_summary(results["optimization_bias"])
        lines.append(_df_to_md(deb.round(2), index=False))
        curse_ds = deb[deb.curse_present]["dataset"].tolist()
        lines.append(
            f"\nCurse present on: **{curse_ds or 'none'}** "
            "(continuous / known-effect datasets). Pooling across all "
            "datasets understates de-biasing — report per dataset.\n"
        )

    if "selection" in results:
        lines.append("\n## RQ4 — selection quality\n")
        sel = agg.rq4_selection_table(results["selection"])
        lines.append(_df_to_md(sel.round(3), index=False))

    out_path.write_text("\n".join(lines))
    return out_path


def write_hardening_digest(results: dict, out_path: Path) -> Path:
    """Digest for the ACIC known-effect hardening run (accuracy + diagnostics)."""
    est = results["estimates"]
    lines = ["# ACIC hardening digest\n"]
    lines.append("## Accuracy (median relative RMSE per DGP setting)\n")
    acc = agg.rq1_accuracy_table(est)
    lines.append(_df_to_md(acc.round(3)))
    _mc = "mean_exact" if "mean_exact" in acc.columns else "mean_rct"
    best = acc[_mc].idxmin()
    lines.append(f"\nBest mean-across-settings accuracy: **{best}**\n")
    lines.append(f"\nMean |true value| across cells: {est['true_value'].abs().mean():.4f}\n")

    lines.append("\n## Diagnostics vs IPS error (Spearman)\n")
    corr = agg.rq2_diagnostic_correlation(est, estimator="ips")
    lines.append(_df_to_md(corr.round(3), index=False))
    out_path.write_text("\n".join(lines))
    return out_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="WP6 analysis: figures + tables + digest")
    ap.add_argument("--results-dir", default="results/full_run", help="dir with parquets")
    ap.add_argument("--out-dir", default=None, help="output root (default: results-dir)")
    ap.add_argument(
        "--hardening",
        action="store_true",
        help="ACIC known-effect hardening mode: write tab_acic_hardening + digest only",
    )
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    results = agg.load_results(results_dir)

    if args.hardening:
        tables.table_acic_hardening(results["estimates"], out_dir / "tables")
        digest = write_hardening_digest(results, out_dir / "analysis_digest.md")
        print(f"Wrote tab_acic_hardening.tex -> {out_dir / 'tables'}")
        print(f"Wrote digest -> {digest}")
        return

    fig_paths = figures.make_all_figures(results, out_dir / "figures")
    tab_paths = tables.make_all_tables(results, out_dir / "tables")
    digest = write_digest(results, out_dir / "analysis_digest.md")

    print(f"Wrote {len(fig_paths)} figures -> {out_dir / 'figures'}")
    for p in fig_paths:
        print(f"  {p.name}")
    print(f"Wrote {len(tab_paths)} tables  -> {out_dir / 'tables'}")
    print(f"Wrote digest -> {digest}")


if __name__ == "__main__":
    main()

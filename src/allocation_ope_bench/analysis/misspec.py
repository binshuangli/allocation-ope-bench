"""Outcome-model misspecification analysis (review item 5).

Combines the three `repro-misspec` sub-runs (lightgbm / stump / ridge shared
mu-hat) into the two-dimensional decision map the RQ1 caveat promises:
estimator error as a joint function of logging overlap and outcome-model
quality. Outputs a digest and a LaTeX table:

    python -m allocation_ope_bench.analysis.misspec \
        --results-root results/misspec_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from allocation_ope_bench.analysis import aggregate as agg

MODELS = ["lightgbm", "stump", "ridge"]  # strong -> weak -> misspecified
MODEL_LABEL = {
    "lightgbm": "strong (LightGBM)",
    "stump": "weak (stumps)",
    "ridge": "linear (missp.)",
}
TEMP_LABEL = {0.5: "poor", 2.0: "moderate", 5.0: "good"}


def load_runs(root: Path) -> pd.DataFrame:
    frames = []
    for m in MODELS:
        p = root / m / "estimates.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["outcome_model"] = m  # present already; overwrite defensively
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No estimates under {root}/<model>/")
    return pd.concat(frames, ignore_index=True)


SHOWN = ("dm", "dr", "snips", "ips")
DGP_LABEL = {
    "synthetic": "synthetic (nonlin.)",
    "acic_s3": "ACIC nonlin.",
    "acic_s5": "ACIC step",
}


def map_table(est: pd.DataFrame) -> pd.DataFrame:
    """Median rel-RMSE per (dataset, outcome_model, estimator), pooled over
    overlap/budgets/policies; `best` ranges over the shown estimators only.

    Per-DGP (not pooled across datasets): pooling averages away the synthetic
    inversion the text describes."""
    rows = []
    for (ds, m), g in est.groupby(["dataset", "outcome_model"]):
        cells = agg.rq1_cell_rel_rmse(g)
        med = cells.groupby("estimator")["rel_rmse"].median()
        shown = {e: float(med[e]) for e in SHOWN if e in med.index}
        rec = {
            "dataset": ds,
            "outcome_model": m,
            "best": min(shown, key=shown.get) if shown else "--",
            **shown,
        }
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["model_rank"] = out["outcome_model"].map({m: i for i, m in enumerate(MODELS)})
    out["ds_rank"] = out["dataset"].map({d: i for i, d in enumerate(DGP_LABEL)})
    return out.sort_values(["ds_rank", "model_rank"]).drop(columns=["model_rank", "ds_rank"])


def latex_table(tab: pd.DataFrame, out_dir: Path | None = None) -> str:
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Outcome-model misspecification, \emph{per DGP}: median relative RMSE "
        r"by DGP and outcome-model quality (pooled over overlap and budgets). ``best'' "
        r"ranges over the four shown estimators. The inversion is DGP-specific: on the "
        r"nonlinear \dataset{synthetic} surface a degraded $\hat\mu$ makes \est{DM} the "
        r"worst of the four; on the smoother ACIC surfaces \est{DM} stays competitive.}",
        r"\label{tab:misspec}",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"DGP & $\hat\mu$ quality & DM & DR & SNIPS & IPS & best \\",
        r"\midrule",
    ]
    prev = None
    for _, r in tab.iterrows():
        dlabel = DGP_LABEL.get(r["dataset"], r["dataset"])
        show = dlabel if dlabel != prev else ""
        if dlabel != prev and prev is not None:
            lines.append(r"\midrule")
        prev = dlabel
        lines.append(
            f"{show} & {MODEL_LABEL[r['outcome_model']]} & "
            f"{r['dm']:.3f} & {r['dr']:.3f} & {r['snips']:.3f} & {r['ips']:.3f} & "
            + str(r["best"]).replace("_", r"\_")
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tab_misspec.tex").write_text(tex)
    return tex


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/misspec_run")
    args = ap.parse_args(argv)
    root = Path(args.results_root)

    est = load_runs(root)
    tab = map_table(est)
    latex_table(tab, root / "tables")
    with pd.option_context("display.width", 120):
        print(tab.round(3).to_string(index=False))
    print(f"\nWrote {root / 'tables' / 'tab_misspec.tex'}")


if __name__ == "__main__":
    main()

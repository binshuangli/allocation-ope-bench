"""WP6 — publication figures from the aggregated analysis tables.

Each ``make_*`` function takes the raw result frames, builds one figure, saves it
as both PDF (vector, for the paper) and PNG (preview) into ``fig_dir``, and returns
the saved Path(s). Matplotlib only (no seaborn dependency); a single consistent
style is applied via :func:`set_style`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from allocation_ope_bench.analysis import aggregate as agg  # noqa: E402

# Consistent estimator colors across all figures.
_PALETTE = {
    "dm": "#4C72B0",
    "dr": "#55A868",
    "switch_dr": "#8172B3",
    "perturbation_dr": "#CCB974",
    "snips": "#64B5CD",
    "ips": "#C44E52",
    "bips": "#E377C2",
    "cross_fitted_dr": "#937860",
    "cross_fitted_dr_algo": "#DA8BC3",
}


def set_style() -> None:
    # Multi-panel figures are downscaled to \linewidth in the paper (~40-50%),
    # so fonts are set large here to remain legible after scaling.
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            # Embed TrueType (42) rather than Type 3 fonts: Type 3 is not
            # searchable/selectable and some venues reject it.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _save(fig, fig_dir: Path, name: str) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf = fig_dir / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    return pdf


def _color(est: str) -> str:
    return _PALETTE.get(est, "#888888")


# "bips" is a mixture-propensity IPS (not covariate balancing); show as mIPS (review #7).
_DISPLAY = {
    "bips": "mIPS",
    "dm": "DM",
    "dr": "DR",
    "switch_dr": "Switch-DR",
    "snips": "SNIPS",
    "ips": "IPS",
    "perturbation_dr": "Perturbation-DR",
    "cross_fitted_dr": "Frozen-policy nuisance cross-fit",
    "cross_fitted_dr_algo": "Honest fold-policy evaluation",
}


def _disp(est: str) -> str:
    return _DISPLAY.get(est, est)


def _displist(ests) -> list[str]:
    return [_disp(e) for e in ests]


# ── Figure 1 — RQ1 accuracy (relative RMSE per dataset) ─────────────────────────


def fig_rq1_accuracy(estimates, fig_dir: Path) -> Path:
    """Grouped bars: median relative RMSE by estimator, one panel per dataset."""
    tab = agg.rq1_accuracy_table(estimates)  # index=estimator, cols=datasets(+mean)
    datasets = [c for c in tab.columns if c not in ("mean_exact", "mean_rct")]
    estimators = list(tab.index)

    n = len(datasets)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)

    # Bootstrap the median over cells so the bars carry uncertainty rather than
    # implying the point estimates are exact.
    cells = agg.rq1_cell_rel_rmse(estimates)
    rng = np.random.default_rng(0)
    err = {}
    for ds in datasets:
        lo_hi = []
        for e in estimators:
            x = cells[(cells.dataset == ds) & (cells.estimator == e)]["rel_rmse"].to_numpy()
            if x.size == 0:
                lo_hi.append((0.0, 0.0))
                continue
            bt = np.array([np.median(rng.choice(x, x.size, True)) for _ in range(2000)])
            med = np.median(x)
            lo, hi = np.percentile(bt, [2.5, 97.5])
            lo_hi.append((max(med - lo, 0.0), max(hi - med, 0.0)))
        err[ds] = np.array(lo_hi).T

    for i, ds in enumerate(datasets):
        ax = axes[i // ncol][i % ncol]
        vals = tab[ds].values
        colors = [_color(e) for e in estimators]
        ax.bar(
            range(len(estimators)),
            vals,
            color=colors,
            yerr=err[ds],
            capsize=3,
            error_kw={"elinewidth": 1.0, "ecolor": "0.25"},
        )
        ax.set_title(ds)
        ax.set_xticks(range(len(estimators)))
        ax.set_xticklabels(_displist(estimators), rotation=45, ha="right", fontsize=12)
        if i % ncol == 0:
            ax.set_ylabel("median relative RMSE")
    # Hide any unused panels.
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(
        "RQ1 — estimator accuracy (median relative RMSE, lower is better;\n"
        "bars show 95% bootstrap intervals for the median over cells)",
        y=1.04,
    )
    fig.tight_layout()
    return _save(fig, fig_dir, "fig1_rq1_accuracy")


# ── Figure 2 — RQ2 trust: error vs overlap stress ───────────────────────────────


def fig_rq2_overlap(estimates, fig_dir: Path) -> Path:
    """Relative RMSE vs overlap temperature (low temp = poor overlap), per dataset."""
    ev = agg.rq2_error_vs_overlap(estimates)
    datasets = sorted(ev["dataset"].unique())
    estimators = [e for e in agg.ESTIMATOR_ORDER if e in ev["estimator"].unique()]

    n = len(datasets)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)

    for i, ds in enumerate(datasets):
        ax = axes[i // ncol][i % ncol]
        sub = ev[ev.dataset == ds]
        for est in estimators:
            s = sub[sub.estimator == est].sort_values("overlap_temperature")
            if s.empty:
                continue
            ax.plot(
                s["overlap_temperature"],
                s["rel_rmse"],
                marker="o",
                ms=6,
                label=_disp(est),
                color=_color(est),
            )
        ax.set_title(ds)
        if i // ncol == nrow - 1 or i + ncol >= n:
            ax.set_xlabel(r"logging temperature $\tau$ (low = poor overlap)")
        if i % ncol == 0:
            ax.set_ylabel("median relative RMSE")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("RQ2 — error vs logging overlap", y=1.10)
    fig.tight_layout()
    return _save(fig, fig_dir, "fig2_rq2_overlap")


def fig_rq2_trust_map(estimates, fig_dir: Path) -> Path:
    """Trust map: IPS |relative bias| vs ESS fraction and vs support deficiency."""
    d = agg.rq2_diagnostic_vs_error(estimates, estimator="ips")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    for ax, xcol, xlabel in (
        (axes[0], "diag_ess_fraction", "ESS fraction (higher = more support)"),
        (axes[1], "diag_support_deficiency", "support deficiency (higher = worse)"),
    ):
        for ds in sorted(d["dataset"].unique()):
            s = d[d.dataset == ds]
            ax.scatter(s[xcol], s["abs_rel_bias"], s=18, alpha=0.5, label=ds)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("IPS |relative bias|")
    axes[0].legend(fontsize=12, title="dataset")
    fig.suptitle("RQ2 — fragility map: logged-data diagnostics track IPS error", y=1.02)
    fig.tight_layout()
    return _save(fig, fig_dir, "fig3_rq2_trust_map")


# ── Figure 3 — RQ3 optimization bias (per dataset, conditioned on curse) ─────────


def fig_rq3_optimization_bias(optbias, fig_dir: Path) -> Path:
    """Per-dataset signed bias of DR / cross_fitted_dr / perturbation_dr.

    Datasets where the optimizer's curse is present (plain DR materially biased)
    are marked; cross_fitted_dr should collapse the bias there.
    """
    a = agg.rq3_optimization_bias(optbias)
    datasets = sorted(
        a["dataset"].unique(),
        key=lambda d: -(
            a[(a.dataset == d) & (a.estimator == "dr")]["mean_bias"].abs().iloc[0]
            / max(abs(a[(a.dataset == d) & (a.estimator == "dr")]["mean_true"].iloc[0]), 1e-8)
        ),
    )
    ests = [e for e in agg.OPTBIAS_ORDER if e in a["estimator"].unique()]

    x = np.arange(len(datasets))
    w = 0.8 / len(ests)
    fig, ax = plt.subplots(figsize=(1.6 * len(datasets) + 2, 4.0))
    for k, est in enumerate(ests):
        # Normalize by |V_ref|: outcome scale spans ~30x across datasets, so raw
        # signed bias on a shared axis is a scale artifact (Table rq3-debias makes
        # the same point and reports both).
        vals = [
            a[(a.dataset == ds) & (a.estimator == est)]["mean_bias"].iloc[0]
            / max(abs(a[(a.dataset == ds) & (a.estimator == "dr")]["mean_true"].iloc[0]), 1e-8)
            for ds in datasets
        ]
        ax.bar(x + k * w, vals, width=w, label=_disp(est), color=_color(est))

    ax.axhline(0, color="k", lw=0.8)
    # Annotate datasets with material detected bias.
    curse = a.groupby("dataset")["curse_present"].first()
    labels = [f"{ds}\n(bias)" if curse.get(ds) else ds for ds in datasets]
    ax.set_xticks(x + w * (len(ests) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"mean signed bias / $|V_{ref}|$")
    ax.set_title(
        "RQ3 — honest policy-level splitting, not nuisance cross-fitting, "
        "addresses optimization bias"
    )
    ax.legend()
    fig.tight_layout()
    return _save(fig, fig_dir, "fig4_rq3_optimization_bias")


# ── Figure 4 — RQ4 selection ─────────────────────────────────────────────────────


def fig_rq4_selection(selection, fig_dir: Path) -> Path:
    """Bars: correct-selection rate and mean normalized regret by estimator."""
    tab = agg.rq4_selection_table(selection)
    ests = list(tab["estimator"])
    colors = [_color(e) for e in ests]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].bar(range(len(ests)), tab["correct_rate"], color=colors)
    axes[0].set_title("(a) Reference-best selection rate")
    axes[0].set_xticks(range(len(ests)))
    axes[0].set_xticklabels(_displist(ests), rotation=45, ha="right", fontsize=12)
    axes[0].set_ylabel("P(selects reference-best)")

    axes[1].bar(range(len(ests)), tab["mean_regret_norm"], color=colors)
    axes[1].set_title("(b) Normalized regret")
    axes[1].set_xticks(range(len(ests)))
    axes[1].set_xticklabels(_displist(ests), rotation=45, ha="right", fontsize=12)
    axes[1].set_ylabel("normalized regret")

    fig.suptitle("RQ4 — policy selection quality", y=1.03)
    fig.tight_layout()
    return _save(fig, fig_dir, "fig5_rq4_selection")


def fig_rq4_logger(common3, common7, percand3, percand7, fig_dir: Path) -> Path:
    """PRIMARY RQ4 figure: selection quality under three logging designs.

    The per-candidate (self-aligned) design is what the main sweep used; it scores each
    candidate on a log built from its own score and so compares policy-logger pairs.
    The two common-logger designs score every candidate on one shared log. Showing all
    three makes the size of the design effect the visible message.
    """
    import numpy as np

    ests = ["dm", "dr", "ips"]
    designs = [
        ("per-candidate", percand3, percand7),
        (
            "common,\naligned",
            common3[common3.logger_regime == "aligned_t"],
            common7[common7.logger_regime == "aligned_t"],
        ),
        (
            "common,\nindependent",
            common3[common3.logger_regime == "independent"],
            common7[common7.logger_regime == "independent"],
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), sharex=True)
    width = 0.26
    x = np.arange(len(designs))
    for row, (slate, idx) in enumerate([("3-candidate slate", 1), ("7-policy slate", 2)]):
        for col, (metric, lab) in enumerate(
            [
                ("correct_selection", "P(selects reference-best)"),
                ("regret_normalized", "normalized regret"),
            ]
        ):
            ax = axes[row][col]
            for j, e in enumerate(ests):
                vals = [d[idx][d[idx].estimator == e][metric].mean() for d in designs]
                ax.bar(x + (j - 1) * width, vals, width, color=_color(e), label=_disp(e))
            ax.set_ylabel(lab)
            ax.set_title(f"{slate} — {'selection rate' if col == 0 else 'regret'}")
            if col == 0 and row == 0:
                ax.legend(ncol=3, loc="upper right")
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels([d[0] for d in designs])
    fig.suptitle("RQ4 — selection conclusions depend on the metric and slate more than on\nlogging design", y=1.03)
    fig.tight_layout()
    return _save(fig, fig_dir, "fig5_rq4_logger")


def make_all_figures(results: dict, fig_dir: Path) -> list[Path]:
    """Build every figure that the available frames support."""
    set_style()
    paths: list[Path] = []
    est = results["estimates"]
    paths.append(fig_rq1_accuracy(est, fig_dir))
    paths.append(fig_rq2_overlap(est, fig_dir))
    paths.append(fig_rq2_trust_map(est, fig_dir))
    if "optimization_bias" in results:
        paths.append(fig_rq3_optimization_bias(results["optimization_bias"], fig_dir))
    if "selection" in results:
        paths.append(fig_rq4_selection(results["selection"], fig_dir))
    return paths


REGIME_ORDER = ("self_aligned", "misaligned", "independent")
REGIME_LABEL = {
    "self_aligned": "self-aligned",
    "misaligned": "misaligned",
    "independent": "independent",
}


def fig_rq2_alignment(alignment, fig_dir: Path, alignment_cutoff=None) -> Path:
    """PRIMARY RQ2 figure: overlap risk comes from logger--target action alignment.

    Over the tested temperature range, sharpening a self-aligned (score-aligned) logger
    barely moves overlap -- a finite-range plateau, not a guarantee (see the
    sharpening_limit experiment for the collapse below this range) -- while action-level
    disagreement collapses it at any sharpness. Three panels: what the logger does to
    overlap (ESS), how big the weights get, and how often IPS then breaks.
    """
    import numpy as np

    ips = alignment[alignment.estimator == "ips"].copy()
    ips["abs_rel_bias"] = ips.rel_bias.abs()
    # The failure panel uses the CELL (seed-aggregated) unit, matching every failure
    # share quoted in the paper: a cell fails when its median-over-seeds |rel bias|
    # exceeds 10%. Plotting seed-level rows here once produced a second, incompatible
    # set of "failure rates" (11/19/47 vs the cell-level 8/10/50).
    cell_key = ["dataset", "candidate_policy", "logger_regime",
                "overlap_temperature", "budget_k"]
    cells = ips.groupby(cell_key).abs_rel_bias.median().reset_index()
    cells["unsafe"] = (cells.abs_rel_bias > 0.10).astype(float)
    temps = sorted(ips.overlap_temperature.unique())
    regimes = [r for r in REGIME_ORDER if r in set(ips.logger_regime)]

    panels = [
        ("diag_ess_fraction", "median ESS fraction", "median", ips),
        ("diag_max_weight", "median per-cell max weight", "median", ips),
        ("unsafe", r"IPS failure rate (cells)", "mean", cells),
    ]
    if alignment_cutoff is not None:
        # Same failure computation on the corrected (cutoff-centred) sweep, so the
        # corrected gradient is co-equal with the primary one in the figure itself,
        # not only in the caption and Sec 5.2.
        ipc = alignment_cutoff[alignment_cutoff.estimator == "ips"].copy()
        ipc["abs_rel_bias"] = ipc.rel_bias.abs()
        cells_c = ipc.groupby(cell_key).abs_rel_bias.median().reset_index()
        cells_c["unsafe"] = (cells_c.abs_rel_bias > 0.10).astype(float)
        panels.append(("unsafe", "IPS failure rate (cells),\ncutoff-centred logger",
                       "mean", cells_c))

    n_p = len(panels)
    fig, axes = plt.subplots(1, n_p, figsize=(12.4 if n_p == 3 else 16.0, 3.6))
    width = 0.26
    x = np.arange(len(regimes))
    shades = plt.cm.viridis(np.linspace(0.15, 0.75, len(temps)))
    for ax, (col, ylab, how, frame) in zip(axes, panels, strict=False):
        for j, t in enumerate(temps):
            vals = [
                getattr(
                    frame[(frame.logger_regime == r) & (frame.overlap_temperature == t)][col],
                    how,
                )()
                for r in regimes
            ]
            ax.bar(x + (j - (len(temps) - 1) / 2) * width, vals, width,
                   color=shades[j], label=rf"$\tau={t:g}$")
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABEL[r] for r in regimes], fontsize=9)
        ax.set_xlabel("logger regime", fontsize=9)
    axes[1].legend(title="logging temperature", fontsize=8, title_fontsize=8,
                   loc="upper left", framealpha=0.9)
    fig.suptitle(
        "RQ2 — the effect of logging sharpness depends on logger–target action alignment"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, fig_dir, "fig2_rq2_alignment")

"""Comparison figure: ten GRN methods on the ENCODE ChIP-seq benchmark.

Claim: every method lands within ~0.06 AUROC of chance, and which ones clear
the empirical chance band depends entirely on the cell line -- expression-based
scores clear it on HepG2 labels only, motif-only scores on K562 labels only.
cisTarget pruning leaves GRNBoost2's ranking essentially unchanged despite
discarding >97% of candidate pairs.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL = {
    "pearson": "Pearson correlation",
    "spearman": "Spearman correlation",
    "partial_correlation": "Partial correlation",
    "mutual_info": "Mutual information",
    "genie3": "GENIE3",
    "grnboost2": "GRNBoost2",
    "scenic": "SCENIC (GRNBoost2 + cisTarget)",
    "motif_cistarget": "cisTarget rank, best motif",
    "motif_ctscore": "cisTarget CRM score",
    "motif_pwm": "JASPAR PWM scan of promoters",
}
FAMILY = {
    "expression only": ["pearson", "spearman", "partial_correlation",
                        "mutual_info", "genie3", "grnboost2"],
    "expression + motif": ["scenic"],
    "motif only, no expression": ["motif_cistarget", "motif_ctscore", "motif_pwm"],
}
ORDER = [m for fam in FAMILY.values() for m in fam]


def _ypos():
    """Row positions, top to bottom, with a gap between method families."""
    y, cur = {}, 0.0
    for fam, members in FAMILY.items():
        for m in members:
            y[m] = cur
            cur += 1.0
        cur += 0.6
    top = max(y.values())
    return np.array([top - y[m] for m in ORDER])


YPOS = _ypos()
VIEW_LABEL = {"all_edges": "all 14,318 pairs",
              "motif_annotated_tfs": "motif-annotated TFs",
              "discriminative_edges": "discriminative pairs",
              "per_tf_mean_auroc": "per-TF mean"}
CELLS = ["K562", "HepG2"]


def dot_panel(ax, res, chance, view, title, show_ylabels, cell_col):
    sub = res[(res.comparison == "matched") & (res.view == view)]
    piv = sub.pivot_table(index="method", columns="expression_from",
                          values="auroc").reindex(ORDER)
    y = YPOS

    cb = chance[chance.view == view]
    ax.axvspan(cb.auroc_lo.min(), cb.auroc_hi.max(), color="0.86",
               lw=0, zorder=0)
    ax.axvline(0.5, color="0.55", lw=0.7, zorder=1)

    for yi, m in zip(y, ORDER):
        vals = [piv.loc[m, c] for c in CELLS if c in piv.columns]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) == 2:
            ax.plot(vals, [yi, yi], color="0.7", lw=0.9, zorder=2)
    for c in CELLS:
        if c not in piv.columns:
            continue
        v = piv[c].to_numpy(dtype=float)
        ok = np.isfinite(v)
        ax.scatter(v[ok], y[ok], s=26, color=cell_col[c], zorder=3,
                   label=c, clip_on=False)
        for yi, m in zip(y[~ok], np.array(ORDER)[~ok]):
            ax.text(0.5, yi, "  n.d.", va="center", ha="left",
                    color="0.45", fontsize=6)

    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in ORDER])
    if not show_ylabels:
        ax.tick_params(axis="y", labelleft=False)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.set_ylim(YPOS.min() - 0.7, YPOS.max() + 0.7)
    ax.set_xlabel("AUROC")
    ax.set_title(title, loc="left")
    ax.margins(x=0.06)
    return piv


def family_brackets(ax):
    """Thin spanning brackets naming the three method families."""
    y = np.arange(len(ORDER))[::-1]
    pos = {m: yi for m, yi in zip(ORDER, y)}
    short = {"expression only": "expression",
             "expression + motif": "+ motif",
             "motif only, no expression": "motif only"}
    for fam, members in FAMILY.items():
        lo, hi = min(pos[m] for m in members), max(pos[m] for m in members)
        ax.annotate("", xy=(-0.035, lo - 0.34), xytext=(-0.035, hi + 0.34),
                    xycoords=("axes fraction", "data"),
                    textcoords=("axes fraction", "data"),
                    arrowprops=dict(arrowstyle="-", lw=1.4, color="0.35"),
                    annotation_clip=False)
        if len(members) > 1:
            ax.text(-0.055, (lo + hi) / 2, short[fam], rotation=90,
                    va="center", ha="right", fontsize=6, color="0.35",
                    transform=ax.get_yaxis_transform(), clip_on=False)


def dumbbell_panel(ax, res, cell_col):
    sub = res[(res.comparison == "matched") &
              (res.method.isin(["grnboost2", "scenic"]))]
    piv = sub.pivot_table(index=["expression_from", "view"],
                          columns="method", values="auroc").dropna()
    piv = piv.reindex([(c, v) for c in CELLS for v in VIEW_LABEL]).dropna()
    y = np.arange(len(piv))[::-1]
    ax.axvline(0.5, color="0.55", lw=0.7, zorder=1)
    for yi, (_, r) in zip(y, piv.iterrows()):
        ax.plot([r.grnboost2, r.scenic], [yi, yi], color="0.7", lw=0.9, zorder=2)
    cols = [cell_col[c] for c, _ in piv.index]
    ax.scatter(piv.grnboost2, y, s=26, facecolor="white", edgecolor=cols,
               lw=1.2, zorder=3, clip_on=False)
    ax.scatter(piv.scenic, y, s=26, color=cols, zorder=3, clip_on=False)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}, {VIEW_LABEL[v]}" for c, v in piv.index])
    ax.set_ylim(-0.7, len(piv) - 0.3)
    ax.set_xlabel("AUROC")
    ax.set_title("Pruning barely changes the ranking", loc="left")
    d = (piv.scenic - piv.grnboost2).abs()
    ax.text(0.02, 0.97, f"|ΔAUROC| ≤ {d.max():.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6, color="0.3")
    ax.margins(x=0.10)
    return piv


def gap_panel(ax, res, cell_col):
    d = res.pivot_table(index=["method", "expression_from", "view"],
                        columns="comparison", values="auroc")
    d["gap"] = d["matched"] - d["swapped"]
    g = (d.reset_index().query("view == 'discriminative_edges'")
         .pivot_table(index="method", columns="expression_from", values="gap")
         .reindex(ORDER))
    y = YPOS
    ax.axvline(0, color="0.55", lw=0.7, zorder=1)
    h = 0.34
    for i, c in enumerate(CELLS):
        if c not in g.columns:
            continue
        v = g[c].to_numpy(dtype=float)
        off = (0.5 - i) * h
        ok = np.isfinite(v)
        ax.barh(y[ok] + off, v[ok], height=h, color=cell_col[c],
                zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(YPOS.min() - 0.7, YPOS.max() + 0.7)
    ax.set_xlabel("ΔAUROC, own line − other line")
    ax.set_title("Specificity reverses between lines", loc="left")
    ax.margins(x=0.10)
    return g


def main(out="results/grn_method_comparison.png"):
    res = pd.read_csv("results/grn_method_comparison.tsv", sep="\t")
    chance = pd.read_csv("results/random_chance_band.tsv", sep="\t")
    chance = chance[chance.comparison == "matched"]
    cell_col = {"K562": "#1b6ca8", "HepG2": "#d8731a"}

    fig = plt.figure(figsize=(7.1, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0],
                          hspace=0.50, wspace=0.10,
                          left=0.315, right=0.985, top=0.915, bottom=0.085)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    pa = dot_panel(axa, res, chance, "all_edges",
                   "Expression and motif split by cell line", True, cell_col)
    pb = dot_panel(axb, res, chance, "motif_annotated_tfs",
                   "Sequence alone is the best K562 score",
                   False, cell_col)
    pc = dumbbell_panel(axc, res, cell_col)
    pd_ = gap_panel(axd, res, cell_col)

    axa.legend(loc="upper left", frameon=False, fontsize=6,
               handletextpad=0.3, borderpad=0.2)
    axc.scatter([], [], s=26, facecolor="white", edgecolor="0.35", lw=1.2,
                label="GRNBoost2 (no pruning)")
    axc.scatter([], [], s=26, color="0.35", label="SCENIC (pruned)")
    axc.legend(loc="lower right", frameon=False, fontsize=6,
               handletextpad=0.3, borderpad=0.2)

    for ax, L, dx in ((axa, "a", -0.40), (axb, "b", -0.05),
                      (axc, "c", -0.40), (axd, "d", -0.05)):
        ax.text(dx, 1.13, L, transform=ax.transAxes, fontweight="bold",
                fontsize=10, va="bottom", ha="left")

    fig.text(0.30, 0.004, "grey band: 2.5–97.5th percentile of 20 random-score "
             "replicates   ·   points right of a band beat chance",
             fontsize=6, color="0.35", ha="left")
    fig.savefig(out, dpi=300)
    return fig, (pa, pb, pc, pd_)


if __name__ == "__main__":
    main()

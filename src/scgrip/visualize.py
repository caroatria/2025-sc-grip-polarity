"""
sc-GRIP result visualization
==============================
General-purpose plotting helpers for a `ScGRIP.fit()` results DataFrame
(columns: edge, tf, target, activation_score_mean, ...). Unlike the
figure-making scripts used for the paper (hardcoded to specific gene lists
and absolute file paths), everything here takes its inputs as parameters so
it works on any sc-GRIP output.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.colors import LinearSegmentedColormap

# Polarity palette (dataviz skill reference instance) -- kept IDENTICAL to
# scripts/plot_candidate_ranking.py's BLUE/RED so "blue = activation, red =
# repression" means the same thing on every figure across the whole talk,
# not just this module.
POLARITY_BLUE = "#2a78d6"   # activation
POLARITY_RED = "#e34948"    # repression
POLARITY_CMAP = LinearSegmentedColormap.from_list(
    "polarity_red_white_blue", [POLARITY_RED, "#f2f1ec", POLARITY_BLUE]
)  # low score (repression) -> red, high score (activation) -> blue


def plot_score_distribution(
    df: pd.DataFrame,
    score_col: str = "activation_score_mean",
    gene: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Histogram of predicted polarity scores, optionally filtered to edges
    touching a single gene (as TF or target)."""
    subset = df[df["tf"].eq(gene) | df["target"].eq(gene)] if gene else df

    ax = ax or plt.figure(figsize=(6, 4)).gca()
    subset[score_col].hist(bins=20, color="skyblue", edgecolor="black", ax=ax)
    ax.set_xlabel("Predicted activation score")
    ax.set_ylabel("Number of edges")
    ax.set_title(f"Score distribution{f' — {gene}' if gene else ''}")

    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def plot_score_distribution_by_truth(
    df: pd.DataFrame,
    ground_truth: dict[tuple[str, str], str],
    score_col: str = "activation_score_mean",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Predicted-score distribution split by TRUE polarity (from e.g.
    `scgrip.evaluate.load_trrust`) — the quality-check plot: if sc-GRIP's
    scores carry real signal, the true-Activation edges should sit visibly
    higher than the true-Repression ones, not just overlap. Restricted to
    (tf, target) pairs actually present in `ground_truth`; edges without a
    ground-truth label are silently excluded (this plots evaluated edges
    only, not a general score histogram — use `plot_score_distribution` for
    that)."""
    truth = pd.Series(
        [ground_truth.get((tf, tgt)) for tf, tgt in zip(df["tf"], df["target"])], index=df.index
    )
    labeled = df.assign(truth=truth).dropna(subset=["truth"])

    ax = ax or plt.figure(figsize=(5.5, 4.5)).gca()
    parts = ax.violinplot(
        [labeled.loc[labeled["truth"] == lab, score_col].to_numpy() for lab in ("Repression", "Activation")],
        positions=[0, 1], showmedians=True, widths=0.7,
    )
    for body, color in zip(parts["bodies"], (POLARITY_RED, POLARITY_BLUE)):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    for i, lab in enumerate(("Repression", "Activation")):
        y = labeled.loc[labeled["truth"] == lab, score_col]
        ax.scatter(np.full(len(y), i) + np.random.default_rng(0).uniform(-0.05, 0.05, len(y)),
                   y, s=10, color="black", alpha=0.4, zorder=3)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"True Repression\n(n={ (labeled['truth']=='Repression').sum() })",
                        f"True Activation\n(n={ (labeled['truth']=='Activation').sum() })"])
    ax.set_ylabel(f"predicted {score_col}")
    ax.set_title("Predicted score vs. TRRUST ground truth")

    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def build_network(
    df: pd.DataFrame,
    score_col: str = "activation_score_mean",
    min_out_degree: int = 1,
    focus_genes: Optional[list[str]] = None,
    ego_gene: Optional[str] = None,
    ego_direction: str = "both",
    supervised_only: bool = True,
    top_n_edges: Optional[int] = None,
) -> nx.DiGraph:
    """
    Build a directed TF -> target graph from a results DataFrame, edge
    weight = predicted polarity score.

    Parameters
    ----------
    min_out_degree : drop TFs with fewer than this many surviving out-edges
        (after `focus_genes` filtering), to declutter dense predictions.
        IGNORED when `ego_gene` is given — that mode is specifically about
        showing one gene's individual target/regulator leaves, which this
        filter would otherwise prune away entirely.
    focus_genes : if given, keep only edges where BOTH endpoints are in
        this list — for restricting to a known gene set (e.g. sponge-mouse
        conserved genes).
    ego_gene : if given, keep only edges touching this ONE gene (as TF or
        target) — for "show me everything this TF regulates" or "everything
        that regulates this target", the more common single-TF-network use
        case. Combine with `top_n_edges` since a TF's full out-edge set is
        usually far too dense to plot as-is.
    ego_direction : which edges to keep when `ego_gene` is set —
        "out" = only ego_gene's own targets (ego_gene as TF; "what does
        this gene regulate"), "in" = only ego_gene's own regulators
        (ego_gene as target; "what regulates this gene"), "both" = both
        (default). Found by hand-checking the "both" figure for Myc: every
        other TF that merely points AT Myc also counts as a "TF" node
        (large, drawn at hub size) but usually has just that one edge, so
        a handful of them pile up into an unreadable label clump right next
        to Myc in the force layout — "out"/"in" avoid mixing the two
        relationships into one graph.
    supervised_only : drop edges sc-GRIP left "uncertain" (no confident
        proxy label) — an uncertain edge has no real polarity to color an
        arrow by, so including it just adds visual noise at score~0.5.
    top_n_edges : if given, keep only the N edges with the most extreme
        (most confident) |score - 0.5|, applied AFTER every other filter.
    """
    rows = df
    if focus_genes is not None:
        focus = set(focus_genes)
        rows = rows[rows["tf"].isin(focus) & rows["target"].isin(focus)]
    if ego_gene is not None:
        if ego_direction == "out":
            rows = rows[rows["tf"].eq(ego_gene)]
        elif ego_direction == "in":
            rows = rows[rows["target"].eq(ego_gene)]
        elif ego_direction == "both":
            rows = rows[rows["tf"].eq(ego_gene) | rows["target"].eq(ego_gene)]
        else:
            raise ValueError(f"ego_direction must be 'out', 'in', or 'both', got {ego_direction!r}")
    if supervised_only and "supervised" in rows.columns:
        rows = rows[rows["supervised"]]
    if top_n_edges is not None:
        rows = rows.reindex((rows[score_col] - 0.5).abs().sort_values(ascending=False).index).head(top_n_edges)

    G = nx.DiGraph()
    for _, row in rows.iterrows():
        if row["tf"] == row["target"]:
            continue
        G.add_edge(row["tf"], row["target"], weight=float(row[score_col]))

    if ego_gene is not None:
        return G

    hubs = {n for n in G.nodes if G.out_degree(n) >= min_out_degree}
    keep_edges = [(u, v, d) for u, v, d in G.edges(data=True) if u in hubs]
    return nx.DiGraph(keep_edges)


def plot_network(
    G: nx.DiGraph,
    node_label_fn: Optional[Callable[[str], str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    seed: int = 0,
    layout_k: Optional[float] = None,
) -> plt.Axes:
    """
    Draw a TF-target graph, edges colored by predicted polarity score —
    blue = activation, red = repression (POLARITY_CMAP, same convention as
    scripts/plot_candidate_ranking.py's lollipop figure, so "blue/red" means
    the same thing on every polarity figure in the talk). TFs drawn larger
    than targets.

    layout_k : passed to `nx.spring_layout` (its `k` — target distance
        between nodes). Default (None) is networkx's own `1/sqrt(n)` guess,
        which is fine for a hub-shaped graph (one or two dominant TFs) but
        packs same-size, low-degree nodes too close together in a BROAD,
        non-hub-shaped graph (many TFs with just 1-2 edges each), causing
        their labels to overlap into an unreadable clump — found by hand
        on exactly that kind of graph. Try ~1.0-1.5 for those.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes to plot — check your filters.")

    ax = ax or plt.figure(figsize=(8, 6)).gca()
    pos = nx.spring_layout(G, seed=seed, k=layout_k, iterations=100)

    tf_nodes = {u for u, _ in G.out_edges()}
    target_nodes = set(G.nodes) - tf_nodes
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    norm = colors.Normalize(vmin=0, vmax=1)
    cmap = POLARITY_CMAP

    nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes, node_size=600, node_color="#2b2a28", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_size=200, node_color="#c9c7c0", ax=ax)

    labels = {n: node_label_fn(n) for n in G.nodes} if node_label_fn else {n: n for n in G.nodes}
    label_colors = {n: "white" if n in tf_nodes else "#2b2a28" for n in G.nodes}
    for n, (x, y) in pos.items():
        ax.text(x, y, labels[n], fontsize=8, ha="center", va="center", color=label_colors[n], zorder=5)

    nx.draw_networkx_edges(
        G, pos, edge_color=weights, edge_cmap=cmap, edge_vmin=0, edge_vmax=1,
        width=1.4, arrows=True, arrowsize=20, arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08", ax=ax,
    )
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("predicted polarity  (red = repression, blue = activation)")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def plot_degree_distribution(G: nx.DiGraph, ax: Optional[plt.Axes] = None, save_path: Optional[str] = None) -> plt.Axes:
    """Histogram of in-/out-degree for every node in the graph."""
    in_deg = [d for _, d in G.in_degree()]
    out_deg = [d for _, d in G.out_degree()]
    max_deg = max(in_deg + out_deg, default=0)

    ax = ax or plt.figure(figsize=(6, 4)).gca()
    bins = np.arange(0, max_deg + 2) - 0.5
    ax.hist(in_deg, bins=bins, alpha=0.6, color="tab:blue", label="in-degree")
    ax.hist(out_deg, bins=bins, alpha=0.6, color="tab:orange", label="out-degree")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of nodes")
    ax.legend()

    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def compute_polarity_agreement(
    dfs: dict[str, pd.DataFrame],
    confident_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise agreement matrix across several sc-GRIP result DataFrames
    (e.g. one per cell population) — for each pair, restricted to edges
    present and confidently predicted (`supervised`, i.e. not "uncertain")
    in BOTH populations, the fraction where `predicted_polarity` matches.

    Diagonal is always 1.0 (self-agreement, not "how repeatable is a
    rerun" — that's a different question, not this one). Off-diagonal
    NaN means two populations share zero confidently-predicted edges in
    common (nothing to compare).

    Parameters
    ----------
    dfs : {label: results_df}. Each results_df needs at least
        tf, target, predicted_polarity, and (if confident_only) supervised.
    confident_only : drop "uncertain"-polarity edges from each population
        before comparing — an edge neither population was confident about
        isn't informative about whether they *agree*.

    Returns
    -------
    (agreement, n_shared) — two same-shaped DataFrames (returned
    separately, not attached via `.attrs`: pandas compares `.attrs` with
    `==` in some display code paths, which raises on a DataFrame-valued
    attr — found by hand while testing this function). n_shared is the
    edge count each agreement percentage was computed over — check it
    before trusting a cell close to 0% or 100%, which is meaningless from
    only a handful of shared edges.
    """
    labels = list(dfs.keys())
    edge_maps = {}
    for label, df in dfs.items():
        d = df[df["supervised"]] if (confident_only and "supervised" in df.columns) else df
        edge_maps[label] = dict(zip(zip(d["tf"], d["target"]), d["predicted_polarity"]))

    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    n_shared = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for label in labels:
        matrix.loc[label, label] = 1.0
        n_shared.loc[label, label] = len(edge_maps[label])
    for a, b in combinations(labels, 2):
        shared = set(edge_maps[a]) & set(edge_maps[b])
        n_shared.loc[a, b] = n_shared.loc[b, a] = len(shared)
        if shared:
            agree = sum(edge_maps[a][e] == edge_maps[b][e] for e in shared) / len(shared)
            matrix.loc[a, b] = matrix.loc[b, a] = agree

    return matrix, n_shared


def plot_polarity_agreement_heatmap(
    matrix: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Heatmap of a `compute_polarity_agreement` matrix — sequential
    colormap since this is a magnitude (% agreement), not a polarity."""
    ax = ax or plt.figure(figsize=(0.6 * len(matrix) + 2.5, 0.6 * len(matrix) + 2)).gca()
    cmap = LinearSegmentedColormap.from_list("agreement_seq", ["#f2f1ec", "#2a78d6"])
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1)

    n = len(matrix)
    ax.set_xticks(range(n)); ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(matrix.index, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = matrix.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=7.5, color="white" if v > 0.55 else "#2b2a28")
    plt.colorbar(im, ax=ax, shrink=0.75, label="polarity agreement (shared confident edges)")
    ax.set_title("Cross-population polarity agreement", fontsize=12, fontweight="bold")

    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def plot_roc(y_true, y_score, ax: Optional[plt.Axes] = None, save_path: Optional[str] = None) -> plt.Axes:
    """ROC curve against ground-truth polarity labels (requires scikit-learn)."""
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ax = ax or plt.figure(figsize=(5, 5)).gca()
    ax.plot(fpr, tpr, label=f"sc-GRIP (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")

    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def plot_permutation_panel(
    result: dict,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    One panel of the conservation-permutation-test figure: null-distribution
    histogram (from `scgrip.evaluate.run_conservation_permutation_test`)
    with the real observed overlap marked and the empirical p-value
    annotated. Call once per network and lay the panels out yourself
    (`plt.subplots(1, 2, ...)`) to compare two networks side by side — see
    the notebook's Part 4 for the two-network version.
    """
    observed, perm_overlaps = result["observed"], result["perm_overlaps"]
    p_value, n_testable, n_tfs = result["p_value"], result["n_testable"], result["n_tfs"]

    ax = ax or plt.figure(figsize=(5.5, 4.3)).gca()
    ax.hist(perm_overlaps, bins=30, color="#9ec5f4", edgecolor="white", linewidth=0.6, zorder=2)
    ax.axvline(observed, color=POLARITY_RED, linewidth=2.2, zorder=3)

    y_top = ax.get_ylim()[1]
    ax.annotate(f"observed = {observed}", xy=(observed, y_top * 0.92), xytext=(8, 0),
                textcoords="offset points", fontsize=10, color=POLARITY_RED, fontweight="bold", va="center")

    p_label = "p < 0.0001" if p_value == 0 else f"p = {p_value:.4f}"
    ax.text(0.03, 0.95, f"{p_label}\n({n_testable} testable edges, {n_tfs} TFs)",
            transform=ax.transAxes, fontsize=9.5, va="top", ha="left")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("overlap (degree-preserving null)")
    ax.set_ylabel("permutations")

    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax

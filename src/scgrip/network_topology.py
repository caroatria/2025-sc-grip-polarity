"""Basic directed-graph topology summaries for CONFIRMED (conserved) sc-GRIP
edges -- i.e. TF->target relationships predicted in a non-model species that are
ALSO edges in a curated model-species network (as returned by
`evaluate.get_testable_edges(...)` filtered on `conserved_in_network`).

Four questions, each with its own function:
  - which TFs regulate more than one confirmed target?      -> multi_target_tfs
  - which targets are regulated by more than one TF?        -> multi_regulator_targets
  - are there cyclic relationships (A -> B -> ... -> A)?     -> find_cycles
  - are there cascades (TF1 -> TF2 -> TF3 -> ...)?           -> find_cascades

Everything works on the MOUSE gene names (`matched_mouse_tf`/`matched_mouse_target`
columns `get_testable_edges` adds when given `ortholog_map` -- not `mouse_tf`/
`mouse_target`, which only hold each gene's single display name and can silently
under-count real matches) -- since a "cascade" only makes sense as a real chain
of model-species regulatory relationships, not non-model gene ids. The confirmed-edge sets here
are small (dozens to a few hundred edges), so plain bounded DFS is fast and keeps
the four functions above dependency-free (no networkx needed).

Three plotting functions at the bottom (`plot_cycles`, `plot_cascade_core`,
`plot_cascade_chains`) DO use `networkx` -- already a dependency of
`scgrip.visualize` (install with `pip install "scgrip[plot]"`).
"""
from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.colors import LinearSegmentedColormap


def _edge_list(edges, tf_col: str = "matched_mouse_tf", target_col: str = "matched_mouse_target") -> list[tuple[str, str]]:
    """Accept either a DataFrame (with tf_col/target_col) or a plain iterable of
    (tf, target) pairs, and always return a deduplicated list of (tf, target)
    string pairs with self-loops dropped (a gene "regulating itself" isn't a
    cascade or cycle in the sense asked about here)."""
    if isinstance(edges, pd.DataFrame):
        pairs = zip(edges[tf_col], edges[target_col])
    else:
        pairs = edges
    return sorted({(tf, tgt) for tf, tgt in pairs if tf != tgt})


def multi_target_tfs(edges, tf_col: str = "matched_mouse_tf", target_col: str = "matched_mouse_target") -> pd.DataFrame:
    """TFs that regulate more than one confirmed target -- one row per such TF,
    with its target count and the sorted list of targets. These are the
    "hub" TFs on the regulator side."""
    targets_by_tf: dict[str, list[str]] = defaultdict(list)
    for tf, tgt in _edge_list(edges, tf_col, target_col):
        targets_by_tf[tf].append(tgt)

    rows = [{"tf": tf, "n_targets": len(tgts), "targets": sorted(tgts)}
            for tf, tgts in targets_by_tf.items() if len(tgts) > 1]
    if not rows:
        return pd.DataFrame(columns=["tf", "n_targets", "targets"])
    return pd.DataFrame(rows).sort_values("n_targets", ascending=False).reset_index(drop=True)


def multi_regulator_targets(edges, tf_col: str = "matched_mouse_tf", target_col: str = "matched_mouse_target") -> pd.DataFrame:
    """Targets regulated by more than one confirmed TF -- one row per such
    target, with its regulator count and the sorted list of TFs. These are
    "convergence points" -- candidate combinatorial-control targets."""
    tfs_by_target: dict[str, list[str]] = defaultdict(list)
    for tf, tgt in _edge_list(edges, tf_col, target_col):
        tfs_by_target[tgt].append(tf)

    rows = [{"target": tgt, "n_regulators": len(tfs), "tfs": sorted(tfs)}
            for tgt, tfs in tfs_by_target.items() if len(tfs) > 1]
    if not rows:
        return pd.DataFrame(columns=["target", "n_regulators", "tfs"])
    return pd.DataFrame(rows).sort_values("n_regulators", ascending=False).reset_index(drop=True)


def _canonical_cycle(path: list[str]) -> tuple[str, ...]:
    """Rotate a cycle's node list to start at its lexicographically-smallest
    node, so A->B->C->A and B->C->A->B collapse to the same reported cycle."""
    i = path.index(min(path))
    return tuple(path[i:] + path[:i])


def find_cycles(
    edges,
    tf_col: str = "matched_mouse_tf",
    target_col: str = "matched_mouse_target",
    max_length: int = 6,
) -> pd.DataFrame:
    """All simple directed cycles up to max_length nodes (A -> B -> ... -> A),
    found by bounded DFS from every node. One row per cycle, deduplicated so
    the same cycle found from a different starting node isn't reported twice.
    Returns an empty DataFrame (0 rows) if there are none -- for a small,
    mostly-tree-like regulatory network built from a handful of hub TFs
    pointing at largely non-TF targets, that's the expected, unremarkable
    result, not a sign anything is broken."""
    adjacency: dict[str, list[str]] = defaultdict(list)
    for tf, tgt in _edge_list(edges, tf_col, target_col):
        adjacency[tf].append(tgt)

    cycles_found: set[tuple[str, ...]] = set()

    def dfs(start: str, node: str, path: list[str], visited: set[str]) -> None:
        for nxt in adjacency.get(node, []):
            if nxt == start and len(path) >= 2:
                cycles_found.add(_canonical_cycle(path))
            elif nxt not in visited and len(path) < max_length:
                dfs(start, nxt, path + [nxt], visited | {nxt})

    for node in adjacency:
        dfs(node, node, [node], {node})

    rows = [{"cycle": " -> ".join(c) + f" -> {c[0]}", "length": len(c), "nodes": list(c)}
            for c in sorted(cycles_found)]
    if not rows:
        return pd.DataFrame(columns=["cycle", "length", "nodes"])
    return pd.DataFrame(rows).sort_values("length").reset_index(drop=True)


def find_cascades(
    edges,
    tf_col: str = "matched_mouse_tf",
    target_col: str = "matched_mouse_target",
    min_length: int = 3,
    max_length: int = 5,
) -> pd.DataFrame:
    """Directed simple paths TF1 -> TF2 -> TF3 (-> ...) with at least
    min_length nodes (default 3, i.e. at least two chained edges) -- real
    regulatory cascades, where an intermediate gene is BOTH a confirmed
    target of the previous TF AND a confirmed TF of the next edge. This falls
    out of a plain graph walk automatically: a path only extends past a node
    if that node has its own outgoing confirmed edges (i.e. is itself a TF
    here), so there's no separate "is this a TF" check needed.

    Only MAXIMAL chains are returned -- e.g. if A -> B -> C -> D is found,
    the shorter A -> B -> C prefix (also technically a valid cascade) is
    dropped, since it's fully contained in the longer one. Paths never
    revisit a node (that's what find_cycles is for)."""
    adjacency: dict[str, list[str]] = defaultdict(list)
    for tf, tgt in _edge_list(edges, tf_col, target_col):
        adjacency[tf].append(tgt)

    all_paths: list[tuple[str, ...]] = []

    def dfs(path: list[str]) -> None:
        node = path[-1]
        for nxt in adjacency.get(node, []):
            if nxt not in path and len(path) < max_length:
                dfs(path + [nxt])
        if len(path) >= min_length:
            all_paths.append(tuple(path))

    for start in adjacency:
        dfs([start])

    unique_paths = set(all_paths)
    maximal = [p for p in unique_paths
               if not any(len(other) > len(p) and other[:len(p)] == p for other in unique_paths)]

    rows = [{"chain": " -> ".join(p), "length": len(p) - 1, "nodes": list(p)} for p in maximal]
    if not rows:
        return pd.DataFrame(columns=["chain", "length", "nodes"])
    return pd.DataFrame(rows).sort_values("length", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting -- called from the notebook, kept here so the notebook itself only
# ever calls one function per figure instead of carrying the plotting code.
# ---------------------------------------------------------------------------

_NODE_DARK = "#2b2a28"   # same "TF" node color as scgrip.visualize.plot_network
_NODE_LIGHT = "#c9c7c0"  # same "target" node color as scgrip.visualize.plot_network
_EDGE_GRAY = "#6b6a66"
_EDGE_RED = "#e34948"    # POLARITY_RED in scgrip.visualize -- reused here only as a
                         # plain accent color, NOT to mean "repression" (there's no
                         # polarity information at this level, just structure)
_DRIVER_GREEN = "#2e9e5b"  # a third, new accent -- deliberately NOT reusing red/blue
                           # (POLARITY_*) or any scgrip.visualize.POP_COLORS entry,
                           # since "driver candidate" isn't a polarity or a stage

# for plot_tf_network_with_drivers' optional `regulation`/`scgrip_results` colorings --
# a distinct up/down pair (NOT reusing _DRIVER_GREEN, so "upregulated" and "driver
# candidate" can never be visually confused with each other), plus a mirror of
# scgrip.visualize's own POLARITY_* colors/colormap. Defined locally rather than
# imported from scgrip.visualize -- this module is loaded standalone
# (importlib.util.spec_from_file_location) with no package context, so a real `from . import
# visualize` would break; these are exact copies of that module's own hex values instead.
_UP_GREEN = "#3aa657"
_DOWN_RED = "#c0392b"
_DRIVER_RING = "#f0b429"  # amber outline -- driver status, shown as a ring instead of a
                          # fill color whenever `regulation` is active, so it never
                          # competes with the red/green fill for the same visual channel
_POLARITY_BLUE = "#2a78d6"   # == scgrip.visualize.POLARITY_BLUE (activation)
_POLARITY_RED = "#e34948"    # == scgrip.visualize.POLARITY_RED (repression) == _EDGE_RED
_POLARITY_CMAP = LinearSegmentedColormap.from_list(
    "polarity_red_white_blue", [_POLARITY_RED, "#f2f1ec", _POLARITY_BLUE]
)


def _circular_positions(nodes: list[str]) -> dict[str, tuple[float, float]]:
    """Evenly space nodes around a unit circle, in the given order -- used
    for cycles specifically so the loop shape is drawn unambiguously (a
    force-directed layout gives no such guarantee, especially for a 2- or
    3-node cycle, where it can just as easily draw a straight line)."""
    n = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    return {node: (np.cos(-a), np.sin(-a)) for node, a in zip(nodes, angles)}


def plot_cycles(cycles_df: pd.DataFrame, max_cols: int = 4, save_path: str | None = None):
    """One small circular diagram per cycle, nodes placed in their actual
    cycle order (see `_circular_positions`). Returns the Figure, or None (and
    prints a note) if `cycles_df` is empty -- e.g. for a network whose
    confirmed edges happen to form a pure tree/DAG, which is a real,
    unremarkable possible outcome, not an error."""
    if len(cycles_df) == 0:
        print("No cycles to plot.")
        return None

    n = len(cycles_df)
    ncols = min(max_cols, n)
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, cycles_df.iterrows()):
        nodes = row["nodes"]
        G = nx.DiGraph()
        for i in range(len(nodes)):
            G.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])
        pos = _circular_positions(nodes)

        nx.draw_networkx_nodes(G, pos, node_size=750, node_color=_NODE_DARK, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=7.5, font_color="white", ax=ax)
        nx.draw_networkx_edges(
            G, pos, edge_color=_EDGE_RED, width=1.6, arrows=True, arrowsize=22,
            arrowstyle="-|>", connectionstyle="arc3,rad=0.15", ax=ax,
        )
        ax.set_title(f"{row['length']}-node cycle", fontsize=9)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Cyclic relationships among the confirmed edges", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _cascade_edges(cascades_df: pd.DataFrame) -> list[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for nodes in cascades_df["nodes"]:
        edges.update(zip(nodes[:-1], nodes[1:]))
    return sorted(edges)


def cascade_participant_tfs(cascades_df: pd.DataFrame) -> list[str]:
    """The TF-role genes among any maximal cascade -- every gene that acts
    as a SOURCE somewhere in a cascade chain (out-degree > 0 in the union of
    cascade edges). Same "backbone" node set `plot_cascade_core` draws, but
    as a plain sorted list of mouse gene names -- for cross-referencing
    against something else (e.g. Part 6's stem->terminal driver-candidate
    screen) rather than plotting. Excludes pure single-hop leaf targets,
    which by construction can't drive anything further downstream within
    these cascades themselves."""
    if len(cascades_df) == 0:
        return []
    out_degree: dict[str, int] = defaultdict(int)
    for tf, _ in _cascade_edges(cascades_df):
        out_degree[tf] += 1
    return sorted(out_degree)


def plot_cascade_core(
    cascades_df: pd.DataFrame,
    title: str | None = None,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    seed: int = 9,
    layout_k: float = 4.0,
    layout: str = "spring",
    highlight_tfs: list[str] | set[str] | None = None,
    highlight_color: str = _DRIVER_GREEN,
):
    """Draw the TF<->TF core of every maximal cascade -- i.e. every edge
    whose SOURCE also appears as a TF somewhere else in a cascade, dropping
    edges into pure single-hop leaf targets (a leaf target has in-degree 1,
    out-degree 0 here, so it can never be part of a multi-hop chain itself,
    just its endpoint). This is the useful summary when there are too many
    individual chains to look at one by one (see `find_cascades`' docstring
    on the combinatorial explosion a dense hub sub-module causes): drawing
    every edge, leaves included, produces an unreadable hairball for a
    network with hundreds of leaf targets (found by hand on the 145-node
    hemato cascade set -- the very "packs low-degree nodes into an
    unreadable clump" failure `scgrip.visualize.plot_network` already warns
    about) -- the small TF-to-TF backbone underneath it is what's actually
    novel here, not the huge list of individual paths through it. Node size
    scales with out-degree within this backbone.

    layout : "spring" (default, unchanged -- force-directed, tuned by hand
        for the hemato/embryonic cores via `seed`/`layout_k` below), or one
        of three alternatives that can read more clearly for a densely
        interconnected TF core, where spring's node-node repulsion can still
        leave two symmetric, similarly-connected TFs fighting for the same
        spot:
        - "circular": every node on one ring, in a fixed, deterministic
          order -- no two nodes ever compete for position, at the cost of
          longer edges criss-crossing the middle.
        - "shell": two concentric rings -- TFs that regulate another TF in
          this core (real out-degree here) on the inner ring, pure "sink"
          TFs (in-degree only, within this core) on the outer ring. Useful
          when a handful of upstream hub TFs should read as visually
          central and everything downstream as peripheral.
        - "spectral": positions nodes by the graph Laplacian's
          eigenvectors, which tends to spatially separate genuinely
          distinct sub-clusters -- but degrades to a nearly straight line
          for a very small or singly-connected core (e.g. embryonic's
          2-node Myc<->Klf4 case), so prefer "circular" or "shell" there.
        `seed`/`layout_k` are spring-only and ignored by the other three.
    highlight_tfs : OPTIONAL, default None (-> today's plain dark/light node
        fill, completely unchanged). A list/set of mouse TF names -- any
        node in this backbone whose name is in it gets colored
        `highlight_color` instead of the usual dark/light-by-out-degree
        fill, and a small legend is added explaining the color (only when
        this is given, so the default call's figure is pixel-for-pixel
        identical to before). A name not present among this backbone's
        nodes (e.g. it only shows up as a single-hop leaf target, already
        dropped from this view, or isn't in the cascades at all) is
        silently ignored -- no error.
    highlight_color : the color `highlight_tfs` nodes get -- any matplotlib
        color (a hex string like "#e34948", a named color like "orange",
        an (r, g, b) tuple, ...). Defaults to `_DRIVER_GREEN`, the same
        green `plot_tf_network_with_drivers` uses for driver candidates.
        Has no effect when `highlight_tfs` isn't given.
    """
    if len(cascades_df) == 0:
        print("No cascades to plot.")
        return None

    G_full = nx.DiGraph(_cascade_edges(cascades_df))
    tf_nodes = set(cascade_participant_tfs(cascades_df))
    G = G_full.subgraph(tf_nodes).copy()
    n_leaves = G_full.number_of_nodes() - G.number_of_nodes()
    if n_leaves:
        print(f"({n_leaves} single-hop leaf targets omitted from this backbone view -- "
              f"they can't be part of a multi-hop chain themselves, only its endpoint)")

    ax = ax or plt.figure(figsize=(8, 6)).gca()
    out_deg = dict(G.out_degree())

    if layout == "spring":
        # seed=9, k=4.0 found by hand -- the defaults (seed=0, networkx's own k guess) put
        # Runx1/Stat5b and Irf1/Mef2c right on top of each other for the 13-node hemato core,
        # since those pairs really are symmetrically, densely connected to the same neighbors.
        pos = nx.spring_layout(G, seed=seed, k=layout_k, iterations=300)
    elif layout == "circular":
        pos = nx.circular_layout(G)
        ax.set_aspect("equal")  # else a non-square figure stretches the ring into an ellipse
    elif layout == "shell":
        inner = [n for n in G.nodes if out_deg.get(n, 0) > 0]
        outer = [n for n in G.nodes if out_deg.get(n, 0) == 0]
        shells = [shell for shell in (inner, outer) if shell]  # shell_layout rejects an empty shell
        pos = nx.shell_layout(G, nlist=shells) if len(shells) > 1 else nx.circular_layout(G)
        ax.set_aspect("equal")  # same reason as "circular" -- these are concentric rings
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
        ax.set_aspect("equal")
    else:
        raise ValueError(f"layout must be one of 'spring', 'circular', 'shell', 'spectral' -- got {layout!r}")

    highlight_set = set(highlight_tfs) if highlight_tfs else set()

    def _node_color(n):
        if n in highlight_set:
            return highlight_color
        return _NODE_DARK if out_deg.get(n, 0) > 0 else _NODE_LIGHT

    node_colors = [_node_color(n) for n in G.nodes]
    label_colors = {n: ("white" if (n in highlight_set or out_deg.get(n, 0) > 0) else _NODE_DARK) for n in G.nodes}
    node_sizes = [260 + 110 * out_deg.get(n, 0) for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=ax)
    for node, (x, y) in pos.items():
        ax.text(x, y, node, fontsize=7, ha="center", va="center", color=label_colors[node], zorder=5)
    # Reciprocal pairs (A->B AND B->A both present) drawn as ONE straight edge with an
    # arrowhead on both ends, instead of two overlapping curved arcs pointing opposite
    # ways -- found by hand that the two-arc version gets visually noisy fast in a densely
    # interconnected core (e.g. Runx1<->Stat5b, Irf1<->Runx1 in the hemato backbone).
    # `reciprocal_repr` keeps exactly one (u, v) per pair, alphabetically -- either
    # direction is a real edge in G when the pair is reciprocal, so either works to draw.
    reciprocal_pairs = {frozenset((u, v)) for u, v in G.edges() if G.has_edge(v, u)}
    reciprocal_repr = [tuple(sorted(pair)) for pair in reciprocal_pairs]
    single_edges = [(u, v) for u, v in G.edges() if frozenset((u, v)) not in reciprocal_pairs]

    if single_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=single_edges, edge_color=_EDGE_GRAY, width=1.1, arrows=True, arrowsize=20,
            arrowstyle="-|>", connectionstyle="arc3,rad=0.08", ax=ax,
        )
    if reciprocal_repr:
        nx.draw_networkx_edges(
            G, pos, edgelist=reciprocal_repr, edge_color=_EDGE_GRAY, width=1.1, arrows=True, arrowsize=20,
            arrowstyle="<|-|>", connectionstyle="arc3,rad=0", ax=ax,
        )

    if highlight_tfs:
        handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=highlight_color,
                               markersize=11, label="highlighted TF")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8.5, frameon=False)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    if save_path:
        ax.figure.savefig(save_path, dpi=200, bbox_inches="tight")
    return ax


def plot_cascade_chains(cascades_df: pd.DataFrame, top_n: int = 12, save_path: str | None = None):
    """Draw the top_n LONGEST cascades as literal left-to-right flow diagrams,
    one per row -- for picking out specific example chains to point to (e.g.
    in a talk), rather than reading the `chain` string column. Complements
    `plot_cascade_core`: that one shows the whole backbone at once, this one
    reads out individual paths through it."""
    if len(cascades_df) == 0:
        print("No cascades to plot.")
        return None

    chosen = cascades_df.sort_values("length", ascending=False).head(top_n).reset_index(drop=True)
    n = len(chosen)
    max_chain_len = max(len(nodes) for nodes in chosen["nodes"])
    fig, ax = plt.subplots(figsize=(1.9 * max_chain_len + 1, 0.65 * n + 0.6))

    for row_i, nodes in enumerate(chosen["nodes"]):
        y = n - row_i
        for x, node in enumerate(nodes):
            ax.text(x, y, node, fontsize=8.5, ha="center", va="center", zorder=3, color="white",
                    bbox=dict(boxstyle="round,pad=0.35", fc=_NODE_DARK, ec="none"))
        for x0, x1 in zip(range(len(nodes) - 1), range(1, len(nodes))):
            ax.annotate("", xy=(x1 - 0.22, y), xytext=(x0 + 0.22, y), zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color=_EDGE_GRAY, lw=1.3, mutation_scale=22))

    ax.set_xlim(-0.6, max_chain_len - 0.4)
    ax.set_ylim(0.3, n + 0.7)
    ax.axis("off")
    ax.set_title(f"Longest confirmed cascades (top {n})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _edge_scgrip_scores(confirmed_edges, tf_col, target_col, scgrip_results) -> dict[tuple[str, str], float]:
    """(mouse_tf, mouse_target) -> mean sc-GRIP activation_score_mean, for every
    drawn edge whose underlying sponge (tf, target) pair(s) have a prediction in
    `scgrip_results`. Joins through `confirmed_edges`' own `sponge_tf`/
    `sponge_target` columns (already sitting alongside `tf_col`/`target_col` in
    the same rows) rather than by name, since that's the only unambiguous way to
    connect a mouse-name edge back to the sponge prediction(s) behind it."""
    if isinstance(scgrip_results, pd.DataFrame):
        scgrip_results = [scgrip_results]

    raw_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for df in scgrip_results:
        for tf, tgt, score in zip(df["tf"], df["target"], df["activation_score_mean"]):
            raw_scores[(tf, tgt)].append(score)
    score_by_sponge_pair = {k: float(np.mean(v)) for k, v in raw_scores.items()}

    scores_by_mouse_edge: dict[tuple[str, str], list[float]] = defaultdict(list)
    for df in confirmed_edges:
        for sp_tf, sp_tgt, m_tf, m_tgt in zip(df["sponge_tf"], df["sponge_target"], df[tf_col], df[target_col]):
            score = score_by_sponge_pair.get((sp_tf, sp_tgt))
            if score is not None:
                scores_by_mouse_edge[(m_tf, m_tgt)].append(score)
    return {k: float(np.mean(v)) for k, v in scores_by_mouse_edge.items()}


def plot_tf_network_with_drivers(
    confirmed_edges,
    tf_list,
    driver_tfs,
    tf_col: str = "matched_mouse_tf",
    target_col: str = "matched_mouse_target",
    hide_leaf_targets: bool = False,
    regulation: dict[str, float] | None = None,
    scgrip_results=None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    seed: int = 0,
    layout_k: float | None = None,
):
    """Draw every confirmed edge whose TF (source) is in `tf_list` -- the
    full confirmed subgraph for a given set of TFs, NOT restricted to
    cascades the way `plot_cascade_core` is -- and highlight whichever of
    those TFs are ALSO in `driver_tfs` in a third color, so a TF that's
    independently confirmed by BOTH lines of evidence (network conservation
    AND expression-based driver screening) stands out at a glance.

    confirmed_edges : one DataFrame, or a list of DataFrames to pool (e.g.
        [hemato_confirmed, embryonic_confirmed], to draw both networks'
        edges for the same TF list in one figure).
    tf_list : restrict edges to ones whose TF is in this set -- e.g. every
        confirmed TF across both networks (`multi_target_tfs`' "tf" column,
        or `cascade_participant_tfs`, or any other TF list you have).
    driver_tfs : the subset of tf_list to highlight -- e.g. driver
        candidates for one specific trajectory from `find_driver_candidates`
        (mapped to mouse names via `ortholog_map`, NOT sponge ids).
    hide_leaf_targets : if True, also drop every edge whose TARGET is not
        itself in `tf_list` -- i.e. keep only TF<->TF edges. Default False
        draws the literal "every interaction of this TF list" the name
        promises, but for a `tf_list` of any real size that's a hairball of
        mostly single-hop leaf targets (same failure `plot_cascade_core` had
        to work around) -- set this to True for the legible TF-only core
        instead, same idea as that function's own leaf-pruning.

    regulation : OPTIONAL, default None (-> today's plain dark/green node fill,
        completely unchanged). A {mouse TF name: signed value} dict -- typically a
        `delta_mean` from `find_driver_candidates`' output for ONE chosen terminal
        fate, mapped from sponge gene id to mouse name the same way `driver_tfs`
        already is. When given, TF node FILL color becomes green (value > 0,
        "upregulated" in that terminal fate vs. stem) or red (value < 0) instead of
        the driver dark/green scheme -- a TF with no entry in this dict (e.g. it
        failed the QC detection floor for this specific fate) stays neutral dark,
        same as an untested gene always has. Driver-candidate status doesn't
        disappear when this is active -- it moves from the fill color to a
        distinct amber RING around the node (`_DRIVER_RING`), so the two signals
        (regulation direction, driver status) stay visually independent instead of
        competing for the same fill color. Target-only (non-TF) nodes are never
        affected -- always the plain light gray, regardless of this parameter.
    scgrip_results : OPTIONAL, default None (-> today's plain uniform gray edges,
        completely unchanged). One sc-GRIP results DataFrame, or a list of them
        (e.g. one per cell population -- `cluster_dfs[cluster_name]` from Part 3),
        with `tf` / `target` (SPONGE gene ids) / `activation_score_mean` columns.
        When given, an edge whose underlying sponge (tf, target) pair(s) have a
        prediction there is colored by that score using the same red/white/blue
        diverging scale `scgrip.visualize.plot_network` uses (red = repression,
        blue = activation) instead of plain gray, with a colorbar added -- i.e.
        "the predicted interaction type from sc-GRIP" for edges we have one for.
        An edge with no prediction available (not present in any given
        `scgrip_results` DataFrame) stays plain gray, same as always -- this
        parameter only ever adds information, never removes an edge. Matching is
        done via `confirmed_edges`' own `sponge_tf`/`sponge_target` columns
        (already present alongside `tf_col`/`target_col`), not by name -- several
        sponge (tf, target) pairs can collapse onto the same drawn mouse-name
        edge (paralogs sharing one ortholog), in which case their scores are
        averaged for that one edge.
    """
    if isinstance(confirmed_edges, pd.DataFrame):
        confirmed_edges = [confirmed_edges]

    tf_set = set(tf_list)
    driver_set = set(driver_tfs)

    edges: set[tuple[str, str]] = set()
    for df in confirmed_edges:
        edges.update((tf, tgt) for tf, tgt in zip(df[tf_col], df[target_col])
                     if tf in tf_set and tf != tgt)
    if not edges:
        print("No confirmed edges found for the given TF list.")
        return None

    if hide_leaf_targets:
        kept = {(tf, tgt) for tf, tgt in edges if tgt in tf_set}
        n_dropped = len(edges) - len(kept)
        if n_dropped:
            print(f"({n_dropped} edge(s) into non-TF leaf targets dropped -- hide_leaf_targets=True)")
        edges = kept
        if not edges:
            print("No TF<->TF edges left after dropping leaf targets -- nothing to plot.")
            return None
        if layout_k is None:
            layout_k = 1.8  # found by hand -- networkx's own guess under-spaces a small,
                             # densely-interconnected TF-only core (same issue plot_cascade_core
                             # tunes around); irrelevant for the default (leaves included) case,
                             # where hundreds of low-degree leaves dominate the layout anyway

    G = nx.DiGraph(sorted(edges))
    ax = ax or plt.figure(figsize=(9, 7)).gca()
    pos = nx.spring_layout(G, seed=seed, k=layout_k, iterations=200)

    tf_nodes = {n for n in G.nodes if n in tf_set}
    target_only_nodes = set(G.nodes) - tf_nodes
    driver_nodes = {n for n in tf_nodes if n in driver_set}
    plain_tf_nodes = tf_nodes - driver_nodes

    if regulation is None:
        nx.draw_networkx_nodes(G, pos, nodelist=target_only_nodes, node_size=200, node_color=_NODE_LIGHT, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=plain_tf_nodes, node_size=550, node_color=_NODE_DARK, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=driver_nodes, node_size=650, node_color=_DRIVER_GREEN, ax=ax)
    else:
        def _regulation_color(name: str) -> str:
            value = regulation.get(name)
            if value is None or value == 0:
                return _NODE_DARK
            return _UP_GREEN if value > 0 else _DOWN_RED

        nx.draw_networkx_nodes(G, pos, nodelist=target_only_nodes, node_size=200, node_color=_NODE_LIGHT, ax=ax)
        for node_set, size in [(plain_tf_nodes, 550), (driver_nodes, 650)]:
            if not node_set:
                continue
            fills = [_regulation_color(n) for n in node_set]
            ring_colors = [_DRIVER_RING if n in driver_nodes else "none" for n in node_set]
            ring_widths = [2.5 if n in driver_nodes else 0 for n in node_set]
            nx.draw_networkx_nodes(G, pos, nodelist=list(node_set), node_size=size, node_color=fills,
                                    edgecolors=ring_colors, linewidths=ring_widths, ax=ax)

    label_colors = {n: ("white" if n in tf_nodes else _NODE_DARK) for n in G.nodes}
    for n, (x, y) in pos.items():
        ax.text(x, y, n, fontsize=7.5, ha="center", va="center", color=label_colors[n], zorder=5)

    if scgrip_results is None:
        nx.draw_networkx_edges(
            G, pos, edge_color=_EDGE_GRAY, width=1.0, arrows=True, arrowsize=20,
            arrowstyle="-|>", connectionstyle="arc3,rad=0.08", ax=ax,
        )
    else:
        edge_scores = _edge_scgrip_scores(confirmed_edges, tf_col, target_col, scgrip_results)
        scored_edges = [e for e in G.edges if e in edge_scores]
        unscored_edges = [e for e in G.edges if e not in edge_scores]
        print(f"{len(scored_edges)}/{G.number_of_edges()} edge(s) matched to an sc-GRIP prediction "
              f"(the rest stay plain gray -- no prediction available for either direction).")
        if unscored_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=unscored_edges, edge_color=_EDGE_GRAY, width=1.0,
                arrows=True, arrowsize=20, arrowstyle="-|>", connectionstyle="arc3,rad=0.08", ax=ax,
            )
        if scored_edges:
            weights = [edge_scores[e] for e in scored_edges]
            nx.draw_networkx_edges(
                G, pos, edgelist=scored_edges, edge_color=weights, edge_cmap=_POLARITY_CMAP,
                edge_vmin=0, edge_vmax=1, width=1.4, arrows=True, arrowsize=20,
                arrowstyle="-|>", connectionstyle="arc3,rad=0.08", ax=ax,
            )
            sm = cm.ScalarMappable(cmap=_POLARITY_CMAP, norm=colors.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
            cbar.set_label("predicted polarity\n(red = repression, blue = activation)", fontsize=8)

    if regulation is None:
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=_DRIVER_GREEN, markersize=11, label="TF + driver candidate"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=_NODE_DARK, markersize=11, label="TF (not a driver candidate)"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=_NODE_LIGHT, markersize=9, label="target only (not a TF here)"),
        ]
    else:
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=_UP_GREEN, markersize=11, label="upregulated (this fate vs. stem)"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=_DOWN_RED, markersize=11, label="downregulated (this fate vs. stem)"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=_NODE_DARK, markersize=11, label="no data for this fate"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=_DRIVER_RING,
                       markeredgewidth=2, markersize=11, label="+ driver candidate (ring)"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=_NODE_LIGHT, markersize=9, label="target only (not a TF here)"),
        ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8.5, frameon=False)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    if save_path:
        ax.figure.savefig(save_path, dpi=200, bbox_inches="tight")
    return ax

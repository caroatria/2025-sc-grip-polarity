"""
sc-GRIP data loading and preparation
=====================================
Turns the two inputs described in the paper (Section III-D/E) into the
per-cell graph objects and correlation-derived proxy labels that the model
trains on:

    1. a single-cell gene expression matrix (cells x genes), typically
       exported from a Scanpy/Seurat preprocessing pipeline as .h5ad or .csv
    2. a TF-target adjacency matrix (TFs x target genes), typically derived
       from FIMO motif scanning of promoter regions

Everything here is pure data wrangling — no model code, no training loop —
so it can be unit-tested and reused independently of `scgrip.model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_expression_matrix(
    path: str,
    adata_path: Optional[str] = None,
    layer: Optional[str] = None,
    genes: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load a cells x genes expression matrix.

    Parameters
    ----------
    path : CSV path, used when `adata_path` is not given.
    adata_path : path to a .h5ad file (as produced by the Scanpy/Seurat
        preprocessing described in the paper). Takes priority over `path`.
    layer : which `adata.layers[...]` to read; `adata.X` if None.
    genes : if given (CSV path only), only parse these columns (plus the
        index column) via pandas' `usecols` instead of the whole file.
        `align_genes` would filter down to this same subset anyway, but
        parsing only what's needed up front matters when the file has many
        more genes than the adjacency matrix uses (e.g. a genome-wide
        export with 13k+ columns for a ~5k-gene adjacency matrix), where a
        plain `pd.read_csv` takes minutes and several GB of memory.

    Returns
    -------
    DataFrame indexed by cell barcode, columns = gene symbols.
    """
    if adata_path:
        import scanpy as sc
        import scipy.sparse as sp

        adata = sc.read_h5ad(adata_path)
        X = adata.layers[layer] if layer else adata.X
        if sp.issparse(X):
            X = X.toarray()
        return pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

    if genes is not None:
        header = pd.read_csv(path, nrows=0)
        index_col_name = header.columns[0]
        wanted = set(genes)
        usecols = [index_col_name] + [c for c in header.columns[1:] if c in wanted]
        return pd.read_csv(path, index_col=0, usecols=usecols)

    return pd.read_csv(path, index_col=0)


def read_adjacency_matrix(path: str, sheet_name=0) -> pd.DataFrame:
    """
    Load a TF (rows) x target gene (columns) adjacency matrix from CSV or
    Excel. Non-zero entries are treated as "candidate regulatory edge".
    """
    path = str(path)
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=sheet_name, index_col=0)
    return pd.read_csv(path, index_col=0)


# ---------------------------------------------------------------------------
# Gene alignment
# ---------------------------------------------------------------------------

def align_genes(expr_df: pd.DataFrame, edge_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restrict both matrices to genes that are (a) present in the expression
    data and (b) actually used as a TF or a target in the adjacency matrix.

    Returns
    -------
    (expr_df, edge_df) : subset copies, with `edge_df` reindexed so its rows
        are exactly the TFs and columns exactly the targets that survived.
    """
    tfs = edge_df.index.tolist()
    targets = edge_df.columns.tolist()
    expr_genes = set(expr_df.columns)

    common_tfs = sorted(set(tfs) & expr_genes)
    common_targets = sorted(set(targets) & expr_genes)
    common_genes = sorted(set(common_tfs) | set(common_targets))

    if not common_tfs or not common_targets:
        raise ValueError(
            "No overlap between the expression matrix's genes and the "
            "adjacency matrix's TFs/targets — check that gene identifiers "
            "match (e.g. both use gene symbols, not IDs)."
        )

    expr_out = expr_df[common_genes]
    edge_out = edge_df.loc[common_tfs, common_targets]
    return expr_out, edge_out


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

@dataclass
class EdgeIndex:
    """Bundles the directed TF->target edges with their human-readable names."""
    directed: torch.Tensor      # [2, n_edges], indices into `genes`
    names: list[str]            # e.g. "TF1->GeneA", aligned with `directed`'s columns
    genes: list[str]            # gene at each node index, aligned with expr_df columns

    @property
    def n_edges(self) -> int:
        return self.directed.size(1)


def build_edge_index(expr_df: pd.DataFrame, edge_df: pd.DataFrame, drop_self_loops: bool = True) -> EdgeIndex:
    """
    Turn the adjacency matrix into a directed edge index over the gene
    ordering of `expr_df.columns`, plus the "TF->target" edge names in the
    same order — this ordering is what every downstream tensor (labels,
    correlations, predicted scores) is aligned to.

    Parameters
    ----------
    drop_self_loops : if True (default), drop any TF->TF edge (a gene
        appearing as its own regulator in the adjacency matrix — real,
        e.g. some genes' promoters do contain their own binding motif).
        Every proxy-label mode in this package treats a gene's
        predictability from itself as maximal evidence — correlation of X
        with itself is trivially 1.0, and a combinatorial-logic fit
        (`logic_labels.py`) trivially recovers the identity function with
        F1=1.0 — and the identity function is monotonically increasing,
        so every self-loop gets confidently called "activation" purely by
        construction, regardless of the gene's real regulatory behavior.
        Checked directly against TRRUST: `Myc->Myc` and `Crebbp->Crebbp`
        are annotated *repression* there, but this trivial-tautology
        effect would call both "activation" with maximum confidence.
    """
    genes = expr_df.columns.tolist()
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    tf_idx, tgt_idx = np.where(edge_df.values != 0)
    tf_names = edge_df.index.to_numpy()[tf_idx]
    tgt_names = edge_df.columns.to_numpy()[tgt_idx]

    if drop_self_loops:
        keep = tf_names != tgt_names
        tf_names, tgt_names = tf_names[keep], tgt_names[keep]

    src = [gene_to_idx[t] for t in tf_names]
    dst = [gene_to_idx[t] for t in tgt_names]
    directed = torch.tensor([src, dst], dtype=torch.long)
    names = [f"{tf}->{tgt}" for tf, tgt in zip(tf_names, tgt_names)]

    return EdgeIndex(directed=directed, names=names, genes=genes)


def add_reverse_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Append the reverse of every edge, for undirected GCN message passing."""
    reverse = torch.stack([edge_index[1], edge_index[0]], dim=0)
    return torch.cat([edge_index, reverse], dim=1)


def build_cell_graphs(
    expr_df: pd.DataFrame,
    conv_edge_index: torch.Tensor,
    normalize: bool = True,
) -> list[Data]:
    """
    Build one `torch_geometric.data.Data` object per cell: same graph
    topology (`conv_edge_index`) throughout, node features are that cell's
    per-gene expression level.

    Parameters
    ----------
    normalize : if True, z-score each gene's expression across cells before
        building node features. Recommended — otherwise genes with a large
        dynamic range dominate the GCN's message passing.
    """
    values = expr_df.to_numpy(dtype=np.float32)
    if normalize:
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        values = (values - mean) / (std + 1e-6)

    return [
        Data(x=torch.from_numpy(values[i]).view(-1, 1), edge_index=conv_edge_index)
        for i in range(values.shape[0])
    ]


# ---------------------------------------------------------------------------
# Correlation-derived proxy labels (paper Eq. 1, Algorithm 1)
# ---------------------------------------------------------------------------

def _edge_correlations(
    expr_df: pd.DataFrame,
    edge_index: EdgeIndex,
    method: str = "pearson",
    condition_on_tf_detected: bool = False,
    min_cells: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-candidate-edge TF-target correlation, shared by every proxy-label
    function below.

    Parameters
    ----------
    condition_on_tf_detected : if True, each edge's correlation is computed
        using only the cells where that edge's TF is actually detected
        (expression > 0), instead of every cell. This follows the paper's
        own hypothesis (Section III-B): "a regulating TF must first be
        produced in order to affect ... any target genes" — a cell where
        the TF is absent carries no information about what the TF does
        when it *is* present. On data this sparse (routinely >90% zeros),
        pooling in TF-absent cells mostly adds shared technical-dropout
        noise common to many gene pairs, rather than biological signal.
    min_cells : minimum number of qualifying cells (all cells, or
        TF-detected cells when `condition_on_tf_detected=True`) required to
        trust an edge's correlation. Edges with fewer are reported as not
        `valid`, rather than silently defaulting to a correlation of 0.0 —
        which would otherwise be indistinguishable from genuine evidence of
        repression.

    Returns
    -------
    corr  : per-edge correlation (0.0 where not `valid`).
    valid : bool array, True where the correlation could be estimated from
        at least `min_cells` qualifying cells with nonzero variance.
    """
    src, dst = edge_index.directed
    src_idx, dst_idx = src.numpy(), dst.numpy()

    if not condition_on_tf_detected:
        # Deliberately not `expr_df.corr()`: that materializes a full
        # genes x genes matrix, which is fine at mouse-dataset gene counts
        # (a few hundred) but is O(n_genes^2) and becomes the dominant cost
        # by orders of magnitude on a genome-scale export (e.g. 5.6k genes -> a 31M-entry matrix to get ~80k specific
        # pairs out of). Standardize each column once, then take only the
        # dot products actually needed for `edge_index`'s specific pairs.
        values = expr_df.to_numpy(dtype=np.float64)
        if method == "spearman":
            values = pd.DataFrame(values).rank().to_numpy(dtype=np.float64)

        std = values.std(axis=0)
        z = np.zeros_like(values)
        nonzero_std = std > 0
        z[:, nonzero_std] = (values[:, nonzero_std] - values[:, nonzero_std].mean(axis=0)) / std[nonzero_std]

        corr = (z[:, src_idx] * z[:, dst_idx]).mean(axis=0)
        corr = np.nan_to_num(corr, nan=0.0)
        valid = np.full(edge_index.n_edges, expr_df.shape[0] >= min_cells)
        return corr, valid

    values = expr_df.to_numpy()
    corr = np.zeros(edge_index.n_edges, dtype=np.float64)
    valid = np.zeros(edge_index.n_edges, dtype=bool)

    for e, (i, j) in enumerate(zip(src_idx, dst_idx)):
        detected = values[:, i] > 0
        if detected.sum() < min_cells:
            continue
        x = pd.Series(values[detected, i])
        y = pd.Series(values[detected, j])
        if x.std() == 0 or y.std() == 0:
            continue
        r = x.corr(y, method=method)
        if np.isnan(r):
            continue
        corr[e], valid[e] = r, True

    return corr, valid


def correlation_proxy_labels(
    expr_df: pd.DataFrame,
    edge_index: EdgeIndex,
    tau: float = 0.5,
    method: str = "pearson",
    condition_on_tf_detected: bool = False,
    min_cells: int = 20,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Binary proxy labels, `y_uv = 1[corr(u, v) > tau]` (paper Eq. 1): every
    TF-target pair is labeled "activation" if their expression is
    correlated above `tau` across cells, "repression" otherwise.

    See `_edge_correlations` for `condition_on_tf_detected`/`min_cells`.

    Returns
    -------
    labels     : float tensor of {0, 1} for edges with `known_mask == True`.
    known_mask : bool array, one per edge in `edge_index.directed` — False
        only where the correlation itself couldn't be estimated (see
        `min_cells`); every other edge always gets a label here, since a
        fixed threshold never abstains the way the ternary/adaptive
        variants below do.
    corr       : the underlying correlation values (for diagnostics/plots).
    """
    corr, valid = _edge_correlations(expr_df, edge_index, method=method,
                                      condition_on_tf_detected=condition_on_tf_detected, min_cells=min_cells)
    full_labels = (corr > tau).astype(np.float32)
    labels = torch.tensor(full_labels[valid])
    return labels, valid, corr


def correlation_proxy_labels_ternary(
    expr_df: pd.DataFrame,
    edge_index: EdgeIndex,
    tau_low: float = -0.15,
    tau_high: float = 0.15,
    method: str = "pearson",
    condition_on_tf_detected: bool = False,
    min_cells: int = 20,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Ternary variant used for the TRRUST-style evaluation in the paper
    (Section IV, "these interaction types are ternary"): each edge is
    activation (1), repression (0), or unknown/uncertain (masked out of
    training and evaluation).

    See `_edge_correlations` for `condition_on_tf_detected`/`min_cells`.

    Returns
    -------
    labels     : float tensor of {0, 1} for edges with `known_mask == True`.
    known_mask : bool array, one per edge in `edge_index.directed`, True
                 where the correlation was both estimable and decisive
                 enough to assign a label.
    corr       : the underlying correlation values.
    """
    corr, valid = _edge_correlations(expr_df, edge_index, method=method,
                                      condition_on_tf_detected=condition_on_tf_detected, min_cells=min_cells)

    decisive = (corr >= tau_high) | (corr <= tau_low)
    known_mask = decisive & valid
    full_labels = np.where(corr >= tau_high, 1.0, 0.0)
    labels = torch.tensor(full_labels[known_mask], dtype=torch.float32)
    return labels, known_mask, corr


# ---------------------------------------------------------------------------
# Adaptive (dataset-relative) proxy labels
# ---------------------------------------------------------------------------
#
# The fixed threshold above (paper Eq. 1, tau=0.5) assumes TF-target
# correlations that actually reach 0.5 for real interactions. On sparse,
# zero-inflated single-cell data that's often not true: e.g. on the mouse
# liver/muscle candidate edges here, well under 5% of edges ever exceed a
# 0.5 Pearson correlation, so a fixed threshold labels almost everything
# "repression" by default and the model has near-degenerate training
# signal. Thresholding on the *dataset's own* candidate-edge correlation
# distribution (quantiles) keeps the same "is this edge relatively
# correlated or anti-correlated" logic while adapting to how strong
# correlations get on a given dataset.

def correlation_proxy_labels_adaptive(
    expr_df: pd.DataFrame,
    edge_index: EdgeIndex,
    low_q: float = 0.3,
    high_q: float = 0.7,
    method: str = "pearson",
    condition_on_tf_detected: bool = False,
    min_cells: int = 20,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Ternary proxy labels using quantiles of the candidate edges' own
    correlation distribution instead of fixed cutoffs: the bottom `low_q`
    fraction is labeled repression, the top `1 - high_q` fraction
    activation, the rest left unsupervised (uncertain).

    Parameters
    ----------
    low_q, high_q : quantiles (0-1) of the candidate-edge correlation
        distribution defining the repression/activation cutoffs. The
        defaults (0.3 / 0.7) label the bottom and top 30% of edges.
        Quantiles are computed only over edges with a valid correlation
        (see `min_cells`) so a handful of inestimable edges can't skew them.

    See `_edge_correlations` for `condition_on_tf_detected`/`min_cells`.

    Returns
    -------
    Same shape/semantics as `correlation_proxy_labels_ternary`.
    """
    if not 0 <= low_q < high_q <= 1:
        raise ValueError(f"Require 0 <= low_q < high_q <= 1, got low_q={low_q}, high_q={high_q}")

    corr, valid = _edge_correlations(expr_df, edge_index, method=method,
                                      condition_on_tf_detected=condition_on_tf_detected, min_cells=min_cells)
    if not valid.any():
        raise ValueError("No candidate edge had enough qualifying cells to estimate a correlation.")

    tau_low = np.quantile(corr[valid], low_q)
    tau_high = np.quantile(corr[valid], high_q)

    decisive = (corr >= tau_high) | (corr <= tau_low)
    known_mask = decisive & valid
    full_labels = np.where(corr >= tau_high, 1.0, 0.0)
    labels = torch.tensor(full_labels[known_mask], dtype=torch.float32)
    return labels, known_mask, corr

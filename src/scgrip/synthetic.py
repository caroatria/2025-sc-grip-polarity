"""
Synthetic TF-target datasets for smoke-testing sc-GRIP
=========================================================
Generates a small, fast expression matrix + adjacency matrix with a *known*
ground-truth polarity for every edge (some activating, some repressing), so
the pipeline's correctness can be checked without any real biological data:
if sc-GRIP is working, activating edges should get a high activation score
and repressing edges a low one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticDataset:
    expr_df: pd.DataFrame
    edge_df: pd.DataFrame
    truth: dict[str, str]   # "TF->target" -> "activation" | "repression"


def make_synthetic_dataset(n_cells: int = 300, noise: float = 0.15, seed: int = 0) -> SyntheticDataset:
    """
    Build a toy dataset with 4 TFs and 6 target genes:
        TF1, TF2 activate their targets (target tracks the TF's level)
        TF3, TF4 repress their targets  (target tracks 1 - the TF's level)

    Parameters
    ----------
    n_cells : number of synthetic "cells" (independent observations).
    noise   : Gaussian noise std added to each target's expression.
    seed    : RNG seed, for reproducibility.
    """
    rng = np.random.default_rng(seed)

    edges = [
        ("TF1", "GeneA", "activation"),
        ("TF1", "GeneB", "activation"),
        ("TF2", "GeneC", "activation"),
        ("TF3", "GeneD", "repression"),
        ("TF3", "GeneE", "repression"),
        ("TF4", "GeneF", "repression"),
    ]
    tfs = sorted({tf for tf, _, _ in edges})
    targets = sorted({tgt for _, tgt, _ in edges})

    tf_expr = {tf: rng.uniform(0, 1, n_cells) for tf in tfs}
    target_expr = {}
    for tf, target, polarity in edges:
        base = tf_expr[tf] if polarity == "activation" else 1.0 - tf_expr[tf]
        target_expr[target] = np.clip(base + rng.normal(0, noise, n_cells), 0, 1)

    expr_df = pd.DataFrame({**tf_expr, **target_expr})
    expr_df.index = [f"Cell{i + 1}" for i in range(n_cells)]

    edge_df = pd.DataFrame(0, index=tfs, columns=targets)
    for tf, target, _ in edges:
        edge_df.loc[tf, target] = 1

    truth = {f"{tf}->{target}": polarity for tf, target, polarity in edges}
    return SyntheticDataset(expr_df=expr_df, edge_df=edge_df, truth=truth)


def make_combinatorial_dataset(
    n_cells: int = 2000, p_tf: float = 0.2, noise: float = 0.2, seed: int = 0
) -> SyntheticDataset:
    """
    A target gated by a genuine AND of two TFs, plus a decoy TF with no
    real effect: `GeneAND` is on only when *both* TF1 and TF2 are on.
    Pairwise correlation (paper Eq. 1) can only ever see a monotonic,
    single-TF relationship — with a 3-edge candidate set this small, its
    adaptive quantile threshold (`label_mode="adaptive"`) has been
    observed to force TF1/TF2 into inconsistent known/unknown calls
    despite near-identical correlations, and to mislabel the decoy TF3 as
    a confident repressor purely from being the (arbitrary) lowest of the
    three. `label_mode="logic"` fits the AND directly and calls all three
    correctly (docs/design_notes.md, "Combinatorial-logic labels").

    Cells are simulated with technical zero-inflation in mind: an "off"
    TF/target is exactly 0 (no noise), matching real scRNA-seq's hard
    zeros for undetected genes; an "on" one has variable positive
    magnitude. `p_tf` controls how rare "on" is — lower values are closer
    to the sparsity of real scRNA-seq data (routinely >90% zero).
    """
    rng = np.random.default_rng(seed)

    def _noisy_expr(is_on: np.ndarray) -> np.ndarray:
        out = np.zeros(len(is_on))
        out[is_on] = np.clip(1 + rng.normal(0, noise, int(is_on.sum())), 0.05, None)
        return out

    tf1_on = rng.random(n_cells) < p_tf
    tf2_on = rng.random(n_cells) < p_tf
    tf3_on = rng.random(n_cells) < p_tf   # decoy — no real regulatory role
    gene_and_on = tf1_on & tf2_on

    expr_df = pd.DataFrame({
        "TF1": _noisy_expr(tf1_on),
        "TF2": _noisy_expr(tf2_on),
        "TF3": _noisy_expr(tf3_on),
        "GeneAND": _noisy_expr(gene_and_on),
    })
    expr_df.index = [f"Cell{i + 1}" for i in range(n_cells)]

    edge_df = pd.DataFrame(0, index=["TF1", "TF2", "TF3"], columns=["GeneAND"])
    edge_df.loc[["TF1", "TF2", "TF3"], "GeneAND"] = 1

    truth = {"TF1->GeneAND": "activation", "TF2->GeneAND": "activation", "TF3->GeneAND": None}
    return SyntheticDataset(expr_df=expr_df, edge_df=edge_df, truth=truth)

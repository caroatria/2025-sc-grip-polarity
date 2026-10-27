"""
Pseudobulking
==============
The paper's Algorithm 1 trains on individual cells, but on data this
sparse (routinely >90% zero) a single cell's expression is mostly
technical dropout riding on top of a much smaller number of real
biological states. Averaging random small groups of cells together
("pseudobulking", standard practice elsewhere in single-cell analysis)
trades single-cell resolution for a much less noisy expression profile —
useful when label collapse or the GCN not beating raw correlation look
like a signal-to-noise problem rather than a model-capacity one.

This is deliberately a preprocessing step, not something baked into
`ScGRIP` itself — like `scgrip.celltype.restrict_to_cell_type`, it
transforms an expression matrix before a `ScGRIP` is ever constructed, so
it composes freely with cell-type restriction, any label mode, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pseudobulk(
    expr_df: pd.DataFrame,
    group_size: int = 10,
    method: str = "mean",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Collapse groups of `group_size` randomly-chosen cells into one
    averaged "pseudo-cell" profile each, reducing N cells to N // group_size
    pseudo-cells.

    Parameters
    ----------
    expr_df : cells x genes expression matrix.
    group_size : number of real cells averaged into each pseudo-cell.
        Larger = less noisy but fewer effective training examples; there's
        no principled default, it trades off against how many cells
        `expr_df` has to begin with.
    method : "mean" or "median" aggregation within each group.
    seed : controls the random grouping (not stratified by anything —
        callers wanting pseudobulking within a specific population, e.g.
        one cell type, should restrict `expr_df` to it first, as
        `scgrip.celltype.restrict_to_cell_type` does).

    Returns
    -------
    DataFrame with `len(expr_df) // group_size` rows (any remainder cells
    are dropped, not left as a smaller final group), same columns as
    `expr_df`, indexed "pseudobulk_0", "pseudobulk_1", ...
    """
    if method not in ("mean", "median"):
        raise ValueError(f"method must be 'mean' or 'median', got {method!r}")
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")

    n_groups = len(expr_df) // group_size
    if n_groups == 0:
        raise ValueError(
            f"group_size={group_size} exceeds the number of cells ({len(expr_df)}) — "
            "no pseudo-cells could be formed."
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(expr_df))[: n_groups * group_size]
    groups = np.array_split(order, n_groups)

    values = expr_df.to_numpy()
    agg = np.mean if method == "mean" else np.median
    rows = np.stack([agg(values[group], axis=0) for group in groups])

    return pd.DataFrame(
        rows, columns=expr_df.columns, index=[f"pseudobulk_{i}" for i in range(n_groups)]
    )

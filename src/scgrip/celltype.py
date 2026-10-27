"""
Cell-type-conditioned analysis
================================
Expression matrices exported as CSV for a TF/target gene panel usually drop
the cell-type annotations of the original `.h5ad` file. This module reads
those annotations back out (keyed by cell barcode) so a `ScGRIP` run can be
restricted to one, more homogeneous, cell type instead of a whole mixed
tissue sample (see docs/design_notes.md, "Cell-type-conditioned labels").

Reads `obs` directly via h5py rather than through scanpy/anndata, because
older AnnData files store `obs` as one compound HDF5 dataset with categories
in `uns/<column>_categories`, and this keeps the dependency footprint small.
"""

from __future__ import annotations

import h5py
import pandas as pd


def read_obs_categorical(h5ad_path: str, column: str) -> pd.Series:
    """
    Read one categorical `.obs` column from an `.h5ad` file, indexed by
    cell barcode.

    Parameters
    ----------
    column : e.g. "cell_ontology_class", "louvain", "cluster_names" — any
        column that has a matching `uns/<column>_categories` array.
    """
    with h5py.File(h5ad_path, "r") as f:
        obs = f["obs"]
        barcodes = [b.decode() if isinstance(b, bytes) else b for b in obs["index"][:]]
        codes = obs[column][:]
        categories = [
            c.decode() if isinstance(c, bytes) else c
            for c in f["uns"][f"{column}_categories"][:]
        ]

    labels = [categories[c] if 0 <= c < len(categories) else None for c in codes]
    return pd.Series(labels, index=barcodes, name=column)


def restrict_to_cell_type(
    expr_df: pd.DataFrame,
    labels: pd.Series,
    cell_type: str = "auto",
) -> tuple[pd.DataFrame, str]:
    """
    Subset `expr_df` (cells x genes, indexed by barcode) to cells of a
    single cell type.

    Parameters
    ----------
    cell_type : a specific label from `labels`, or "auto" to pick the
        single largest cell type present in `expr_df`'s own cells.

    Returns
    -------
    (subset, cell_type_used) — `subset` has the same columns as
    `expr_df`, just fewer rows; `cell_type_used` is the resolved label
    (useful when `cell_type="auto"`).
    """
    aligned = labels.reindex(expr_df.index)
    if cell_type == "auto":
        counts = aligned.value_counts()
        if counts.empty:
            raise ValueError("No cell-type labels matched this expression matrix's cell barcodes.")
        cell_type = counts.idxmax()

    keep = aligned[aligned == cell_type].index
    if len(keep) == 0:
        raise ValueError(f"No cells labeled {cell_type!r} found among this expression matrix's barcodes.")

    return expr_df.loc[keep], cell_type

"""Readers for BEELINE files (expression, reference networks, TF lists)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_expression(path: Path) -> pd.DataFrame:
    """BEELINE `ExpressionData.csv` is genes x cells; return cells x genes, upper-case symbols."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.upper()
    df = df[~df.index.duplicated()]
    return df.T


def _to_sign(v) -> int:
    if pd.isna(v):
        return 0
    s = str(v).strip().lower()
    if s in ("+", "activation", "activator", "act"):
        return 1
    if s in ("-", "repression", "repressor", "inhibition", "rep"):
        return -1
    try:
        return int(np.sign(float(s)))
    except ValueError:
        return 0


def read_network(path: Path) -> pd.DataFrame:
    """
    Read a TF-target table into columns ``Gene1, Gene2, sign``
    (+1 activation, -1 repression, 0 unknown or reported with both signs).

    Accepts BEELINE networks (``Gene1,Gene2[,Type]``), the TRRUST raw TSV
    (TF, target, Activation/Repression/Unknown, PMIDs; no header) and
    DoRothEA/CollecTRI-style tables (source/tf, target, mor/sign/weight).
    """
    path = Path(path)
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    first = pd.read_csv(path, sep=sep, nrows=1, header=None).iloc[0].astype(str).str.lower().tolist()
    has_header = any(h in ("gene1", "source", "tf", "target", "type", "mor", "sign") for h in first)
    df = pd.read_csv(path, sep=sep, header=0 if has_header else None)

    if not has_header:
        df = df.iloc[:, :3]
        df.columns = ["Gene1", "Gene2", "Type"]
    else:
        cols = {c.lower(): c for c in df.columns}
        g1 = cols.get("gene1") or cols.get("source") or cols.get("tf")
        g2 = cols.get("gene2") or cols.get("target")
        t = cols.get("type") or cols.get("mor") or cols.get("sign") or cols.get("mode") or cols.get("weight")
        df = df.rename(columns={g1: "Gene1", g2: "Gene2"})
        df["Type"] = df[t] if t else np.nan

    df["sign"] = df["Type"].map(_to_sign)
    df["Gene1"] = df["Gene1"].astype(str).str.upper()
    df["Gene2"] = df["Gene2"].astype(str).str.upper()
    df = df[df.Gene1 != df.Gene2]

    def collapse(signs):
        known = sorted(set(signs) - {0})
        return known[0] if len(known) == 1 else 0

    return df.groupby(["Gene1", "Gene2"])["sign"].agg(collapse).reset_index()


def to_adjacency(net: pd.DataFrame) -> pd.DataFrame:
    """Edge list (Gene1, Gene2) -> TF x target 0/1 matrix, the format `scgrip` reads."""
    adj = pd.DataFrame(0, index=sorted(net.Gene1.unique()), columns=sorted(net.Gene2.unique()), dtype=np.int8)
    for a, b in zip(net.Gene1, net.Gene2):
        adj.at[a, b] = 1
    return adj


def select_genes(data_dir: Path, tf_file: Path, num_genes: int = 500, pval: float = 0.01):
    """
    BEELINE's gene selection (`utils/generateExpInputs.py`): all significantly
    varying TFs plus the `num_genes` most variable non-TF genes, Bonferroni-
    corrected p < `pval`. Returns (sorted genes, set of selected TFs).
    """
    go = pd.read_csv(Path(data_dir) / "GeneOrdering.csv", index_col=0)
    go.index = go.index.astype(str).str.upper()
    go = go.sort_values(go.columns[0])
    go = go[go.iloc[:, 0] < pval / len(go)]
    tfs = set(pd.read_csv(tf_file).iloc[:, 0].astype(str).str.upper())
    var_tfs = set(go.index) & tfs
    non_tf = go.drop(index=list(var_tfs))
    if len(go.columns) > 1:
        non_tf = non_tf.sort_values(non_tf.columns[1], ascending=False)
    return sorted(set(non_tf.index[:num_genes]) | var_tfs), var_tfs


def find_ground_truth(run_dir: Path, root: Path):
    """Nearest GroundTruthNetwork.csv / refNetwork.csv at or above `run_dir` (up to `root`)."""
    d = Path(run_dir)
    while True:
        for name in ("GroundTruthNetwork.csv", "refNetwork.csv"):
            if (d / name).exists():
                return d / name
        if d == Path(root) or d.parent == d:
            return None
        d = d.parent


def run_dirs(root: Path, max_per_group: int = 0) -> list[Path]:
    """All folders under `root` holding an ExpressionData.csv, optionally capped per parent folder."""
    dirs = sorted(p.parent for p in Path(root).rglob("ExpressionData.csv"))
    if not max_per_group:
        return dirs
    seen, keep = {}, []
    for d in dirs:
        seen[d.parent] = seen.get(d.parent, 0) + 1
        if seen[d.parent] <= max_per_group:
            keep.append(d)
    return keep

"""Motif-only baselines: how much of the ChIP-seq label is sequence-predictable
without looking at expression at all.

Two independent constructions over the same cisTarget resources, as a cross-check
against aggregation artefacts:

  motif_cistarget  best (lowest) cisTarget *rank* of the target gene across all
                   motifs annotated to the TF, taken over both the 500bp and the
                   10kbp promoter databases.
  motif_pwm        highest cisTarget *CRM score* of the target gene across all
                   motifs annotated to the TF, from the 500bp score database.

Both ignore expression entirely, so a TF's scores are identical in K562 and
HepG2; only the labelled pair set differs between the two lines. TFs with no
motif annotation in the v10 collection get a score of 0 for every target and are
flagged in the detail file.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

for _name, _val in [("object", object), ("bool", bool), ("int", int),
                    ("float", float), ("str", str)]:
    if not hasattr(np, _name):
        setattr(np, _name, _val)

from ctxcore.genesig import GeneSignature
from ctxcore.rnkdb import FeatherRankingDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_grn_methods as R
from pyscenic.utils import load_motif_annotations

CTX = "build/cistarget"
MOTIF_TBL = os.path.join(CTX, "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")
RANK_DBS = [
    os.path.join(CTX, "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"),
    os.path.join(CTX, "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"),
]
SCORE_DB = os.path.join(CTX, "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.scores.feather")
OUT = "predictions"


def tf2motifs(ann, tfs):
    out = {}
    for tf in tfs:
        if tf in ann.index.get_level_values("TF"):
            out[tf] = list(ann.xs(tf, level="TF").index.unique())
        else:
            out[tf] = []
    return out


def aggregate(path, genes, mapping, how):
    """how='min' over ranks (lower=better) or 'max' over scores (higher=better)."""
    db = FeatherRankingDatabase(fname=path, name=os.path.basename(path))
    gs = GeneSignature(name="targets", gene2weight={g: 1.0 for g in genes})
    mat = db.load(gs)                      # motifs x genes
    available = set(mat.index)
    res = {}
    for tf, motifs in mapping.items():
        use = [m for m in motifs if m in available]
        if not use:
            continue
        sub = mat.loc[use]
        res[tf] = sub.min(axis=0) if how == "min" else sub.max(axis=0)
    return pd.DataFrame(res).T, mat.shape        # TFs x genes


def run(cell):
    os.makedirs(OUT, exist_ok=True)
    X, tfs, cand = R.load(cell)
    genes = list(X.columns)
    ann = load_motif_annotations(MOTIF_TBL)
    mapping = tf2motifs(ann, tfs)
    annotated = sorted(t for t, m in mapping.items() if m)
    print(f"=== motif-only {cell}: {len(annotated)}/{len(tfs)} TFs have motif annotation",
          flush=True)

    # --- construction 1: best rank across annotated motifs, over both databases
    best = None
    for path in RANK_DBS:
        r, shape = aggregate(path, genes, mapping, "min")
        print(f"  ranks {os.path.basename(path)[:34]} -> {r.shape} (db {shape})", flush=True)
        best = r if best is None else np.minimum(best, r.reindex_like(best))
    total_genes = 27015
    rank_score = (total_genes - best).astype(float)        # higher = better

    # --- construction 2: max CRM score across annotated motifs, 500bp database
    pwm_score, shape = aggregate(SCORE_DB, genes, mapping, "max")
    print(f"  scores {os.path.basename(SCORE_DB)[:34]} -> {pwm_score.shape} (db {shape})",
          flush=True)

    def to_edges(mat, name):
        s = mat.stack()
        s.index.names = ["tf", "target"]
        out = cand.copy()
        out["score"] = out.join(s.rename("v"), on=["tf", "target"])["v"].fillna(0.0).to_numpy()
        out["annotated"] = out["tf"].isin(annotated).to_numpy()
        out[["tf", "target", "score"]].to_csv(
            os.path.join(OUT, f"preds_{name}_{cell}.tsv"), sep="\t", index=False)
        out.to_csv(os.path.join(OUT, f"{name}_detail_{cell}.tsv"), sep="\t", index=False)
        meta = {"method": name, "cell": cell, "n_edges": int(len(out)),
                "n_tfs": len(tfs), "n_tfs_annotated": len(annotated),
                "tfs_without_motif": sorted(set(tfs) - set(annotated)),
                "n_edges_on_annotated_tfs": int(out["annotated"].sum()),
                "score_nonzero_frac": float((out["score"] > 0).mean())}
        with open(os.path.join(OUT, f"meta_{name}_{cell}.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        print("  " + json.dumps({k: v for k, v in meta.items() if k != "tfs_without_motif"}),
              flush=True)

    to_edges(rank_score, "motif_cistarget")
    to_edges(pwm_score, "motif_ctscore")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="all", choices=["K562", "HepG2", "all"])
    a = ap.parse_args()
    for c in (["K562", "HepG2"] if a.cell == "all" else [a.cell]):
        run(c)

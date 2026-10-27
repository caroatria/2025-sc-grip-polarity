"""Full SCENIC (GRNBoost2 -> cisTarget motif pruning) on the ChIP-seq benchmark.

Protocol: GRNBoost2 is run on the unrestricted TF x gene space (not pre-filtered
to the labelled candidate pairs), co-expression modules are built with pySCENIC's
own defaults, and each module is pruned against the hg38 cisTarget ranking
databases with the published `pyscenic ctx` thresholds
(rank_threshold=5000, auc_threshold=0.05, nes_threshold=3.0,
motif_similarity_fdr=0.001, orthologous_identity_threshold=0.0).
Scoring happens afterwards, on the labelled pairs only.

pySCENIC's own prune2df drives dask/multiprocessing_on_dill, which cannot start
here, so module2df (the per-module worker prune2df calls) is driven directly in
a thread pool. The thresholds and code path are otherwise stock.
"""
import functools
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

# pySCENIC 0.12.1 predates numpy 2: np.object etc. are used only as entries in a
# column-dtype mapping. Restore them before importing pyscenic.
for _name, _val in [("object", object), ("bool", bool), ("int", int),
                    ("float", float), ("str", str)]:
    if not hasattr(np, _name):
        setattr(np, _name, _val)

from ctxcore.genesig import GeneSignature, Regulon
from ctxcore.rnkdb import FeatherRankingDatabase
from joblib import Parallel, delayed

import pyscenic.transform as PT
import pyscenic.utils as PU

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_grn_methods as R

CTX = "build/cistarget"
MOTIF_TBL = os.path.join(CTX, "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")
DBS = [
    ("hg38_500bp_up_100bp_down",
     os.path.join(CTX, "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather")),
    ("hg38_10kbp_up_10kbp_down",
     os.path.join(CTX, "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather")),
]
OUT = "predictions"
RANK_THRESHOLD = 5000
AUC_THRESHOLD = 0.05
NES_THRESHOLD = 3.0
N_JOBS = 8


def _modules4top_factors(adjacencies, n, context=frozenset()):
    """pandas-3 compatible replacement: groupby.apply no longer yields the group key."""
    df = (adjacencies.sort_values(PU.COLUMN_NAME_WEIGHT, ascending=False)
                     .groupby(PU.COLUMN_NAME_TARGET, sort=False).head(n))
    for tf_name, df_grp in df.groupby(by=PU.COLUMN_NAME_TF):
        if len(df_grp) > 0:
            yield Regulon(
                name=tf_name,
                context=frozenset(["top{}perTarget".format(n)]).union(context),
                transcription_factor=tf_name,
                gene2weight=list(zip(df_grp[PU.COLUMN_NAME_TARGET].values,
                                     df_grp[PU.COLUMN_NAME_WEIGHT].values)),
                gene2occurrence=[])


PU.modules4top_factors = _modules4top_factors


def adjacency(cell):
    """Unrestricted GRNBoost2 adjacency (every TF x gene link with importance > 0)."""
    X, tfs, cand = R.load(cell)
    t0 = time.time()
    M = R._arboreto(X, tfs, "grnboost2")
    secs = time.time() - t0
    adj = M.stack().rename(PU.COLUMN_NAME_WEIGHT).reset_index()
    adj.columns = [PU.COLUMN_NAME_TF, PU.COLUMN_NAME_TARGET, PU.COLUMN_NAME_WEIGHT]
    adj = adj[adj[PU.COLUMN_NAME_WEIGHT] > 0].reset_index(drop=True)
    return X, tfs, cand, adj, secs


def prune(modules, ann, warm_genes):
    """Run module2df for every (database, module) pair; concatenate as prune2df does.

    ctxcore caches database columns incrementally and appends to a shared pyarrow
    table, which is not thread-safe. Loading every gene once up front means each
    module's genes are already cached, so the appending code path never runs and
    the worker threads only read.
    """
    jobs = [(name, path, m) for name, path in DBS for m in modules]
    m2f = functools.partial(PT.module2features_auc1st_impl,
                            rank_threshold=RANK_THRESHOLD,
                            auc_threshold=AUC_THRESHOLD,
                            nes_threshold=NES_THRESHOLD,
                            filter_for_annotation=True)
    dbs = {name: FeatherRankingDatabase(fname=path, name=name) for name, path in DBS}
    warm = GeneSignature(name="warm", gene2weight={g: 1.0 for g in warm_genes})
    for name, d in dbs.items():
        loaded = d.load(warm)
        print(f"  warmed {name}: {loaded.shape[1]} of {len(warm_genes)} genes", flush=True)

    def one(name, path, module):
        try:
            return PT.module2df(dbs[name], module, ann, module2features_func=m2f)
        except Exception as exc:  # a module with no enriched feature is normal
            print(f"  ! {name} {module.transcription_factor}: {type(exc).__name__}: {exc}",
                  flush=True)
            return None

    parts = Parallel(n_jobs=N_JOBS, backend="threading", verbose=0)(
        delayed(one)(*j) for j in jobs)
    parts = [p for p in parts if p is not None and len(p)]
    return pd.concat(parts) if parts else pd.DataFrame()


def edge_scores(df, adj, cand):
    """Per-edge SCENIC score over the labelled candidate pairs.

    An edge survives pruning if the target gene is recovered by at least one
    motif that is enriched in a module of that TF and annotated to that TF.
    Surviving edges are ranked by the best NES among their supporting motifs,
    with the GRNBoost2 importance breaking ties; non-surviving edges rank below
    all survivors. The reported score is that lexicographic order mapped to
    [0, 1], so it is a pure ranking with no scale mixing.
    """
    best_nes, n_motifs = {}, {}
    if len(df):
        nes_col = [c for c in df.columns if c[-1] == "NES"][0]
        tgt_col = [c for c in df.columns if c[-1] == "TargetGenes"][0]
        for (tf, _motif), row in df.iterrows():
            nes = float(row[nes_col])
            for gene, _w in row[tgt_col]:
                key = (tf, gene)
                if nes > best_nes.get(key, -np.inf):
                    best_nes[key] = nes
                n_motifs[key] = n_motifs.get(key, 0) + 1

    imp = adj.set_index([PU.COLUMN_NAME_TF, PU.COLUMN_NAME_TARGET])[PU.COLUMN_NAME_WEIGHT]
    out = cand.copy()
    keys = list(zip(out["tf"], out["target"]))
    out["nes"] = [best_nes.get(k, 0.0) for k in keys]
    out["n_motifs"] = [n_motifs.get(k, 0) for k in keys]
    out["importance"] = [float(imp.get(k, 0.0)) for k in keys]
    order = out.sort_values(["nes", "importance"], ascending=True).index
    rank = pd.Series(np.arange(1, len(out) + 1) / len(out), index=order)
    out["score"] = rank.reindex(out.index).to_numpy()
    return out


def run(cell):
    os.makedirs(OUT, exist_ok=True)
    print(f"=== SCENIC {cell}", flush=True)
    X, tfs, cand, adj, grn_secs = adjacency(cell)
    print(f"  grnboost2: {len(adj)} links over {len(tfs)} TFs x {X.shape[1]} genes "
          f"({grn_secs:.0f}s)", flush=True)

    modules = PU.modules_from_adjacencies(adj, X)
    print(f"  modules: {len(modules)} over {len(set(m.transcription_factor for m in modules))} TFs",
          flush=True)

    ann = PU.load_motif_annotations(MOTIF_TBL)
    t0 = time.time()
    df = prune(modules, ann, list(X.columns))
    prune_secs = time.time() - t0
    print(f"  enriched motif rows: {len(df)} ({prune_secs:.0f}s)", flush=True)

    edges = edge_scores(df, adj, cand)
    edges[["tf", "target", "score"]].to_csv(
        os.path.join(OUT, f"preds_scenic_{cell}.tsv"), sep="\t", index=False)
    edges.to_csv(os.path.join(OUT, f"scenic_detail_{cell}.tsv"), sep="\t", index=False)
    if len(df):
        df.to_pickle(os.path.join(OUT, f"scenic_enrichment_{cell}.pkl"))

    regulons = []
    try:
        regulons = PT.df2regulons(df) if len(df) else []
        with open(os.path.join(OUT, f"regulons_{cell}.pkl"), "wb") as fh:
            pickle.dump(regulons, fh)
        pd.DataFrame([{"tf": r.transcription_factor, "n_targets": len(r)} for r in regulons]) \
          .to_csv(os.path.join(OUT, f"regulons_{cell}.tsv"), sep="\t", index=False)
    except Exception as exc:
        print(f"  ! df2regulons failed: {type(exc).__name__}: {exc}", flush=True)

    meta = {
        "method": "scenic", "cell": cell,
        "grnboost2_seconds": round(grn_secs, 1),
        "prune_seconds": round(prune_secs, 1),
        "n_adjacency_links": int(len(adj)),
        "n_modules": len(modules),
        "n_enrichment_rows": int(len(df)),
        "n_regulons": len(regulons),
        "n_edges_scored": int(len(edges)),
        "n_edges_surviving_pruning": int((edges["nes"] > 0).sum()),
        "n_tfs_with_surviving_edge": int(edges.loc[edges["nes"] > 0, "tf"].nunique()),
        "databases": [n for n, _ in DBS],
        "rank_threshold": RANK_THRESHOLD, "auc_threshold": AUC_THRESHOLD,
        "nes_threshold": NES_THRESHOLD,
    }
    with open(os.path.join(OUT, f"meta_scenic_{cell}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("  " + json.dumps(meta), flush=True)
    return meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="all", choices=["K562", "HepG2", "all"])
    a = ap.parse_args()
    for c in (["K562", "HepG2"] if a.cell == "all" else [a.cell]):
        run(c)

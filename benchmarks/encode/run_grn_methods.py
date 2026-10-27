"""Run established GRN-inference methods on the ENCODE edge-existence benchmark.

Every method sees exactly what sc-GRIP sees: the cells x genes expression matrix
and the list of candidate TFs. Nothing sees the labels.

Methods
-------
pearson, spearman      marginal co-expression magnitude
mutual_info            kNN mutual information (Kraskov), ARACNe / PIDC family
partial_correlation    full-order partial correlation from a Ledoit-Wolf
                       shrunk precision matrix (PPCOR-equivalent)
genie3                 Huynh-Thu 2010, per-target random forest
grnboost2              Moerman 2019, per-target stochastic GBM with early stop
random                 chance reference

genie3 and grnboost2 call arboreto's own per-target routine
(arboreto.core.infer_partial_network) with arboreto's published regressor
settings. Only arboreto's dask scheduling layer is replaced by a joblib loop,
so the run does not need a dask cluster; the estimators,
early-stopping monitor and importance extraction are arboreto's.

Usage
-----
    python run_grn_methods.py --cell K562 --method genie3
    python run_grn_methods.py --cell K562 --method all
"""
import argparse
import json
import os
import time

import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logging.getLogger("arboreto").setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "benchmark")
OUT = os.path.join(HERE, "predictions")
SEED = 0
N_JOBS = 8

METHODS = ["pearson", "spearman", "mutual_info", "partial_correlation",
           "genie3", "grnboost2", "random"]


def load(cell):
    X = pd.read_csv(os.path.join(BENCH, f"expression_{cell}.csv"), index_col=0)
    adj = pd.read_csv(os.path.join(BENCH, "candidate_adjacency.csv"), index_col=0)
    tfs = [t for t in adj.index if t in X.columns]
    cand = pd.read_csv(os.path.join(BENCH, f"labels_{cell}.tsv"),
                       sep="\t")[["tf", "target"]]
    # drop genes with no variance in this line: no method can score them
    keep = X.columns[X.std(axis=0).to_numpy() > 0]
    X = X[keep]
    tfs = [t for t in tfs if t in keep]
    return X, tfs, cand


def _matrix_to_edges(M, cand):
    """M: DataFrame tf x target of scores -> score per candidate edge."""
    s = M.stack()
    s.index.names = ["tf", "target"]
    out = cand.join(s.rename("score"), on=["tf", "target"])
    out["score"] = out["score"].fillna(0.0)
    return out


# --------------------------------------------------------------- correlation
def _corr(X, tfs, rank):
    A = X.rank(axis=0).to_numpy() if rank else X.to_numpy()
    A = A - A.mean(0)
    A /= np.linalg.norm(A, axis=0) + 1e-12
    C = np.abs(A.T @ A)
    return pd.DataFrame(C, index=X.columns, columns=X.columns).loc[tfs]


def pearson(X, tfs):
    return _corr(X, tfs, rank=False)


def spearman(X, tfs):
    return _corr(X, tfs, rank=True)


# ------------------------------------------------------- partial correlation
def partial_correlation(X, tfs):
    from sklearn.covariance import LedoitWolf
    P = np.linalg.pinv(LedoitWolf().fit(X.to_numpy()).covariance_)
    d = np.sqrt(np.diag(P))
    R = np.abs(-P / np.outer(d, d))
    return pd.DataFrame(R, index=X.columns, columns=X.columns).loc[tfs]


# ------------------------------------------------------- mutual information
def mutual_info(X, tfs):
    from sklearn.feature_selection import mutual_info_regression
    T = X[tfs].to_numpy()
    cols = list(X.columns)

    def one(j):
        y = X.iloc[:, j].to_numpy()
        mi = mutual_info_regression(T, y, n_neighbors=3, random_state=SEED)
        return j, mi

    res = Parallel(n_jobs=N_JOBS, backend="threading", verbose=0)(
        delayed(one)(j) for j in range(len(cols)))
    M = np.zeros((len(tfs), len(cols)))
    for j, mi in res:
        M[:, j] = mi
    return pd.DataFrame(M, index=tfs, columns=cols)


# ------------------------------------------------------------ tree ensembles
def _arboreto(X, tfs, kind):
    from arboreto.core import (infer_partial_network, to_tf_matrix,
                               RF_KWARGS, SGBM_KWARGS, EARLY_STOP_WINDOW_LENGTH)
    genes = list(X.columns)
    mat = X.to_numpy(dtype=np.float32)
    tf_matrix, tf_names = to_tf_matrix(mat, genes, tfs)
    rtype, rkw = (("RF", RF_KWARGS) if kind == "genie3" else ("GBM", SGBM_KWARGS))

    def one(j):
        return infer_partial_network(
            regressor_type=rtype, regressor_kwargs=rkw,
            tf_matrix=tf_matrix, tf_matrix_gene_names=tf_names,
            target_gene_name=genes[j], target_gene_expression=mat[:, j],
            include_meta=False,
            early_stop_window_length=EARLY_STOP_WINDOW_LENGTH, seed=SEED)

    parts = Parallel(n_jobs=N_JOBS, backend="threading", verbose=0)(
        delayed(one)(j) for j in range(len(genes)))
    links = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True)
    M = links.pivot_table(index="TF", columns="target", values="importance",
                          aggfunc="first")
    return M.reindex(index=tfs, columns=genes).fillna(0.0)


def genie3(X, tfs):
    return _arboreto(X, tfs, "genie3")


def grnboost2(X, tfs):
    return _arboreto(X, tfs, "grnboost2")


# ------------------------------------------------------------------- random
def random_scores(X, tfs, seed=SEED):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.random((len(tfs), X.shape[1])),
                        index=tfs, columns=X.columns)


FUNCS = {"pearson": pearson, "spearman": spearman, "mutual_info": mutual_info,
         "partial_correlation": partial_correlation, "genie3": genie3,
         "grnboost2": grnboost2, "random": random_scores}


def run(cell, method):
    X, tfs, cand = load(cell)
    t0 = time.time()
    M = FUNCS[method](X, tfs)
    dt = time.time() - t0
    edges = _matrix_to_edges(M, cand)
    assert len(edges) == len(cand), f"{len(edges)} != {len(cand)}"
    assert np.isfinite(edges.score).all(), "non-finite scores"
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"preds_{method}_{cell}.tsv")
    edges.to_csv(path, sep="\t", index=False)
    meta = {"method": method, "cell": cell, "seconds": round(dt, 1),
            "n_edges": len(edges), "n_tfs": len(tfs), "n_genes": X.shape[1],
            "n_cells": X.shape[0], "score_nonzero_frac": float((edges.score > 0).mean())}
    with open(os.path.join(OUT, f"meta_{method}_{cell}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta), flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["K562", "HepG2"])
    ap.add_argument("--method", required=True, choices=METHODS + ["all"])
    a = ap.parse_args()
    for m in (METHODS if a.method == "all" else [a.method]):
        run(a.cell, m)

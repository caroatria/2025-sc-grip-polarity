#!/usr/bin/env python3
"""
BEELINE benchmark: edge EXISTENCE (TF -> target, sign ignored), label-free.

scGRIP scores every candidate regulator -> gene pair with `scgrip.existence`
(no ground truth used in training). The reference network is used only for
scoring. Regulators are the selected TFs (scRNA-Seq) or all genes
(Curated/Synthetic), as in BEELINE.

Metrics (BEELINE definitions; 1 = random for the ratios):
  AUPRC_ratio      AUPRC / edge density
  EPR              precision in the top-k (k = number of true edges) / density
  EPR@100, EPR@500 precision in the top 100 / 500 pairs / density
  AUROC
  within_TF_AUROC  mean AUROC of target ranking inside each TF (0.5 = random);
                   unaffected by simply ranking hub TFs first

Baselines on the same pairs: |Spearman|, |Pearson|, CLR(|Spearman|), optional
GENIE3 (--genie3; ExtraTrees re-implementation), and two "hub" baselines that
ignore which target belongs to which TF: TF_degree and degree_product (degree
in the proxy-label graph). If a hub baseline scores as well as a method, the
benchmark is rewarding hub ranking rather than edge recovery.

    python existence.py simulated --root data/BEELINE-data/inputs/Curated \
        --max-runs-per-group 1 --genie3 --out results_local/existence_curated

    python existence.py experimental --data data/BEELINE-data/inputs/scRNA-Seq/hESC \
        --network data/Networks/human/hESC-ChIP-seq-network.csv \
        --tfs data/human-tfs.csv --out results_local/existence_hESC
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scgrip.existence import ExistenceConfig, abs_correlation, candidate_pairs, clr_normalize, score_candidate_edges

from beeline_io import find_ground_truth, read_expression, read_network, run_dirs, select_genes


def genie3_scores(X, genes, regulators, src, dst, n_trees=100, seed=0):
    """GENIE3 (Huynh-Thu et al. 2010): one ExtraTrees regressor per target gene."""
    from sklearn.ensemble import ExtraTreesRegressor
    g_idx = {g: i for i, g in enumerate(genes)}
    reg_idx = np.array([g_idx[r] for r in regulators if r in g_idx])
    Xn = X / (X.std(0) + 1e-9)
    imp = np.zeros((len(genes), len(genes)))
    for j in range(len(genes)):
        feats = reg_idx[reg_idx != j]
        if len(feats) == 0:
            continue
        et = ExtraTreesRegressor(n_estimators=n_trees, max_features="sqrt", n_jobs=-1, random_state=seed)
        et.fit(Xn[:, feats], Xn[:, j])
        imp[feats, j] = et.feature_importances_
    return imp[src, dst]


def existence_metrics(y: np.ndarray, s: np.ndarray, reg: np.ndarray) -> dict:
    n_pos, n = int(y.sum()), len(y)
    if n_pos in (0, n):
        return {}
    dens = n_pos / n
    order = np.argsort(-s, kind="stable")
    top = s >= s[order[n_pos - 1]]  # ties at the cut-off included, as in BEELINE
    out = {"AUPRC_ratio": float(average_precision_score(y, s) / dens),
           "EPR": float(y[top].mean() / dens),
           "AUROC": float(roc_auc_score(y, s))}
    for k in (100, 500):
        if k < n:
            out[f"EPR@{k}"] = float(y[order[:k]].mean() / dens)
    per_tf = [roc_auc_score(g.y, g.s) for _, g in pd.DataFrame({"r": reg, "y": y, "s": s}).groupby("r")
              if 0 < g.y.sum() < len(g)]
    if per_tf:
        out["within_TF_AUROC"] = float(np.mean(per_tf))
    return out


def evaluate(expr: pd.DataFrame, genes, regulators, truth_edges: set, a, out_dir: Path, meta: dict) -> dict:
    cfg = ExistenceConfig(label_scheme=a.label_scheme, pos_frac=a.pos_frac, neg_frac=a.neg_frac,
                          features=a.features, clr=a.clr, n_runs=a.n_runs, epochs=a.epochs,
                          profile_steps=a.profile_steps, batch_size=a.batch_size,
                          device=a.device, seed=a.seed, verbose=a.verbose)
    t0 = time.time()
    res = score_candidate_edges(expr, regulators, cfg, genes=genes)
    seconds = time.time() - t0

    X = expr[genes].to_numpy(dtype=float)
    src, dst = candidate_pairs(genes, regulators)  # same order as `res`
    y = np.array([(t, g) in truth_edges for t, g in zip(res.tf, res.target)])
    pos = res.train_positive.to_numpy()
    deg = np.bincount(src[pos], minlength=len(genes))[src].astype(float)
    tdeg = np.bincount(dst[pos], minlength=len(genes))[dst].astype(float)
    sp = res.abs_spearman.to_numpy()

    scores = {"scgrip": res.score.to_numpy()}
    if a.clr:
        scores["scgrip_noCLR"] = res.score_raw.to_numpy()
    scores["abs_spearman"] = sp
    scores["abs_pearson"] = abs_correlation(X, src, dst, "pearson")
    scores["clr_abs_spearman"] = clr_normalize(sp, src, dst, len(genes))
    scores["TF_degree"] = deg + 1e-3 * sp
    scores["degree_product"] = deg * (tdeg + 1) + 1e-3 * sp
    if a.genie3:
        print("  running GENIE3 baseline ...")
        scores["genie3"] = genie3_scores(X, genes, regulators, src, dst, a.genie3_trees, a.seed)

    m = {**meta, "label_scheme": cfg.label_scheme, "features": cfg.features, "clr": cfg.clr,
         "n_runs": cfg.n_runs, "n_genes": len(genes), "n_regulators": int(len(set(src))),
         "n_pairs": int(len(y)), "n_true_edges": int(y.sum()), "edge_density": float(y.mean()),
         "n_regulators_with_true_edges": int(len(set(src[y]))), "n_proxy_positives": int(pos.sum()),
         "seconds_scgrip": round(seconds, 1)}
    for name, s in scores.items():
        for k, v in existence_metrics(y, s, src).items():
            m[f"{name}_{k}"] = v

    out_dir.mkdir(parents=True, exist_ok=True)
    df = res[["tf", "target"]].assign(true_edge=y, proxy_positive=pos,
                                      **{f"score_{k}": v for k, v in scores.items()})
    df.sort_values("score_scgrip", ascending=False).to_csv(out_dir / "edge_scores.csv", index=False)
    df.rename(columns={"tf": "Gene1", "target": "Gene2", "score_scgrip": "EdgeWeight"})[
        ["Gene1", "Gene2", "EdgeWeight"]].sort_values("EdgeWeight", ascending=False) \
        .to_csv(out_dir / "rankedEdges.csv", sep="\t", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    return m


def _line(m: dict) -> str:
    names = [k[:-4] for k in m if k.endswith("_EPR")]
    return " | ".join(f"{n}: EPR={m[n + '_EPR']:.2f} AUPRCr={m[n + '_AUPRC_ratio']:.2f}" for n in names)


def summarize(rows, out: Path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)
    cols = [c for c in df.columns if c.endswith(("_EPR", "_AUPRC_ratio", "_within_TF_AUROC"))]
    if "group" in df and len(df) > 1:
        print("\nmedian per group:")
        print(df.groupby("group")[cols].median().round(3).T.to_string())


def cmd_simulated(a):
    root = Path(a.root).resolve()
    dirs = run_dirs(root, a.max_runs_per_group)
    print(f"found {len(dirs)} run directories under {root}")
    rows = []
    for d in dirs:
        gt = find_ground_truth(d, root)
        if gt is None:
            continue
        rel = d.relative_to(root)
        net = read_network(gt)
        expr = read_expression(d / "ExpressionData.csv")
        genes = list(expr.columns)
        m = evaluate(expr, genes, genes, set(zip(net.Gene1, net.Gene2)), a, Path(a.out) / rel,
                     {"dataset": str(rel), "group": rel.parts[0], "cells": int(expr.shape[0])})
        rows.append(m)
        print(f"  {rel}: {_line(m)}  ({m['n_true_edges']} true edges, {m['seconds_scgrip']}s)")
    summarize(rows, Path(a.out))


def cmd_experimental(a):
    data = Path(a.data)
    genes, var_tfs = select_genes(data, Path(a.tfs), a.num_genes)
    expr = read_expression(data / "ExpressionData.csv")
    genes = [g for g in genes if g in expr.columns]
    regs = sorted(var_tfs & set(genes))
    net = read_network(Path(a.network))
    print(f"{data.name}: {len(genes)} genes, {len(regs)} TFs, {expr.shape[0]} cells")
    m = evaluate(expr, genes, regs, set(zip(net.Gene1, net.Gene2)), a, Path(a.out),
                 {"dataset": data.name, "network": Path(a.network).name,
                  "num_genes": a.num_genes, "cells": int(expr.shape[0])})
    print(json.dumps(m, indent=2))
    summarize([m], Path(a.out))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--out", required=True)
        sp.add_argument("--label-scheme", default="per_tf", choices=["per_tf", "global"])
        sp.add_argument("--pos-frac", type=float, default=0.05)
        sp.add_argument("--neg-frac", type=float, default=0.5)
        sp.add_argument("--features", default="per_cell", choices=["per_cell", "profile"])
        sp.add_argument("--clr", action="store_true", help="CLR-normalise scGRIP scores")
        sp.add_argument("--n-runs", type=int, default=3)
        sp.add_argument("--epochs", type=int, default=30)
        sp.add_argument("--profile-steps", type=int, default=300)
        sp.add_argument("--batch-size", type=int, default=16)
        sp.add_argument("--device", default="cpu", help="cpu, cuda or mps")
        sp.add_argument("--genie3", action="store_true", help="also run a GENIE3 baseline")
        sp.add_argument("--genie3-trees", type=int, default=100)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--verbose", action="store_true")

    s = sub.add_parser("simulated", help="Curated / Synthetic runs")
    s.add_argument("--root", required=True)
    s.add_argument("--max-runs-per-group", type=int, default=0)
    common(s)
    s.set_defaults(fn=cmd_simulated)

    e = sub.add_parser("experimental", help="one scRNA-Seq dataset + a BEELINE reference network")
    e.add_argument("--data", required=True)
    e.add_argument("--network", required=True, help="reference, e.g. data/Networks/human/hESC-ChIP-seq-network.csv")
    e.add_argument("--tfs", required=True)
    e.add_argument("--num-genes", type=int, default=500)
    common(e)
    e.set_defaults(fn=cmd_experimental)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

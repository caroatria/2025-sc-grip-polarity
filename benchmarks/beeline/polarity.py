#!/usr/bin/env python3
"""
BEELINE benchmark: edge POLARITY (activation vs repression).

scGRIP gets the reference topology as its candidate network and predicts only
the sign of each edge. Scored on edges whose sign is known:
  * Curated / Synthetic (BoolODE): the `Type` column (+/-) of GroundTruthNetwork.csv
  * scRNA-Seq: BEELINE's networks are unsigned, so pass a signed resource with
    --signs (TRRUST raw TSV, DoRothEA/CollecTRI table).
A plain Spearman/Pearson correlation baseline is scored on the same edges.

    python polarity.py simulated --root data/BEELINE-data/inputs/Curated \
        --max-runs-per-group 1 --n-runs 2 --epochs 30 --out results_local/polarity_curated

    python polarity.py experimental --data data/BEELINE-data/inputs/scRNA-Seq/hESC \
        --network data/Networks/human/STRING-network.csv --tfs data/human-tfs.csv \
        --signs data/trrust_rawdata.human.tsv --add-sign-edges --out results_local/polarity_hESC

Outputs per dataset: predictions.csv, rankedEdges.csv (BEELINE format, signed
EdgeWeight = 2p-1), metrics.json; summary.csv over all datasets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scgrip import ScGRIP, ScGRIPConfig

from beeline_io import find_ground_truth, read_expression, read_network, run_dirs, select_genes, to_adjacency


def run_scgrip(expr: pd.DataFrame, adj: pd.DataFrame, a) -> pd.DataFrame:
    cfg = ScGRIPConfig(n_runs=a.n_runs, num_epochs=a.epochs, label_mode=a.label_mode,
                       batch_size=a.batch_size, device=a.device, seed=a.seed, verbose=a.verbose)
    model = ScGRIP(expr, adj, cfg)
    res = model.fit()
    z = model.expr_df
    z = (z - z.mean()) / (z.std() + 1e-6)
    ranks = z.rank()
    sp = [ranks[t].corr(ranks[g]) for t, g in zip(res.tf, res.target)]
    pe = [z[t].corr(z[g]) for t, g in zip(res.tf, res.target)]
    return pd.DataFrame({
        "Gene1": res.tf.values, "Gene2": res.target.values,
        "activation_prob": res.activation_score_mean.values,
        "activation_prob_std": res.activation_score_std.values,
        "spearman": np.nan_to_num(sp), "pearson": np.nan_to_num(pe),
        "supervised": res.supervised.values.astype(bool),
    })


def _safe(fn, y, s):
    return float(fn(y, s)) if len(np.unique(y)) == 2 else None


def score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    m = pred.merge(truth[truth.sign != 0], on=["Gene1", "Gene2"], how="inner")
    out = {"n_edges": int(len(pred)), "n_signed_edges": int(len(m)),
           "n_activating": int((m.sign == 1).sum()), "n_repressing": int((m.sign == -1).sum()),
           "n_supervised": int(pred.supervised.sum())}
    if len(m) == 0:
        return out
    y = (m.sign == 1).astype(int).to_numpy()
    for name, s, thr in (("scgrip", m.activation_prob.to_numpy(), 0.5),
                         ("spearman", m.spearman.to_numpy(), 0.0),
                         ("pearson", m.pearson.to_numpy(), 0.0)):
        out[f"{name}_AUROC"] = _safe(roc_auc_score, y, s)
        out[f"{name}_AUPRC_act"] = _safe(average_precision_score, y, s)
        out[f"{name}_AUPRC_rep"] = _safe(average_precision_score, 1 - y, -s)
        out[f"{name}_accuracy"] = float(((s > thr).astype(int) == y).mean())
    out["accuracy_always_activation"] = float(y.mean())
    u = m[~m.supervised]
    out["n_signed_unlabelled"] = int(len(u))
    if len(u):
        yu = (u.sign == 1).astype(int).to_numpy()
        out["scgrip_AUROC_unlabelled_edges"] = _safe(roc_auc_score, yu, u.activation_prob.to_numpy())
        out["spearman_AUROC_unlabelled_edges"] = _safe(roc_auc_score, yu, u.spearman.to_numpy())
    return out


def write_outputs(pred, truth, out_dir: Path, meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.merge(truth.rename(columns={"sign": "true_sign"}), on=["Gene1", "Gene2"], how="left") \
        .to_csv(out_dir / "predictions.csv", index=False)
    ranked = pred.assign(EdgeWeight=2 * pred.activation_prob - 1)[["Gene1", "Gene2", "EdgeWeight"]]
    ranked.reindex(ranked.EdgeWeight.abs().sort_values(ascending=False).index) \
        .to_csv(out_dir / "rankedEdges.csv", sep="\t", index=False)
    metrics = {**meta, **score(pred, truth)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def summarize(rows, out: Path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)
    if "group" in df and len(df) > 1:
        cols = [c for c in df.columns if c.endswith(("AUROC", "accuracy"))]
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
            print(f"  skip {d}: no ground truth")
            continue
        rel = d.relative_to(root)
        t0 = time.time()
        truth = read_network(gt)
        expr = read_expression(d / "ExpressionData.csv")
        pred = run_scgrip(expr, to_adjacency(truth), a)
        m = write_outputs(pred, truth, Path(a.out) / rel,
                          {"dataset": str(rel), "group": rel.parts[0], "cells": int(expr.shape[0]),
                           "seconds": round(time.time() - t0, 1)})
        rows.append(m)
        print(f"  {rel}: AUROC scGRIP={m.get('scgrip_AUROC')}  spearman={m.get('spearman_AUROC')}  "
              f"({m['n_activating']}+/{m['n_repressing']}-, {m['seconds']}s)")
    summarize(rows, Path(a.out))


def cmd_experimental(a):
    data = Path(a.data)
    genes, var_tfs = select_genes(data, Path(a.tfs), a.num_genes)
    expr = read_expression(data / "ExpressionData.csv")
    genes = [g for g in genes if g in expr.columns]
    expr = expr[genes]
    gs = set(genes)

    net = read_network(Path(a.network))
    net = net[net.Gene1.isin(gs) & net.Gene2.isin(gs) & net.Gene1.isin(var_tfs)][["Gene1", "Gene2", "sign"]]
    if a.signs:
        signs = read_network(Path(a.signs))
        net = net[["Gene1", "Gene2"]].merge(signs, on=["Gene1", "Gene2"], how="left").fillna({"sign": 0})
        if a.add_sign_edges:
            extra = signs[signs.Gene1.isin(var_tfs) & signs.Gene2.isin(gs) & (signs.sign != 0)]
            net = pd.concat([net, extra]).drop_duplicates(["Gene1", "Gene2"])
    print(f"{data.name}: {len(genes)} genes, {len(var_tfs)} TFs, {len(net)} candidate edges, "
          f"{int((net.sign != 0).sum())} with known sign")
    t0 = time.time()
    pred = run_scgrip(expr, to_adjacency(net), a)
    m = write_outputs(pred, net, Path(a.out),
                      {"dataset": data.name, "network": Path(a.network).name,
                       "signs": Path(a.signs).name if a.signs else None, "num_genes": a.num_genes,
                       "cells": int(expr.shape[0]), "seconds": round(time.time() - t0, 1)})
    print(json.dumps(m, indent=2))
    summarize([m], Path(a.out))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--out", required=True)
        sp.add_argument("--n-runs", type=int, default=5)
        sp.add_argument("--epochs", type=int, default=100)
        sp.add_argument("--label-mode", default="adaptive", choices=["binary", "ternary", "adaptive", "logic"])
        sp.add_argument("--batch-size", type=int, default=16)
        sp.add_argument("--device", default="cpu", help="cpu, cuda or mps")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--verbose", action="store_true")

    s = sub.add_parser("simulated", help="Curated / Synthetic BoolODE runs (signed ground truth)")
    s.add_argument("--root", required=True)
    s.add_argument("--max-runs-per-group", type=int, default=0, help="replicates per folder (0 = all)")
    common(s)
    s.set_defaults(fn=cmd_simulated)

    e = sub.add_parser("experimental", help="one scRNA-Seq dataset + BEELINE network + signed resource")
    e.add_argument("--data", required=True, help="e.g. data/BEELINE-data/inputs/scRNA-Seq/hESC")
    e.add_argument("--network", required=True, help="BEELINE network CSV (candidate topology)")
    e.add_argument("--tfs", required=True, help="human-tfs.csv or mouse-tfs.csv")
    e.add_argument("--signs", help="signed TF-target table (TRRUST raw TSV, DoRothEA/CollecTRI CSV)")
    e.add_argument("--add-sign-edges", action="store_true",
                   help="also add signed edges missing from --network to the candidate set")
    e.add_argument("--num-genes", type=int, default=500)
    common(e)
    e.set_defaults(fn=cmd_experimental)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

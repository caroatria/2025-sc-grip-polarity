"""Score edge-existence predictions against the ENCODE ChIP-seq benchmark.

Usage
-----
    # score your own predictions (TSV/CSV with columns tf, target, score)
    python score_edge_existence.py --predictions my_preds.tsv --cell K562

    # built-in |Pearson r| baseline computed from the benchmark expression
    python score_edge_existence.py --baseline correlation --cell K562

Every run also reports the swapped-cell-type control: the same predictions
scored against the *other* cell line's labels over the same edges. A method
that captures cell-type-specific regulation should score higher on the matched
labels than on the swapped ones; a method driven by cell-type-invariant signal
(sequence, gene identity, generic co-expression) scores about the same on both.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# benchmark data (expression, labels) live in ./benchmark/ (not versioned; see README.md)
DATA = os.environ.get("SCGRIP_ENCODE_DATA", os.path.join(HERE, "benchmark"))
CELLS = ["K562", "HepG2"]


def auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    """Mann-Whitney U form, with ties given mid-ranks."""
    if len(set(y)) < 2:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float | None:
    """Area under the precision-recall curve, the step-wise (AP) estimator."""
    if y.sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def early_precision(y: np.ndarray, s: np.ndarray, k: int | None = None) -> tuple:
    """Precision in the top-k ranked edges (k = number of positives by default),
    and the ratio of that to the prevalence (a random ranker scores 1.0)."""
    n_pos = int(y.sum())
    k = k or n_pos
    if k == 0:
        return None, None
    order = np.argsort(-s, kind="mergesort")[:k]
    ep = float(y[order].mean())
    prev = n_pos / len(y)
    return ep, (ep / prev if prev > 0 else None)


def correlation_scores(cell: str, edges: pd.DataFrame) -> np.ndarray:
    """|Pearson r| between TF and target across cells, as a baseline predictor."""
    expr = pd.read_csv(os.path.join(DATA, f"expression_{cell}.csv"), index_col=0)
    X = expr.to_numpy(dtype=np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    col = {g: i for i, g in enumerate(expr.columns)}
    n = X.shape[0]
    out = np.zeros(len(edges))
    for i, (tf, tgt) in enumerate(zip(edges.tf, edges.target)):
        if tf in col and tgt in col:
            out[i] = abs(float(X[:, col[tf]] @ X[:, col[tgt]] / n))
    return out


def score(y: np.ndarray, s: np.ndarray) -> dict:
    ep, epr = early_precision(y, s)
    return {"n_edges": int(len(y)), "n_positive": int(y.sum()),
            "prevalence": round(float(y.mean()), 4),
            "auroc": None if auroc(y, s) is None else round(auroc(y, s), 4),
            "auprc": None if average_precision(y, s) is None else round(average_precision(y, s), 4),
            "early_precision": None if ep is None else round(ep, 4),
            "early_precision_ratio": None if epr is None else round(epr, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=CELLS, required=True,
                    help="cell line whose expression the predictions were made from")
    ap.add_argument("--predictions", help="TSV/CSV with columns tf, target, score")
    ap.add_argument("--baseline", choices=["correlation"],
                    help="use a built-in baseline instead of --predictions")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--out", default=None, help="write the results table here")
    a = ap.parse_args()

    labels = {c: pd.read_csv(os.path.join(DATA, f"labels_{c}.tsv"), sep="\t") for c in CELLS}
    key = ["tf", "target"]

    if a.baseline == "correlation":
        base = labels[a.cell][key].copy()
        base["score"] = correlation_scores(a.cell, base)
        preds, score_col, name = base, "score", "|pearson r|"
    else:
        if not a.predictions:
            ap.error("pass --predictions or --baseline")
        sep = "\t" if a.predictions.endswith((".tsv", ".txt")) else ","
        preds = pd.read_csv(a.predictions, sep=sep)
        missing = [c for c in key + [a.score_col] if c not in preds.columns]
        if missing:
            ap.error(f"predictions file is missing column(s): {missing}")
        score_col, name = a.score_col, os.path.basename(a.predictions)

    other = [c for c in CELLS if c != a.cell][0]
    disc = pd.read_csv(os.path.join(DATA, "discriminative_edges.tsv"), sep="\t")
    disc_pairs = set(map(tuple, disc[key].to_numpy()))

    rows = []
    for truth_cell in [a.cell, other]:
        m = labels[truth_cell].merge(preds[key + [score_col]], on=key, how="inner")
        if len(m) == 0:
            raise SystemExit("no overlap between predictions and benchmark edges — "
                             "check that tf/target use gene symbols")
        y = m.label.to_numpy(dtype=int)
        s = m[score_col].to_numpy(dtype=float)
        comparison = "matched" if truth_cell == a.cell else "swapped"
        meta = {"predictor": name, "expression_from": a.cell, "labels_from": truth_cell,
                "comparison": comparison, "coverage": round(len(m) / len(labels[truth_cell]), 3)}

        rows.append({**meta, "view": "all_edges", **score(y, s)})

        # per-TF, then averaged: removes the effect of how broadly each factor binds
        per_tf = [auroc(g.label.to_numpy(dtype=int), g[score_col].to_numpy(dtype=float))
                  for _, g in m.groupby("tf") if g.label.nunique() == 2]
        per_tf = [v for v in per_tf if v is not None]
        rows.append({**meta, "view": "per_tf_mean_auroc", "n_edges": len(m),
                     "n_positive": int(y.sum()), "prevalence": round(float(y.mean()), 4),
                     "auroc": round(float(np.mean(per_tf)), 4) if per_tf else None,
                     "auprc": None, "early_precision": None, "early_precision_ratio": None})

        # edges whose label differs between the two lines: chance level is 0.5 AUROC
        # for any predictor without cell-type-specific information
        sub = m[[tuple(x) in disc_pairs for x in m[key].to_numpy()]]
        if len(sub) and sub.label.nunique() == 2:
            rows.append({**meta, "view": "discriminative_edges",
                         **score(sub.label.to_numpy(dtype=int),
                                 sub[score_col].to_numpy(dtype=float))})

    res = pd.DataFrame(rows)[["predictor", "expression_from", "labels_from", "comparison",
                              "view", "n_edges", "coverage", "n_positive", "prevalence",
                              "auroc", "auprc", "early_precision", "early_precision_ratio"]]
    print(res.to_string(index=False))

    def get(view, comp, col="auroc"):
        r = res[(res.view == view) & (res.comparison == comp)]
        return None if r.empty else r.iloc[0][col]

    print("\n--- headline numbers ---")
    if get("all_edges", "matched") is not None and get("all_edges", "swapped") is not None:
        print(f"all edges:  matched AUROC {get('all_edges','matched'):.4f} vs "
              f"swapped {get('all_edges','swapped'):.4f} "
              f"(difference {get('all_edges','matched') - get('all_edges','swapped'):+.4f})")
    if get("discriminative_edges", "matched") is not None:
        v = get("discriminative_edges", "matched")
        print(f"cell-type-discriminative edges: AUROC {v:.4f} "
              f"({'above' if v > 0.5 else 'at or below'} the 0.5 chance level -- this is the "
              f"only view a cell-type-invariant predictor cannot score above chance on)")
    if a.out:
        res.to_csv(a.out, sep="\t", index=False)
        print(f"written {a.out}")


if __name__ == "__main__":
    main()

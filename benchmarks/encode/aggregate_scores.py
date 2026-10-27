"""Score every method's predictions and assemble the comparison table.

Reuses the benchmark's own metric functions (score_edge_existence.auroc /
.score) so the numbers are identical to what a user gets from the CLI.
Adds a 20-seed random band to establish the empirical chance level per view.
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "benchmark")
PRED = os.path.join(HERE, "predictions")
CELLS = ["K562", "HepG2"]
KEY = ["tf", "target"]

spec = importlib.util.spec_from_file_location(
    "see", os.path.join(HERE, "score_edge_existence.py"))
see = importlib.util.module_from_spec(spec)
spec.loader.exec_module(see)

labels = {c: pd.read_csv(os.path.join(BENCH, f"labels_{c}.tsv"), sep="\t") for c in CELLS}
disc_pairs = set(map(tuple, pd.read_csv(
    os.path.join(BENCH, "discriminative_edges.tsv"), sep="\t")[KEY].to_numpy()))


def rows_for(preds, expr_cell, name):
    """Replicates the scorer's three views x two comparisons."""
    out = []
    other = [c for c in CELLS if c != expr_cell][0]
    for truth_cell in [expr_cell, other]:
        m = labels[truth_cell].merge(preds[KEY + ["score"]], on=KEY, how="inner")
        y = m.label.to_numpy(dtype=int)
        s = m.score.to_numpy(dtype=float)
        meta = {"method": name, "expression_from": expr_cell,
                "labels_from": truth_cell,
                "comparison": "matched" if truth_cell == expr_cell else "swapped"}
        out.append({**meta, "view": "all_edges", **see.score(y, s)})

        per_tf = [see.auroc(g.label.to_numpy(dtype=int), g.score.to_numpy(dtype=float))
                  for _, g in m.groupby("tf") if g.label.nunique() == 2]
        per_tf = [v for v in per_tf if v is not None]
        out.append({**meta, "view": "per_tf_mean_auroc", "n_edges": len(m),
                    "n_positive": int(y.sum()),
                    "prevalence": round(float(y.mean()), 4),
                    "auroc": round(float(np.mean(per_tf)), 4) if per_tf else None,
                    "auprc": None, "early_precision": None,
                    "early_precision_ratio": None})

        sub = m[[tuple(x) in disc_pairs for x in m[KEY].to_numpy()]]
        out.append({**meta, "view": "discriminative_edges",
                    **see.score(sub.label.to_numpy(dtype=int),
                                sub.score.to_numpy(dtype=float))})

        ann = m[m.tf.isin(ANNOT_TFS)]
        out.append({**meta, "view": "motif_annotated_tfs",
                    **see.score(ann.label.to_numpy(dtype=int),
                                ann.score.to_numpy(dtype=float))})
    return out


METHODS = ["pearson", "spearman", "mutual_info", "partial_correlation",
           "genie3", "grnboost2", "scenic",
           "motif_cistarget", "motif_ctscore", "motif_pwm"]

# TFs that carry a cisTarget motif annotation at all. No motif-aware method can
# address the rest by construction, so every method is also scored on this subset.
ANNOT_TFS = set(pd.read_csv(os.path.join(PRED, "motif_cistarget_detail_K562.tsv"),
                            sep="\t").query("annotated").tf.unique())


def seconds_for(meta):
    if "seconds" in meta:
        return meta["seconds"]
    if "grnboost2_seconds" in meta:
        return round(meta["grnboost2_seconds"] + meta["prune_seconds"], 1)
    return None

rows, runtimes = [], {}
for cell in CELLS:
    for m in METHODS:
        p = os.path.join(PRED, f"preds_{m}_{cell}.tsv")
        if not os.path.exists(p):
            print(f"MISSING {p}", flush=True)
            continue
        rows += rows_for(pd.read_csv(p, sep="\t"), cell, m)
        runtimes[(m, cell)] = seconds_for(json.load(
            open(os.path.join(PRED, f"meta_{m}_{cell}.json"))))

# ---- empirical chance band: 20 random-score replicates ---------------------
band = []
for cell in CELLS:
    tmpl = labels[cell][KEY].copy()
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        tmpl["score"] = rng.random(len(tmpl))
        for r in rows_for(tmpl, cell, "random"):
            band.append({**r, "seed": seed})
band = pd.DataFrame(band)
chance = (band.groupby(["expression_from", "comparison", "view"])["auroc"]
          .agg(auroc_mean="mean",
               auroc_lo=lambda v: float(np.percentile(v, 2.5)),
               auroc_hi=lambda v: float(np.percentile(v, 97.5)))
          .round(4).reset_index())

res = pd.DataFrame(rows)
res["seconds"] = [runtimes.get((m, c)) for m, c in
                  zip(res.method, res.expression_from)]
res = res[["method", "expression_from", "labels_from", "comparison", "view",
           "n_edges", "n_positive", "prevalence", "auroc", "auprc",
           "early_precision", "early_precision_ratio", "seconds"]]
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
res.to_csv(os.path.join(HERE, "results", "grn_method_comparison.tsv"), sep="\t", index=False)
chance.to_csv(os.path.join(HERE, "results", "random_chance_band.tsv"), sep="\t", index=False)

print(f"methods scored: {res.method.nunique()} | rows: {len(res)}")
print("\n=== AUROC, matched labels ===")
piv = res[res.comparison == "matched"].pivot_table(
    index="method", columns=["expression_from", "view"], values="auroc")
print(piv.to_string())
print("\n=== chance band (random, 20 seeds) ===")
print(chance[chance.comparison == "matched"].to_string(index=False))
print("\n=== matched minus swapped AUROC (cell-type specificity) ===")
d = res.pivot_table(index=["method", "expression_from", "view"],
                    columns="comparison", values="auroc")
d["gap"] = (d["matched"] - d["swapped"]).round(4)
print(d.reset_index().pivot_table(index="method",
                                  columns=["expression_from", "view"],
                                  values="gap").to_string())

"""Independent PWM scan of hg38 promoters -- the motif-only baseline that does not
go through cisTarget at all.

This replicates the step sc-GRIP performs with FIMO: take the TF's position
weight matrices, scan the target gene's promoter on both strands, and score the
pair by the best log-odds hit. Matrices are JASPAR 2024 CORE vertebrates
non-redundant; promoters are TSS +/- 1 kb pulled from Ensembl/GRCh38, matching
the +/- 1 kb window the benchmark itself uses to call a peak promoter-proximal.
Background is uniform (0.25 per base) and a 0.25 pseudocount is added per cell.

Because it uses neither expression nor cisTarget's rankings, agreement with
motif_cistarget is evidence that the motif-only floor is a property of the
sequence and not of cisTarget's gene-centric aggregation.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_grn_methods as R

PWM_DIR = "build/pwm"
JASPAR = os.path.join(PWM_DIR, "JASPAR2024_CORE_vertebrates_nr_pfms.jaspar")
PROMOTERS = os.path.join(PWM_DIR, "promoters.json")
OUT = "predictions"
BG = 0.25
PSEUDO = 0.25
IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def read_jaspar(path):
    """-> list of (matrix_id, name, pfm array of shape (L, 4))."""
    out, mid, name, rows = [], None, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if mid is not None:
                    out.append((mid, name, np.array(rows, dtype=float).T))
                parts = line[1:].split("\t")
                mid, name, rows = parts[0], parts[1] if len(parts) > 1 else parts[0], []
            elif line.strip():
                nums = line[line.index("[") + 1:line.rindex("]")].split()
                rows.append([float(x) for x in nums])
    if mid is not None:
        out.append((mid, name, np.array(rows, dtype=float).T))
    return out


def log_odds(pfm):
    p = (pfm + PSEUDO) / (pfm.sum(axis=1, keepdims=True) + 4 * PSEUDO)
    return np.log2(p / BG)


def tf_matrices(matrices, tfs):
    """Map benchmark TF symbols to their JASPAR matrices (case-insensitive; a
    dimer matrix such as FOS::JUN is assigned to each constituent factor)."""
    upper = {t.upper(): t for t in tfs}
    out = {t: [] for t in tfs}
    for mid, name, pfm in matrices:
        for part in name.split("::"):
            key = part.strip().upper()
            if key in upper:
                out[upper[key]].append((mid, name, pfm))
    return {t: v for t, v in out.items() if v}


def onehot(seq):
    m = np.zeros((len(seq), 4), dtype=np.float32)
    for i, ch in enumerate(seq):
        j = IDX.get(ch)
        if j is not None:
            m[i, j] = 1.0
    return m


def best_hit(oh, lom):
    """Max log-odds over all offsets, both strands."""
    L = lom.shape[0]
    n = oh.shape[0] - L + 1
    if n <= 0:
        return np.nan
    best = -np.inf
    for mat in (lom, lom[::-1, ::-1]):           # forward, reverse complement
        s = np.zeros(n, dtype=np.float32)
        for l in range(L):
            s += oh[l:l + n] @ mat[l].astype(np.float32)
        best = max(best, float(s.max()))
    return best


def scan(tf2mats, seqs):
    """-> DataFrame TFs x genes of best log-odds hit."""
    genes = sorted(seqs)
    oh = {g: onehot(seqs[g]) for g in genes}
    res = {}
    for tf, mats in tf2mats.items():
        loms = [log_odds(pfm) for _, _, pfm in mats]
        res[tf] = [max(best_hit(oh[g], lom) for lom in loms) for g in genes]
        print(f"  {tf}: {len(mats)} matrices", flush=True)
    return pd.DataFrame(res, index=genes).T


def run(cells=("K562", "HepG2")):
    os.makedirs(OUT, exist_ok=True)
    matrices = read_jaspar(JASPAR)
    seqs = json.load(open(PROMOTERS))
    _, tfs, _ = R.load(cells[0])
    tf2mats = tf_matrices(matrices, tfs)
    print(f"JASPAR matrices: {len(matrices)} | benchmark TFs with a matrix: "
          f"{len(tf2mats)}/{len(tfs)} | promoters: {len(seqs)}", flush=True)

    S = scan(tf2mats, seqs)
    S.to_csv(os.path.join(PWM_DIR, "pwm_best_hit.tsv"), sep="\t")

    s = S.stack()
    s.index.names = ["tf", "target"]
    for cell in cells:
        _, tfs_c, cand = R.load(cell)
        out = cand.copy()
        out["score"] = out.join(s.rename("v"), on=["tf", "target"])["v"].to_numpy()
        out["has_matrix"] = out["tf"].isin(tf2mats).to_numpy()
        out["has_promoter"] = out["target"].isin(seqs).to_numpy()
        # unscorable pairs rank below every scored pair
        floor = out["score"].min() - 1.0
        out["score"] = out["score"].fillna(floor)
        out[["tf", "target", "score"]].to_csv(
            os.path.join(OUT, f"preds_motif_pwm_{cell}.tsv"), sep="\t", index=False)
        out.to_csv(os.path.join(OUT, f"motif_pwm_detail_{cell}.tsv"), sep="\t", index=False)
        meta = {"method": "motif_pwm", "cell": cell, "n_edges": int(len(out)),
                "n_tfs": len(tfs_c), "n_tfs_with_matrix": len(tf2mats),
                "tfs_without_matrix": sorted(set(tfs_c) - set(tf2mats)),
                "n_promoters": len(seqs),
                "n_edges_scorable": int((out["has_matrix"] & out["has_promoter"]).sum()),
                "window_bp": 2001, "source": "JASPAR2024 CORE vertebrates nr; Ensembl GRCh38"}
        with open(os.path.join(OUT, f"meta_motif_pwm_{cell}.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        print("  " + json.dumps({k: v for k, v in meta.items()
                                 if k != "tfs_without_matrix"}), flush=True)


if __name__ == "__main__":
    run()

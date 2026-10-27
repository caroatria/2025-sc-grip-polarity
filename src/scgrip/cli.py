"""
Command-line interface
======================

    scgrip polarity  --expression expr.csv --adjacency tf_targets.csv --out polarity.csv
    scgrip existence --expression expr.csv --tfs tfs.txt --out edges.csv

`expr.csv` is cells x genes (first column = cell id). `tf_targets.csv` is a
TF (rows) x target (columns) 0/1 matrix of candidate edges. `tfs.txt` lists
candidate regulators, one per line (or a CSV whose first column is the TF).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_tfs(path: str) -> list[str]:
    p = Path(path)
    if p.suffix in (".csv", ".tsv"):
        return pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",").iloc[:, 0].astype(str).tolist()
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def _polarity(a):
    from .data import read_adjacency_matrix, read_expression_matrix
    from .inference import ScGRIP, ScGRIPConfig

    cfg = ScGRIPConfig(n_runs=a.n_runs, num_epochs=a.epochs, label_mode=a.label_mode,
                       device=a.device, seed=a.seed, verbose=not a.quiet)
    expr = read_expression_matrix(a.expression, adata_path=a.h5ad)
    model = ScGRIP(expr, read_adjacency_matrix(a.adjacency), config=cfg)
    res = model.fit()
    if not a.quiet:
        model.summary(res)
    res.to_csv(a.out, index=False)
    print(f"wrote {a.out}")


def _existence(a):
    from .data import read_expression_matrix
    from .existence import ExistenceConfig, score_candidate_edges

    cfg = ExistenceConfig(label_scheme=a.label_scheme, features=a.features, clr=a.clr,
                          n_runs=a.n_runs, epochs=a.epochs, device=a.device, seed=a.seed,
                          verbose=not a.quiet)
    expr = read_expression_matrix(a.expression, adata_path=a.h5ad)
    res = score_candidate_edges(expr, _read_tfs(a.tfs), cfg)
    res.sort_values("score", ascending=False).to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(res)} pairs)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="scgrip", description="sc-GRIP: label-free TF-target scoring from scRNA-seq")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--expression", help="cells x genes CSV")
        sp.add_argument("--h5ad", help=".h5ad file instead of --expression (needs scgrip[h5ad])")
        sp.add_argument("--out", required=True)
        sp.add_argument("--device", default="cpu", help="cpu, cuda or mps")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--quiet", action="store_true")

    s = sub.add_parser("polarity", help="score activation vs repression for edges of a candidate network")
    common(s)
    s.add_argument("--adjacency", required=True, help="TF x target 0/1 CSV")
    s.add_argument("--n-runs", type=int, default=5)
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--label-mode", default="adaptive", choices=["binary", "ternary", "adaptive", "logic"])
    s.set_defaults(fn=_polarity)

    e = sub.add_parser("existence", help="score every TF -> gene pair for edge existence")
    common(e)
    e.add_argument("--tfs", required=True, help="candidate regulators (txt, one per line, or csv)")
    e.add_argument("--label-scheme", default="per_tf", choices=["per_tf", "global"])
    e.add_argument("--features", default="per_cell", choices=["per_cell", "profile"])
    e.add_argument("--clr", action="store_true")
    e.add_argument("--n-runs", type=int, default=3)
    e.add_argument("--epochs", type=int, default=30)
    e.set_defaults(fn=_existence)

    a = p.parse_args(argv)
    if not (a.expression or a.h5ad):
        p.error("one of --expression or --h5ad is required")
    a.fn(a)


if __name__ == "__main__":
    main()

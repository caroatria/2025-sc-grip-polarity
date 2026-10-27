# ENCODE ChIP-seq benchmark (K562 / HepG2)

Edge-existence ground truth from ENCODE TF ChIP-seq, with matched scRNA-seq for two human
cell lines, plus ten established GRN inference methods scored on it. Full results and
interpretation: [`GRN_BASELINES.md`](GRN_BASELINES.md).

The same 14,318 candidate TF–gene pairs (42 TFs, 1,512 genes) are labelled independently in
both lines; 4,726 pairs have different labels in the two lines. Scoring a predictor run on K562
expression against K562 labels (*matched*) and HepG2 labels (*swapped*) is an exact test of
cell-type specificity. Labels: bound = ChIP summit within 1 kb of the TSS, unbound = none within
50 kb, ambiguous pairs dropped; candidate space balanced per TF. Expression: GEO GSE90063 (K562)
and GSE281160 (HepG2), 4,000 cells each. Parameters: `benchmark_manifest.json`.

| file | what it is |
|---|---|
| `score_edge_existence.py` | scorer: AUROC, AUPRC, early precision; all pairs, per-TF mean, discriminative subset; matched and swapped |
| `run_grn_methods.py` | Pearson, Spearman, partial correlation, mutual information, GENIE3, GRNBoost2, random |
| `run_scenic.py` | SCENIC (GRNBoost2 modules pruned by cisTarget) |
| `run_motif_only.py` | two expression-free cisTarget predictors |
| `run_pwm_scan.py` | expression-free JASPAR 2024 promoter PWM scan |
| `aggregate_scores.py` | scores every method into `results/grn_method_comparison.tsv` |
| `make_figure.py` | renders `results/grn_method_comparison.png` |
| `results/` | comparison table and 20-seed random-chance band |

## Data (not versioned)

The scripts expect three folders next to them:

- `benchmark/` (105 MB): `expression_{K562,HepG2}.csv`, `candidate_adjacency.csv`,
  `labels_{K562,HepG2}.tsv`, `discriminative_edges.tsv`, `full_labels_*.tsv.gz`
- `predictions/` (13 MB): per-method edge scores and run metadata (GENIE3 alone takes ~31 min
  per cell line to regenerate)
- `build/` (990 MB): cisTarget databases, JASPAR matrices and promoter sequences; re-fetched
  from resources.aertslab.org and jaspar.elixir.no by the motif scripts

<!-- TODO before publishing: deposit benchmark/ and predictions/ (e.g. on Zenodo) and link them here. -->
The `benchmark/` and `predictions/` archives are available from the authors on request until
they are deposited publicly.

## Reproduce

```bash
pip install -e "../..[encode,plot]"
python run_grn_methods.py --cell all --method all   # GENIE3 is ~31 min per line
python run_scenic.py      --cell all
python run_motif_only.py  --cell all
python run_pwm_scan.py
python aggregate_scores.py
python make_figure.py
```

## Score scGRIP

```bash
# --tfs reads the first column of the adjacency CSV (the 42 TFs)
scgrip existence --expression benchmark/expression_K562.csv \
  --tfs benchmark/candidate_adjacency.csv --out scgrip_K562_all_pairs.csv
python score_edge_existence.py --predictions scgrip_K562_all_pairs.csv --cell K562
```

The scorer reads columns `tf, target, score` and evaluates only the benchmark's candidate
pairs. Report both cell lines: expression-based and sequence-based evidence clear the chance
band in *opposite* lines, so either line alone supports the opposite conclusion.

## Before quoting a number

1. **The bar is a sequence-only scan, not AUROC 0.5.** An expression-free JASPAR promoter scan
   reaches AUROC 0.556 on K562 (0.581 on motif-annotated TFs), beating every expression method.
   When scGRIP's candidate network comes from motif scanning, that is its own input, unmodelled.
2. **ChIP occupancy is binding, not regulation**, and polarity is not part of this benchmark.

# BEELINE benchmark

Validation of scGRIP on [BEELINE](https://github.com/Murali-group/Beeline)
(Pratapa et al., *Nature Methods* 2020): BoolODE-simulated curated and synthetic
networks, and seven experimental scRNA-seq datasets with ChIP-seq and STRING reference
networks. scGRIP is always run **label-free**; the reference networks are used only for scoring.

| script | what it does |
|---|---|
| `download.py` | downloads BEELINE (Zenodo, DOI 10.5281/zenodo.3378975) and TRRUST v2 into `data/` |
| `polarity.py` | edge sign (activation vs repression) on a given candidate network |
| `existence.py` | edge existence for every TF → gene pair, with correlation, GENIE3 and hub baselines |
| `beeline_io.py` | readers for BEELINE expression, networks and gene selection |
| `RESULTS.md` | results so far |

## Setup

```bash
pip install -e ".[dev]"          # from the repository root
cd benchmarks/beeline
python download.py               # ~200 MB into data/ (not versioned)
```

Run the scripts from this folder. Gene symbols are upper-cased everywhere for matching.

## Polarity

```bash
# curated models, quick: one replicate per model
python polarity.py simulated --root data/BEELINE-data/inputs/Curated \
  --max-runs-per-group 1 --n-runs 2 --epochs 30 --out results_local/polarity_curated

# experimental: BEELINE network as candidate topology, TRRUST for the signs
python polarity.py experimental --data data/BEELINE-data/inputs/scRNA-Seq/hESC \
  --network data/Networks/human/STRING-network.csv --tfs data/human-tfs.csv \
  --signs data/trrust_rawdata.human.tsv --add-sign-edges --out results_local/polarity_hESC
```

Scored on edges with a known sign: AUROC, AUPRC (activation and repression), accuracy, against
Spearman/Pearson correlation and an "always activation" baseline. VSC has only repressive edges,
so its AUROC is undefined; use accuracy.

## Edge existence

```bash
python existence.py simulated --root data/BEELINE-data/inputs/Curated \
  --max-runs-per-group 1 --genie3 --out results_local/existence_curated

python existence.py experimental --data data/BEELINE-data/inputs/scRNA-Seq/hESC \
  --network data/Networks/human/hESC-ChIP-seq-network.csv --tfs data/human-tfs.csv \
  --out results_local/existence_hESC
```

Options: `--label-scheme per_tf|global`, `--features per_cell|profile`, `--clr`, `--n-runs`,
`--epochs`, `--device cpu|cuda|mps`. Mouse datasets (mESC, mDC, mHSC-E/GM/L) use
`data/Networks/mouse/...` and `data/mouse-tfs.csv`.

Read `EPR@100` and `within_TF_AUROC` next to the overall AUPRC ratio and EPR, and compare with
the `TF_degree` / `degree_product` hub baselines: BEELINE's ChIP-seq references cover only a few
TFs (34 of 410 for hESC), so overall scores largely reward ranking those TFs first.

## Runtime

The per-cell model builds one graph per cell. On a 2-core machine, hESC (758 cells, 372,690
pairs) takes about 50 minutes per training run at 30 epochs; `--features profile` takes about
1 minute per run. GENIE3 on hESC takes about 30 minutes with 50 trees. Start with
`--max-runs-per-group 1` on the curated models, and use `--device mps` or `cuda` if available.

## Outputs

Per dataset: `edge_scores.csv` or `predictions.csv` (all pairs, all methods), `rankedEdges.csv`
(BEELINE format, usable with BEELINE's own evaluators), `metrics.json`; `summary.csv` over datasets.
`results_local/` and `data/` are git-ignored.

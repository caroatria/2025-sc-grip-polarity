# sc-GRIP

**Single-cell Gene Regulation Interaction Polarity.** A graph-convolutional framework that
scores transcription factor (TF) → target gene interactions from single-cell RNA-seq data
**without training labels**, so it can be applied to non-model species where no curated
regulatory network or experimental ground truth exists.

> **This implementation has been substantially redesigned since the ICDM 2025 paper.**
> The current version on `main` is the one we recommend using. The code exactly as published
> with the paper is preserved under the Git tag
> [`icdm-2025`](https://github.com/caroatria/2025-sc-grip-polarity/tree/icdm-2025).
> The current code does **not** reproduce the numbers reported in the paper; see
> [What changed since ICDM 2025](#what-changed-since-icdm-2025).

## What it does

sc-GRIP builds one graph per cell, with genes as nodes and the cell's expression as node
features, encodes it with a two-layer GCN, and scores TF → target pairs with a bilinear decoder.
Training labels are *proxy* labels derived from the expression data itself, never from a
reference network. Two tasks are supported:

- **Polarity** (`scgrip.ScGRIP`, `scgrip polarity`): given a candidate network (e.g. from motif
  scanning or transferred from a model species via orthologs), score each edge as activating or
  repressing.
- **Edge existence** (`scgrip.existence`, `scgrip existence`): score every candidate
  TF → gene pair for whether an edge exists, with proxy labels chosen per TF so that hub genes
  do not dominate.

## Installation

```bash
git clone https://github.com/caroatria/2025-sc-grip-polarity.git
cd 2025-sc-grip-polarity
pip install -e .                 # core
pip install -e ".[h5ad,plot,dev]" # optional: .h5ad input, plotting, tests
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0 and PyTorch Geometric ≥ 2.3. For a GPU build of PyTorch,
install it first following [pytorch.org](https://pytorch.org/get-started/locally/).

## Inputs

1. **Expression**: cells × genes matrix, log-normalised (CSV with cell ids in the first column,
   or `.h5ad` with `scgrip[h5ad]`).
2. **Candidate TFs / network**:
   - polarity: a TF (rows) × target (columns) 0/1 matrix of candidate edges;
   - existence: a list of candidate regulators.

Gene identifiers must match between the two (e.g. both gene symbols).

## Quick start

Command line:

```bash
scgrip polarity  --expression expr.csv --adjacency tf_targets.csv --out polarity.csv
scgrip existence --expression expr.csv --tfs tfs.txt --out edges.csv
```

Python:

```python
from scgrip import ScGRIP, ScGRIPConfig, read_expression_matrix, read_adjacency_matrix

expr = read_expression_matrix("expr.csv")
model = ScGRIP(expr, read_adjacency_matrix("tf_targets.csv"), ScGRIPConfig(n_runs=5, num_epochs=100))
polarity = model.fit()          # one row per edge: activation score (mean/std over runs), proxy label, ...
model.summary(polarity)

from scgrip import ExistenceConfig, score_candidate_edges

edges = score_candidate_edges(expr, regulators=["Sox2", "Pou5f1", "Nanog"],
                              config=ExistenceConfig(label_scheme="per_tf"))
```

A runnable example on synthetic data: `python examples/quickstart.py`.

## Non-model species

`scgrip.orthology` translates predicted edges into a model species' gene symbols via an
ortholog table and checks them against a curated network (strict name matching, plus an
optional family-level match reported alongside it). `scgrip.evaluate` adds permutation and
degree-preserving rewiring tests for whether the conserved overlap is larger than expected by
chance.

## Validation

| benchmark | ground truth | folder |
|---|---|---|
| BEELINE (Pratapa et al. 2020): curated BoolODE models and 7 scRNA-seq datasets | signed BoolODE networks; ChIP-seq and STRING networks | [`benchmarks/beeline`](benchmarks/beeline) |
| ENCODE ChIP-seq, K562 / HepG2, with ten baseline GRN methods | promoter-proximal ChIP-seq occupancy | [`benchmarks/encode`](benchmarks/encode) |

Results so far are in [`benchmarks/beeline/RESULTS.md`](benchmarks/beeline/RESULTS.md) and
[`benchmarks/encode/GRN_BASELINES.md`](benchmarks/encode/GRN_BASELINES.md). In short: on polarity,
scGRIP does not yet exceed the correlation it is trained on; on edge existence, per-TF proxy labels
give the best within-TF target ranking among the methods tested on two BEELINE reference networks,
while GENIE3 and plain correlation remain better at the top of the ranked list.

## What changed since ICDM 2025

| | ICDM 2025 (tag `icdm-2025`) | current (`main`) |
|---|---|---|
| code | single script `sc-GRIP.py` + notebook | installable package `scgrip`, CLI, tests, benchmarks |
| graph construction | TF/target matrix positions used as node indices | edges mapped to the correct gene nodes; self-loops dropped |
| proxy labels | fixed threshold (Pearson > 0.5) | `adaptive` (dataset quantiles, default), `binary`, `ternary`, `logic` (combinatorial Boolean rules) |
| training | batch size 1, BCE on sigmoid outputs | mini-batches, logits + `BCEWithLogitsLoss`, decoder scores only the directed TF → target edges, reverse edges used for message passing only |
| tasks | polarity | polarity and edge existence (link prediction with per-TF proxy labels, optional CLR) |
| preprocessing | – | per-gene z-scoring, cell-type restriction, pseudobulking |
| non-model species | – | ortholog translation, conservation and rewiring tests |
| validation | TRRUST, mouse/human datasets | BEELINE and ENCODE benchmarks with correlation, GENIE3 and hub baselines |

See [`CHANGELOG.md`](CHANGELOG.md) for details. To use the published version:

```bash
git checkout icdm-2025
```

## Citation

If you use sc-GRIP, please cite the ICDM 2025 paper (see [`CITATION.cff`](CITATION.cff)) and
state which version you used (the `icdm-2025` tag or a release of the current package).

## License

<!-- TODO: add a LICENSE file (e.g. MIT) and name it here. -->
No license has been chosen yet; until one is added, all rights are reserved by the authors.

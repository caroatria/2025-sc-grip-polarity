# Changelog

## 0.2.0 (unreleased) — redesigned implementation

Substantial rewrite of the ICDM 2025 code. It does not reproduce the paper's numbers; the
published code is preserved under the tag `icdm-2025`.

### Package and tooling
- Installable package `scgrip` (`pip install -e .`), command-line tool `scgrip polarity` /
  `scgrip existence`, lazy imports (helpers such as `scgrip.orthology` work without torch).
- Tests (`pytest`) and a synthetic-data quick start (`examples/quickstart.py`).

### Model and training (polarity)
- Edges are mapped to the correct gene nodes: the ICDM script used positions in the TF × target
  matrix as node indices, so edges connected the wrong genes. Self-loops are dropped.
- Message passing uses TF → target and reverse edges; the decoder scores only the directed
  TF → target edges.
- Mini-batch training on logits with `BCEWithLogitsLoss`; per-cell scores aggregated by median
  within a run and by mean ± std (and median) across runs.
- Per-gene z-scoring of expression before graph construction.

### Proxy labels
- `adaptive` (default): activation / repression from the top / bottom quantiles of the candidate
  edges' own correlation distribution, edges in between left unlabelled. The paper's fixed
  threshold (`binary`, Pearson > 0.5) labels almost every edge repression on sparse data.
- `ternary` (fixed cut-offs with an uncertain band) and `logic` (per-target combinatorial
  Boolean rules over all regulating TFs).
- Optional correlation over only the cells where the TF is detected; edges with too few cells
  are marked inestimable rather than set to 0.

### New: edge existence
- `scgrip.existence.score_candidate_edges`: label-free link prediction over all TF → gene pairs
  with the same encoder/decoder; proxy labels per TF (default) or global; per-cell or
  expression-profile node features; optional CLR normalisation.

### Preprocessing and non-model species
- Cell-type restriction from `.h5ad` annotations, pseudobulking.
- Ortholog translation of predicted edges, strict and family-level conservation checks,
  permutation and degree-preserving rewiring tests, network-topology summaries and plots.

### Validation
- `benchmarks/beeline`: polarity and existence on BEELINE, with correlation, GENIE3 and
  hub baselines, EPR@k and within-TF AUROC.
- `benchmarks/encode`: ENCODE ChIP-seq benchmark (K562 / HepG2) with ten baseline methods.

## icdm-2025 — code as published with the paper
- `sc-GRIP.py` and `sc_grip_notebook.ipynb`.

# Design notes

Short rationale for choices that are not obvious from the code.

## Proxy labels

scGRIP never trains on a reference network. Its labels come from the expression data:

- **binary** (paper Eq. 1): activation if the TF–target correlation exceeds `tau` (0.5),
  repression otherwise. On sparse, zero-inflated scRNA-seq data few TF–target correlations reach
  0.5, so almost every edge is labelled repression and training collapses.
- **adaptive** (default): thresholds at quantiles of the candidate edges' own correlation
  distribution (`low_q`, `high_q`, default 0.3 / 0.7); edges in between are left unlabelled.
- **ternary**: fixed cut-offs (`tau_low`, `tau_high`) with an uncertain band.

Because labels are derived from correlation, compare scGRIP against plain correlation on every
benchmark: if the model does not beat the signal it trains on, the graph adds nothing.

## Cell-type-conditioned labels

Correlations pooled over a mixed tissue can reflect cell-type composition rather than
regulation. `scgrip.celltype.restrict_to_cell_type` runs scGRIP on one cell type, and
`condition_on_tf_detected=True` computes each edge's correlation only over cells where the TF is
detected. Both helped on some datasets and hurt on others, so they are opt-in.

## Combinatorial-logic labels

A target switched on by "TF1 AND NOT TF2" can have near-zero correlation with either TF.
`label_mode="logic"` fits Boolean rules over all of a target's regulating TFs (binarised by
detection) and reads off each TF's monotonic role in the best cross-validated rule.
`scgrip.synthetic.make_combinatorial_dataset` builds such a case (an AND gate plus a decoy TF);
`tests/test_labels.py` checks that logic labels call it correctly. A bimodality-based binarisation is also
implemented but scored worse than plain detection on the mouse datasets it was tested on.

## Edge existence and hub effects

On BEELINE's experimental datasets, a global choice of proxy positives (top pairs by |Spearman|)
concentrates on a few hub genes, and the GCN amplifies this: scores become mostly a function of
the regulator. Choosing positives per TF (`label_scheme="per_tf"`) fixes the regulator side but the
hub effect then appears on the target side. `benchmarks/beeline/RESULTS.md` has the numbers; the
benchmark reports hub baselines and within-TF AUROC so this is visible for any method.

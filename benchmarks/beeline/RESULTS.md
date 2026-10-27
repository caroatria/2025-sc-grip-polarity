# BEELINE results (October 2026, scgrip 0.2.0)

All runs are label-free: scGRIP never sees the reference network during training.
These are development runs (single replicates or short training), recorded so the
current state of the method is transparent. They are not results of the ICDM 2025 paper.

## 1. Polarity on the curated BoolODE models

`polarity.py simulated`, one replicate per model, 2 runs x 30 epochs, `label_mode="adaptive"`.
AUROC: activating vs repressing edges (0.5 = random). VSC has only repressive edges (AUROC undefined).

| Model | scGRIP AUROC | Spearman AUROC | scGRIP accuracy | Spearman accuracy |
|---|---|---|---|---|
| GSD | 0.55 | 0.97 | 0.42 | 0.89 |
| GSD-q50 | 0.53 | 0.97 | 0.43 | 0.91 |
| GSD-q70 | 0.54 | 0.97 | 0.46 | 0.88 |
| HSC | 0.84 | 0.99 | 0.73 | 0.85 |
| HSC-q50 | 0.87 | 0.97 | 0.77 | 0.85 |
| HSC-q70 | 0.84 | 0.94 | 0.81 | 0.85 |
| VSC | – | – | 0.27 | 0.87 |
| VSC-q50 | – | – | 0.47 | 0.87 |
| VSC-q70 | – | – | 0.40 | 0.87 |
| mCAD | 0.81 | 1.00 | 0.69 | 1.00 |
| mCAD-q50 | 0.68 | 1.00 | 0.69 | 1.00 |
| mCAD-q70 | 0.64 | 1.00 | 0.62 | 1.00 |

scGRIP is trained on correlation-derived labels and does not yet reach the correlation it learns from.

## 2. Edge existence on the curated models

`existence.py simulated`, global proxy labels (top 5% |Spearman|), 1 replicate, 3 runs x 30 epochs.
EPR = precision in the top-k / edge density (1 = random). Medians over the 12 models:

| Method | EPR | AUPRC ratio | AUROC |
|---|---|---|---|
| scGRIP | 1.44 | 1.40 | 0.54 |
| abs Spearman | 1.54 | 1.56 | 0.63 |
| abs Pearson | 1.56 | 1.34 | 0.60 |
| GENIE3 | 1.58 | 1.39 | 0.62 |

The networks are tiny (5–19 genes), so these differences are within noise.

## 3. Edge existence on hESC (BEELINE scRNA-Seq)

758 cells, TFs + 500 most variable genes (910 genes, 410 TFs, 372,690 pairs), 3 runs x 30 epochs.
Scored against three reference networks. `within-TF` = mean AUROC of target ranking inside each TF.

| Variant | STRING AUPRC ratio | STRING EPR@100 | STRING within-TF | general ChIP AUPRC ratio | general ChIP within-TF | hESC ChIP AUPRC ratio | hESC ChIP EPR@100 |
|---|---|---|---|---|---|---|---|
| global labels, per-cell (original design) | 1.19 | 0.88 | 0.62 | 0.97 | 0.58 | 2.00 | 0.00 |
| global labels + CLR | 1.22 | 1.75 | 0.58 | 1.00 | 0.54 | 1.69 | 0.00 |
| global labels, profile features | 1.10 | 0.88 | 0.51 | 0.95 | 0.52 | 1.37 | 0.00 |
| per-TF labels, profile features | 1.15 | 0.00 | 0.57 | 1.07 | 0.56 | 0.90 | 0.00 |
| **per-TF labels, per-cell (current default)** | **1.36** | 0.00 | **0.65** | **1.23** | **0.63** | 1.02 | 0.00 |
| per-TF labels, per-cell + CLR | 1.25 | 0.00 | 0.61 | 1.16 | 0.58 | 1.21 | 0.00 |
| abs Spearman | 1.21 | 2.63 | 0.56 | 0.96 | 0.53 | 1.45 | 4.09 |
| GENIE3 (ExtraTrees, 50 trees) | 1.76 | 5.25 | 0.59 | 1.09 | 0.55 | 1.64 | 5.73 |
| TF_degree (hub baseline) | – | – | – | – | – | 2.62 | 0.00 |

Notes:
- In the hESC ChIP-seq reference only 34 of 410 TFs have any edge, so overall AUPRC/EPR mostly reward
  ranking those TFs first: the TF_degree baseline, which ignores targets, beats every method there.
  No method ranks targets within a TF better than random on that reference.
- With global labels, 70% of scGRIP's score variance is explained by the regulator alone; its top 500
  pairs all come from two TFs (NPM1, POU2AF1) without ChIP-seq edges.
- Per-TF labels give the best within-TF target ranking of any method on STRING and the general ChIP
  network. Simple target-hub baselines do not reproduce this (within-TF 0.47–0.52). But the hub effect
  moves to the target side (93% of score variance explained by the target gene), and the top 100 pairs
  contain no true edges. Removing the target effect after training raises EPR@100 to 2.6 (STRING) and
  3.3 (general ChIP) but loses the within-TF advantage.
- For top-of-list precision, GENIE3 and plain correlation are still ahead.

# Established GRN inference methods on the ENCODE ChIP-seq benchmark

Ten published or standard network inference methods, run on the same matrices
and scored against the same ground truth as `scgrip_encode_benchmark`. This
sets the bar sc-GRIP has to clear. The ten span three families: expression-only,
expression combined with motif evidence (SCENIC), and motif-only predictors that
never see expression.

## Two protocols, and why the distinction matters here

sc-GRIP does not take raw expression alone. Its candidate edge set comes from
FIMO motif scanning, and the model scores within that set. So an
expression-only comparison understates what sc-GRIP is given, and a
motif-aware comparison is the fair one.

- **Protocol A** — every method restricted to the 14,318 labelled candidate
  pairs. All results below are Protocol A.
- **Protocol B** — the view restricted to the 25 TFs that carry any cisTarget
  motif annotation (8,112 pairs). This is the subset any motif-aware method can
  address at all; outside it a motif method has no information by construction,
  so including those pairs charges it for a gap it cannot close.

Both appear in `grn_method_comparison.tsv` as the `all_edges` and
`motif_annotated_tfs` views.

## What was run

| method | what it is | settings | K562 | HepG2 |
|---|---|---|---|---|
| Pearson | \|r\| between TF and target | — | 0.3 s | 0.2 s |
| Spearman | \|rho\|, rank-based | — | 0.5 s | 0.6 s |
| mutual information | `sklearn.mutual_info_regression` | k = 3 neighbours | 215 s | 207 s |
| partial correlation | \|partial r\| given all other TFs | Ledoit-Wolf shrinkage precision | 6.5 s | 6.6 s |
| GRNBoost2 | stochastic gradient boosting, arboreto | published `SGBM_KWARGS`, early stop window 25 | 33 s | 32 s |
| GENIE3 | random forest, arboreto | published `RF_KWARGS`, 1000 trees | 1870 s | not run |
| SCENIC | GRNBoost2 modules pruned by cisTarget motif enrichment | pySCENIC defaults: rank 5000, AUC 0.05, NES 3.0; hg38 500bp-up/100bp-down clustered database | 124 s | 162 s |
| cisTarget rank | best motif rank per TF-gene pair, no expression | hg38 promoter rankings, motifs-v10nr_clust HGNC annotation | — | — |
| cisTarget CRM score | max motif CRM score per pair, no expression | same database, score variant | — | — |
| JASPAR PWM scan | independent log-odds scan of promoters, no expression | JASPAR2024 CORE vertebrates nr; Ensembl GRCh38; 2,001 bp window | — | — |
| random | uniform scores, 20 seeds | — | — | — |

The three motif-only methods produce one score vector per pair, used unchanged
for both cell lines. The two cisTarget variants share a single underlying scan,
so they are not independent of each other; the JASPAR scan is, and it replicates
the step sc-GRIP performs with its own motif scanner — different matrix
collection, different genome fetch, no cisTarget resources involved.

SCENIC ran pySCENIC's own `modules_from_adjacencies`, `prune2df` and
`df2regulons` on the saved GRNBoost2 adjacencies, with its pruning driven
in-process (threading) rather than through dask. Two library incompatibilities
had to be patched: one pandas-3 `groupby.apply` call in module construction, and
a non-thread-safe incremental column cache in `ctxcore`'s feather reader, worked
around by pre-warming each database with the full gene panel before threading.

GENIE3 on HepG2 was stopped before completion (about 31 min per line). It therefore has
no swapped-label control and appears in panel-level results for K562 only —
it is marked "not run" in the figure rather than dropped.

GRNBoost2 and GENIE3 come from `arboreto`, called through its own
`infer_partial_network` routine. Only arboreto's dask scheduling layer was
replaced by a joblib loop (no dask cluster needed); the regressor type,
hyperparameters and early-stopping logic are the package's published constants.

Inputs: 4,000 cells x 1,512 genes per line, 42 TFs, the 14,318 benchmark
candidate edges. Every method produces a dense TF x target score matrix,
restricted to those edges and written to `predictions/preds_<method>_<cell>.tsv`.

## Result: the two cell lines reward opposite kinds of evidence

Twenty random-score replicates put the chance band at AUROC 0.490-0.508
(all edges). Every one of the ten methods lands within 0.06 of chance on the
full pair set. Which ones clear the band, though, splits perfectly by family:

| family | K562 labels | HepG2 labels |
|---|---|---|
| expression-only (6 methods) | only Pearson clears (0.518); GENIE3 0.452, mutual info 0.472, GRNBoost2 0.478 fall *below* | 4 of 5 clear, best Pearson 0.545 |
| expression + motif (SCENIC) | 0.479, below the band | 0.523, clears |
| motif-only (3 methods) | all three clear: 0.536, 0.539, 0.556 | all three below: 0.467, 0.470, 0.479 |

Every expression-based method clears the band on HepG2 and fails on K562; every
motif-only method does the exact opposite. The split holds across all six
expression methods and all three motif methods, so it is a property of the two
label sets, not of any one algorithm.

**The best score in the table uses no expression data.** The JASPAR PWM scan
reaches AUROC 0.556 on K562, rising to 0.581 under Protocol B, with the highest
early-precision ratio anywhere in the comparison (1.23). The two cisTarget
variants land just behind it (0.536, 0.539), and their agreement with the
independent scan means the K562 motif signal is a property of promoter sequence
rather than of cisTarget's gene-centric aggregation.

Other framings of the same data:

- Per-TF mean AUROC, which removes the effect of how broadly each factor binds,
  spans 0.480-0.550.
- Early precision / prevalence runs 0.82-1.23 across all ten methods: the
  highest-scoring edges are at best 23% more likely to be bound than a random
  edge, and often less likely.
- The full field spans AUROC 0.359-0.601 across every method, cell line and
  view — nowhere near the range usually quoted for GRN inference.

### Full results

Protocol A — all 14,318 labelled pairs. Bold = above the upper edge of the
random band (0.508). EPR = early precision / prevalence.

| method | K562 AUROC | K562 AUPRC | K562 EPR | HepG2 AUROC | HepG2 AUPRC | HepG2 EPR |
|---|---|---|---|---|---|---|
| Pearson correlation | **0.518** | 0.299 | 1.07 | **0.545** | 0.429 | 1.12 |
| Spearman correlation | 0.493 | 0.283 | 1.00 | **0.522** | 0.410 | 1.06 |
| Partial correlation | 0.492 | 0.283 | 0.97 | 0.492 | 0.376 | 0.98 |
| Mutual information | 0.472 | 0.272 | 0.88 | **0.518** | 0.401 | 1.07 |
| GENIE3 | 0.452 | 0.267 | 0.82 | n.d. | n.d. | n.d. |
| GRNBoost2 | 0.478 | 0.275 | 0.96 | **0.527** | 0.411 | 1.06 |
| SCENIC (GRNBoost2 + cisTarget) | 0.479 | 0.280 | 0.96 | **0.523** | 0.402 | 1.06 |
| cisTarget rank, best motif | **0.536** | 0.320 | 1.12 | 0.467 | 0.360 | 0.92 |
| cisTarget CRM score | **0.539** | 0.322 | 1.17 | 0.469 | 0.358 | 0.92 |
| JASPAR PWM scan of promoters | **0.556** | 0.350 | 1.23 | 0.479 | 0.383 | 0.97 |

Protocol B — the 25 motif-annotated TFs (8,112 pairs); random band upper edge
0.509.

| method | K562 AUROC | K562 AUPRC | K562 EPR | HepG2 AUROC | HepG2 AUPRC | HepG2 EPR |
|---|---|---|---|---|---|---|
| Pearson correlation | 0.506 | 0.312 | 1.04 | **0.551** | 0.414 | 1.16 |
| Spearman correlation | 0.482 | 0.295 | 0.95 | **0.533** | 0.399 | 1.09 |
| Partial correlation | 0.494 | 0.306 | 0.96 | 0.494 | 0.359 | 0.98 |
| Mutual information | 0.467 | 0.284 | 0.91 | 0.490 | 0.351 | 1.04 |
| GENIE3 | 0.437 | 0.264 | 0.71 | n.d. | n.d. | n.d. |
| GRNBoost2 | 0.466 | 0.288 | 0.92 | 0.498 | 0.366 | 1.01 |
| SCENIC (GRNBoost2 + cisTarget) | 0.468 | 0.295 | 0.92 | 0.495 | 0.365 | 0.99 |
| cisTarget rank, best motif | **0.523** | 0.332 | 1.12 | 0.485 | 0.350 | 0.97 |
| cisTarget CRM score | **0.536** | 0.338 | 1.18 | 0.489 | 0.346 | 0.97 |
| JASPAR PWM scan of promoters | **0.581** | 0.388 | 1.29 | 0.491 | 0.383 | 0.98 |

## cisTarget pruning does not change GRNBoost2's ranking

SCENIC lands within 0.011 AUROC of unpruned GRNBoost2 at every one of the eight
cell-line x view settings, despite discarding almost all candidate pairs:

| | K562 | HepG2 |
|---|---|---|
| GRNBoost2 adjacency links | 58,502 | 56,959 |
| modules built | 242 | 212 |
| motif-enrichment rows | 58 | 97 |
| regulons formed | 10 | 12 |
| benchmark pairs surviving pruning | 98 / 14,318 (0.7%) | 419 / 14,318 (2.9%) |
| TFs with any surviving edge | 9 | 12 |

With that few survivors the ranking is dominated by a survived-or-not
tie-break, so the AUROC is pinned to its unpruned input. On a focused 42-TF
panel, pruning at pySCENIC's default thresholds neither helps nor hurts. This is
the direct test of the assumption that adding a motif-enrichment filter to an
expression method is a free improvement — here it is not.

## Two things to read carefully

**The swap control is redundant on panel b.** On the 4,726 cell-type-
discriminative edges the two label vectors are exact complements (verified:
`label_K562 == 1 - label_HepG2` for all 4,726). So for a fixed score vector,
AUROC against K562 labels is exactly 1 - AUROC against HepG2 labels, and the
matched-minus-swapped gap on that view is identically 2 x (AUROC - 0.5). The
large apparent "specificity gaps" on the discriminative subset are an algebraic
restatement of the matched AUROC, not independent evidence. Use the
`all_edges` view for the swap control; that is where the gap carries
information, and there it is small (|gap| <= 0.095 across all ten methods,
largest for GENIE3 on K562; <= 0.077 for every method run on both lines).

**The swap control cannot test a cell-invariant predictor at all.** The three
motif-only methods emit one score vector used for both lines, so their matched
and swapped AUROCs differ only through the labels, which forces
gap(from K562) = -gap(from HepG2) exactly. Verified: the two gaps sum to 0.0 for
all three methods on all four views. The plan had expected this gap to be zero
as a scorer check; the correct invariant is antisymmetry, not zero, and it holds
exactly. The practical consequence is that no swap-based specificity test is
meaningful for an expression-free score — its apparent specificity in one line
is the exact negative of its specificity in the other.

**The expression-level confound is present but mild.** Highly expressed genes
are bound by somewhat more factors (Spearman between mean expression and
fraction of TFs bound: +0.168 in K562, +0.097 in HepG2; top vs bottom
expression quartile, 0.41 vs 0.29 of TFs bound in K562). Any method that
favours high-expression genes inherits a small advantage, but it is too weak to
explain the results either way.

## How this compares to published benchmarks

The result is consistent with the published literature once you match the
evaluation setting. The near-chance outcome is what these methods do against
ChIP-derived ground truth in mammalian cells; their strong reputation comes
from different data.

| study | setting | ground truth | reported performance |
|---|---|---|---|
| BEELINE, Pratapa et al. 2020, *Nat Methods* (10.1038/s41592-019-0690-6) | human/mouse scRNA-seq, 12 methods | cell-type-specific ChIP-seq networks, density 0.08-0.58 | early precision ratio "close to 1", none above 1.5; GENIE3, PIDC, GRNBoost2 the best of them |
| McCalla et al. 2023, *G3* (10.1093/g3journal/jkad004) | 11 methods, 7 scRNA-seq datasets, human/mouse/yeast | ChIP, perturbation, and their intersection | AUPR and F-score "often close to random" |
| DREAM5, Marbach et al. 2012, *Nat Methods* (10.1038/nmeth.2016) | bulk microarray compendia, >30 methods | curated/validated regulatory networks | random-forest method (GENIE3) best overall score |
| **this benchmark** | 2 human cell lines, scRNA-seq, 10 methods | ENCODE promoter-proximal ChIP-seq peaks, shared edge space | AUROC 0.45-0.56, EPR 0.82-1.23 |

### Why this benchmark exists alongside BEELINE, and not instead of it

This benchmark was built before BEELINE's data were available to the project,
so the baseline methods here are positioned against BEELINE's *published
numbers* (above), not re-run on its data. scGRIP itself has since been run on
BEELINE: see `benchmarks/beeline/`.

Read as a complement, the design does contribute three things BEELINE does not:

1. **A shared candidate edge space across two cell lines.** 4,726 of the 14,318
   pairs are labelled oppositely in K562 and HepG2. BEELINE's datasets each
   carry their own gene panel and their own ground truth, so the question "is
   this edge present here and absent there, over identical candidates" cannot be
   posed in it. This is the sharpest test in the set and the only one a
   cell-type-invariant predictor cannot fake.
2. **Per-TF balanced candidates.** BEELINE's cell-type-specific ChIP networks
   run 0.08-0.58 density, unbalanced across factors, so AUROC there partly
   tracks how broadly each TF binds. Balancing per TF removes that.
3. **Matched ENCODE ChIP for the same two lines** as the expression data, rather
   than ground truth aggregated across cell types.

What BEELINE has that this does not: an established evaluation protocol, twelve
methods already run, and numbers a reader can compare to directly. And on the
one requirement it cannot meet — all three of its real-data ground-truth tiers
(cell-type-specific ChIP, non-specific ChIP, STRING) are unsigned, so
activation/repression polarity is unreachable in it — this set is equally short:
polarity is not in it yet either.

To close the gap, either obtain the Zenodo bundle (record 3701939) outside this
environment, or rebuild BEELINE's protocol from its GEO-hosted expression data
(hESC GSE75748, hHep GSE81252, mESC GSE98664, mHSC GSE81682, mDC GSE48968) with
ground truth reconstructed from ENCODE and STRING — the latter is not the
published benchmark and must be labelled as a reconstruction wherever reported.

Our EPR range of 0.82-1.23 sits inside BEELINE's "close to 1, nothing above
1.5" band for cell-type-specific ChIP networks, whose densities (0.08-0.58)
bracket our label prevalence (0.29 K562, 0.38 HepG2). The agreement is close
enough that this benchmark should be read as reproducing the published result,
not contradicting it.

**Why GENIE3 and GRNBoost2 are reputed to be strong.** Three reasons, none of
which transfers to this setting:

1. *DREAM5 was bulk perturbation compendia, not one unperturbed cell line.*
   Each DREAM5 dataset queried its network across hundreds of conditions —
   time courses, gene, drug and environmental perturbations — including direct
   TF perturbations (67 of 806 arrays in E. coli). Here there is a single
   steady-state population per line, so the variance a tree ensemble can
   exploit is largely noise, cell cycle and library size.
2. *In BEELINE they are the best of a near-random set on exactly this kind of
   ground truth.* They score well on synthetic networks and curated Boolean
   models (AUROC up to ~0.75 on some models), and that is what the headline
   reputation rests on. On experimental data with cell-type-specific ChIP
   ground truth, BEELINE reports them as top-ranked while still near random.
3. *SCENIC's success is not GRNBoost2 alone.* GRNBoost2 is step 1 of SCENIC;
   the cisTarget motif-enrichment step then prunes the co-expression modules
   using external sequence evidence. An earlier version of this report excluded
   that step on the grounds that it injects information sc-GRIP does not use.
   That reasoning was wrong — sc-GRIP's candidate edges come from FIMO motif
   scanning, so motif evidence is part of its input budget too, and excluding
   cisTarget made SCENIC an unfairly weakened peer. Full SCENIC has now been
   run; on this benchmark the pruning step changes AUROC by at most 0.011
   (see above), so the earlier exclusion did not affect any conclusion, but the
   comparison is now the fair one. In McCalla et al. SCENIC is among the
   stronger expression-based performers, which makes its flat result here a
   statement about this benchmark rather than about the method.

Two further reasons the ceiling is low in this specific design: ChIP occupancy
is binding, not regulation, and promoter-proximal binding is pervasive (which
is why the benchmark needed balancing); and TF mRNA abundance is a weak proxy
for TF activity, since most regulation of TF function is post-translational.

## Coverage limits on the motif methods

- cisTarget annotates 25 of 42 TFs, covering 8,112 of 14,318 pairs; 56.6% of
  pairs receive a non-zero CRM score.
- JASPAR 2024 CORE vertebrates (non-redundant) supplies a matrix for only 14 of
  42 TFs, leaving 4,049 of 14,318 pairs scorable. Unscorable pairs were floored
  below all scored pairs rather than dropped, so the PWM AUROC is conservative —
  its 0.556 on K562 is a floor on what that scan achieves, not a ceiling.
- The PWM scan used the benchmark's own 2,001 bp promoter window over 1,505
  promoters fetched from Ensembl GRCh38. Dimer matrices were split across their
  constituent factors.

## What this means for sc-GRIP

**The bar is no longer AUROC 0.5 — it is a sequence-only scan.** On K562,
promoter PWM content alone reaches 0.556 (0.581 under Protocol B), beating every
expression method tested. sc-GRIP consumes a motif-derived adjacency as its
candidate edge set, so this is not an external baseline: it is sc-GRIP's own
input, unmodelled. Any claim that sc-GRIP's expression modelling contributes
something must show a gain over this floor, on the same pairs, in the same line.

Four consequences:

1. Run and report both cell lines. A K562-only result would say sequence carries
   the signal; a HepG2-only result would say expression does. Either alone
   supports the opposite conclusion from the other.
2. Report the motif-only floor alongside any sc-GRIP number. Without it, an
   sc-GRIP AUROC near 0.55 on K562 is indistinguishable from its motif input
   passed through untouched.
3. Do not assume a motif-filtering stage helps. SCENIC's cisTarget step
   discarded over 97% of candidate pairs and moved AUROC by at most 0.011. If
   sc-GRIP adds such a stage, measure its effect as an ablation.
4. A modest score is still not automatically a weak result — the whole field of
   ten methods spans 0.359-0.601 here. But it should be read against the
   motif-only floor, not against 0.5.

Beyond that, beating this bar plausibly requires information none of these ten
methods use: perturbation response, chromatin state, or cross-cell-type
contrast. The memory from the sc-GRIP work — that the decisive ground truth is
one not derivable from the correlation statistic a baseline already computes —
applies with equal force to the motif floor.

## Not assessed

No paired test or bootstrap confidence interval on between-method AUROC
differences, so the small gaps between neighbouring methods should not be read
as rankings. No calibration analysis. Whether the K562/HepG2 asymmetry is driven
by mean expression level rather than TF-specific signal is untested; a
gene-expression-level-only predictor is the obvious next baseline and would
separate the two.

## Files

| file | contents |
|---|---|
| `results/grn_method_comparison.tsv` | all ten methods x cell x view x comparison, with AUROC, AUPRC, early precision and runtime |
| `results/random_chance_band.tsv` | 20-seed random band (mean, 2.5th, 97.5th percentile) per view |
| `results/grn_method_comparison.png` | the four-panel figure (written by `make_figure.py`): AUROC vs chance band on all pairs and on motif-annotated TFs, SCENIC vs GRNBoost2, and the specificity gap |
| `run_grn_methods.py` | the six expression-only methods; `--cell {K562,HepG2} --method <name>` |
| `run_scenic.py` | SCENIC: module construction, cisTarget pruning, regulon-to-edge scoring |
| `run_motif_only.py` | the two cisTarget motif-only variants |
| `run_pwm_scan.py` | the independent JASPAR/Ensembl promoter PWM scan |
| `make_figure.py` | the figure |
| `aggregate_scores.py` | scoring and assembly; imports the benchmark's own metric functions |
| `predictions/` | per-method edge scores and run metadata |
| `build/cistarget/` | cisTarget ranking databases and HGNC motif annotation (sha1-verified) |
| `build/pwm/` | JASPAR matrices, gene loci and promoter sequences |

To score sc-GRIP the same way, write its edge scores as `tf, target, score`
and run the benchmark's own scorer:

```
python score_edge_existence.py --predictions scgrip_K562.tsv --cell K562
python score_edge_existence.py --predictions scgrip_HepG2.tsv --cell HepG2
```

Then add it to the comparison table and figure:

```
python aggregate_scores.py            # rescores every method, writes the TSV
python make_figure.py                 # re-renders results/grn_method_comparison.png
```

Protocol B needs no separate run — `aggregate_scores.py` derives the
`motif_annotated_tfs` view from the same prediction files.

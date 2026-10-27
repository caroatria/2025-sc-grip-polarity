"""
Combinatorial-logic proxy labels
===================================
Every other proxy-label function in `scgrip.data` derives an edge's label
from *pairwise* correlation between one TF and its target. That's
structurally blind to combinatorial regulation: a target genuinely turned
on only by "TF1 AND NOT TF2" can show near-zero correlation with either
TF alone, even though both have a real, strong effect — correlation only
sees a monotonic, single-variable relationship, and this one isn't one.

This module derives labels differently: group a target's regulating TFs
together (however many there are), exhaustively or semi-exhaustively
search Boolean functions of their (detected/not-detected) expression for
the one that best predicts the target's own (detected/not-detected)
expression, cross-validated, and read off each TF's *monotonic role*
within that winning rule (does the rule's output ever increase when this
TF turns on, ever decrease, or depend on it in a non-monotonic way, e.g.
XOR, or not depend on it at all).

The implementation is self-contained (numpy/pandas only, no scipy or
scikit-learn) and fits Boolean rules only (no continuous regression).

Binarization defaults to `binarize="detected"` (expression > 0). A more
sophisticated `binarize="bimodal"` is also implemented — a numpy-only
approximation of a KDE-valley threshold (histogram + moving-average
smoothing, gated by Sarle's bimodality coefficient, falling back to a
75th-percentile threshold otherwise) — but it is not the default because it
scored *worse* than plain detection on the three mouse datasets it was
tested on (docs/design_notes.md, "Combinatorial-logic labels"). Best explanation: any
threshold above 0 can only discard cells this simple detection keeps, and
on data this sparse the "weakly detected" cells being discarded carry
real, already-scarce positive signal rather than being noise worth
filtering out — sophistication isn't free here.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .data import EdgeIndex


# ---------------------------------------------------------------------------
# Binarization (no scipy)
# ---------------------------------------------------------------------------

def _bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient (same formula `regulatory_logic.py`
    uses): 0-1, > ~0.555 suggests genuinely bimodal. Needs only pandas'
    `.skew()`/`.kurtosis()`, which are plain numpy under the hood, not
    scipy-backed."""
    n = len(x)
    if n < 4:
        return 0.0
    s = pd.Series(x)
    skew, kurt = s.skew(), s.kurtosis()
    denom = kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    return float(np.clip((skew ** 2 + 1) / denom, 0, 1))


def _bimodal_threshold(
    x: np.ndarray, min_expressing_frac: float, n_bins: int, smooth_window: int, bimodality_cutoff: float = 0.555,
) -> float:
    """
    Numpy-only approximation of `regulatory_logic.py`'s KDE-valley
    threshold: histogram the nonzero values, smooth with a small moving
    average (`np.convolve` instead of `gaussian_kde`), and threshold at
    the deepest valley closest to the median nonzero value — but *only*
    when Sarle's bimodality coefficient says the nonzero distribution is
    actually bimodal enough to trust a valley in the first place. Without
    that gate, a purely noisy, unimodal nonzero distribution (common on
    data this sparse — most genes here just don't have a clean on/off
    split among the cells that detect them at all) still has *some*
    locally-lowest histogram bin from binning noise alone, and trusting it
    unconditionally picks essentially arbitrary thresholds — confirmed
    empirically: without this gate, "bimodal" binarization performed
    *worse* than plain detection on every real dataset tried (docs/design_notes.md).
    Falls back to the 75th percentile of nonzero values (matching
    `regulatory_logic.py`'s own fallback) when not bimodal enough, or too
    little nonzero signal to check; to 0.0 (plain detection) when there's
    too little nonzero signal even for that fallback to be well-founded.
    """
    frac_expr = (x > 0).mean()
    if frac_expr < min_expressing_frac:
        return np.inf  # gene is near-silent — binarize everything to 0

    nonzero = x[x > 0]
    if len(nonzero) < 20:
        return 0.0

    if _bimodality_coefficient(nonzero) < bimodality_cutoff:
        return float(np.percentile(nonzero, 75))

    counts, edges = np.histogram(nonzero, bins=n_bins)
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(counts.astype(float), kernel, mode="same")

    valleys = [i for i in range(1, len(smoothed) - 1) if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]]
    if not valleys:
        return float(np.percentile(nonzero, 75))

    bin_centers = (edges[:-1] + edges[1:]) / 2
    med = np.median(nonzero)
    best = min(valleys, key=lambda i: abs(bin_centers[i] - med))
    return float(bin_centers[best])


def binarize_matrix(
    expr_df: pd.DataFrame,
    method: str = "detected",
    min_expressing_frac: float = 0.05,
    n_bins: int = 30,
    smooth_window: int = 3,
) -> pd.DataFrame:
    """
    Binarize every gene (column) in `expr_df` independently.

    Parameters
    ----------
    method : "detected" (default) uses the simple `expression > 0`;
        "bimodal" uses `_bimodal_threshold` per gene instead — a more
        sophisticated, data-driven threshold that, on this package's real
        data, measurably underperforms plain detection (see module
        docstring). Kept available rather than removed, since a different
        data regime (less extreme sparsity) might favor it.
    """
    if method not in ("bimodal", "detected"):
        raise ValueError(f"method must be 'bimodal' or 'detected', got {method!r}")

    values = expr_df.to_numpy()
    if method == "detected":
        binarized = (values > 0).astype(np.int8)
    else:
        binarized = np.zeros_like(values, dtype=np.int8)
        for j in range(values.shape[1]):
            threshold = _bimodal_threshold(values[:, j], min_expressing_frac, n_bins, smooth_window)
            binarized[:, j] = (values[:, j] > threshold).astype(np.int8)

    return pd.DataFrame(binarized, index=expr_df.index, columns=expr_df.columns)


# ---------------------------------------------------------------------------
# Hand-rolled cross-validation / scoring (no scikit-learn)
# ---------------------------------------------------------------------------

def _kfold_splits(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """k roughly-equal folds over `n` items, shuffled once."""
    order = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(order, k)
    splits = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, test_idx))
    return splits


def _f1(pred: np.ndarray, truth: np.ndarray) -> float:
    tp = int(np.sum((pred == 1) & (truth == 1)))
    fp = int(np.sum((pred == 1) & (truth == 0)))
    fn = int(np.sum((pred == 0) & (truth == 1)))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Candidate Boolean rules, with each candidate's per-TF monotonic role
# tracked at construction time (no rule-string parsing needed later).
# ---------------------------------------------------------------------------

Role = Optional[str]   # "activation" | "repression" | None (no clean monotonic role)


@dataclass
class _RuleSearchResult:
    f1_cv: float
    roles: dict[str, Role] = field(default_factory=dict)


def _exhaustive_search(tf_bin: np.ndarray, tf_names: list[str], target_bin: np.ndarray, seed: int) -> _RuleSearchResult:
    """Full truth-table search over all 2^(2^n_tfs) Boolean functions — only
    used for n_tfs <= 4 (matches `regulatory_logic.py`'s own cutoff), where
    that's at most 65536 candidates over at most 16 truth-table rows."""
    n_tfs = tf_bin.shape[1]
    n_cells = tf_bin.shape[0]
    rows = list(itertools.product([0, 1], repeat=n_tfs))
    n_rows = len(rows)
    rows_arr = np.array(rows)  # [n_rows, n_tfs]

    row_lookup = {row: i for i, row in enumerate(rows)}
    cell_row_idx = np.array([row_lookup[tuple(r)] for r in tf_bin])

    outputs = np.array(list(itertools.product([0, 1], repeat=n_rows)), dtype=np.int8)  # [n_cand, n_rows]

    # Per-TF monotonic role of every candidate, vectorized over candidates.
    role_codes = np.zeros((outputs.shape[0], n_tfs), dtype=np.int8)  # -1 repress, 0 none, 1 activate
    for tf_idx in range(n_tfs):
        key_at_0 = {}
        for i, row in enumerate(rows):
            if row[tf_idx] == 0:
                key_at_0[row[:tf_idx] + row[tf_idx + 1:]] = i
        pairs = []
        for i, row in enumerate(rows):
            if row[tf_idx] == 1:
                other = row[:tf_idx] + row[tf_idx + 1:]
                if other in key_at_0:
                    pairs.append((key_at_0[other], i))
        if not pairs:
            continue
        idx0 = [p[0] for p in pairs]
        idx1 = [p[1] for p in pairs]
        diffs = outputs[:, idx1].astype(np.int8) - outputs[:, idx0].astype(np.int8)  # [n_cand, n_pairs]
        all_nonneg = (diffs >= 0).all(axis=1)
        all_nonpos = (diffs <= 0).all(axis=1)
        any_pos = (diffs > 0).any(axis=1)
        any_neg = (diffs < 0).any(axis=1)
        role_codes[all_nonneg & any_pos, tf_idx] = 1
        role_codes[all_nonpos & any_neg, tf_idx] = -1

    # Score every candidate via per-row positive/negative cell counts
    # (exact, avoids materializing an (n_candidates, n_cells) array).
    pos_count = np.bincount(cell_row_idx[target_bin == 1], minlength=n_rows).astype(float)
    neg_count = np.bincount(cell_row_idx[target_bin == 0], minlength=n_rows).astype(float)
    tp = outputs.astype(float) @ pos_count
    fp = outputs.astype(float) @ neg_count
    fn = pos_count.sum() - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1_full = np.divide(2 * precision * recall, precision + recall,
                         out=np.zeros_like(tp), where=(precision + recall) > 0)
    best_idx = int(np.argmax(f1_full))

    # Cross-validated F1 of whichever candidate wins on each training fold.
    fold_f1s = []
    for train_idx, test_idx in _kfold_splits(n_cells, k=min(5, max(2, n_cells // 10)), seed=seed):
        train_rows, train_target = cell_row_idx[train_idx], target_bin[train_idx]
        pos_c = np.bincount(train_rows[train_target == 1], minlength=n_rows).astype(float)
        neg_c = np.bincount(train_rows[train_target == 0], minlength=n_rows).astype(float)
        tp_t = outputs.astype(float) @ pos_c
        fp_t = outputs.astype(float) @ neg_c
        fn_t = pos_c.sum() - tp_t
        prec_t = np.divide(tp_t, tp_t + fp_t, out=np.zeros_like(tp_t), where=(tp_t + fp_t) > 0)
        rec_t = np.divide(tp_t, tp_t + fn_t, out=np.zeros_like(tp_t), where=(tp_t + fn_t) > 0)
        f1_t = np.divide(2 * prec_t * rec_t, prec_t + rec_t, out=np.zeros_like(tp_t), where=(prec_t + rec_t) > 0)
        train_best = int(np.argmax(f1_t))
        pred_test = outputs[train_best][cell_row_idx[test_idx]]
        fold_f1s.append(_f1(pred_test, target_bin[test_idx]))

    role_map = {"activation": 1, "repression": -1}
    roles = {}
    for i, tf in enumerate(tf_names):
        code = role_codes[best_idx, i]
        roles[tf] = "activation" if code == 1 else ("repression" if code == -1 else None)

    return _RuleSearchResult(f1_cv=float(np.mean(fold_f1s)), roles=roles)


def _structured_search(tf_bin: np.ndarray, tf_names: list[str], target_bin: np.ndarray, seed: int) -> _RuleSearchResult:
    """For n_tfs > 4: a small, structured candidate set (single TF, pairwise
    AND/OR/AND-NOT, ALL/ANY/MAJORITY) evaluated directly on the observed
    cells — same idea as `regulatory_logic.py`'s large-n branch, with each
    candidate's monotonic role known by construction instead of parsed."""
    n_cells, n_tfs = tf_bin.shape
    preds, roles_list = [], []

    def add(pred: np.ndarray, roles: dict[str, Role]):
        preds.append(pred.astype(np.int8))
        roles_list.append(roles)

    none_roles = {tf: None for tf in tf_names}
    for i, tf in enumerate(tf_names):
        r = dict(none_roles)
        add(tf_bin[:, i], {**none_roles, tf: "activation"})
        add(1 - tf_bin[:, i], {**none_roles, tf: "repression"})

    for i in range(n_tfs):
        for j in range(n_tfs):
            if i == j:
                continue
            add(tf_bin[:, i] & tf_bin[:, j], {**none_roles, tf_names[i]: "activation", tf_names[j]: "activation"})
            add(tf_bin[:, i] | tf_bin[:, j], {**none_roles, tf_names[i]: "activation", tf_names[j]: "activation"})
            add(tf_bin[:, i] & (1 - tf_bin[:, j]), {**none_roles, tf_names[i]: "activation", tf_names[j]: "repression"})

    add(tf_bin.all(axis=1).astype(np.int8), {tf: "activation" for tf in tf_names})
    add(tf_bin.any(axis=1).astype(np.int8), {tf: "activation" for tf in tf_names})
    add((tf_bin.sum(axis=1) > n_tfs / 2).astype(np.int8), {tf: "activation" for tf in tf_names})

    preds = np.stack(preds)  # [n_cand, n_cells]
    f1_full = np.array([_f1(p, target_bin) for p in preds])
    best_idx = int(np.argmax(f1_full))

    fold_f1s = []
    for train_idx, test_idx in _kfold_splits(n_cells, k=min(5, max(2, n_cells // 10)), seed=seed):
        f1_train = np.array([_f1(p[train_idx], target_bin[train_idx]) for p in preds])
        train_best = int(np.argmax(f1_train))
        fold_f1s.append(_f1(preds[train_best][test_idx], target_bin[test_idx]))

    return _RuleSearchResult(f1_cv=float(np.mean(fold_f1s)), roles=roles_list[best_idx])


def fit_boolean_roles(
    tf_bin: np.ndarray, tf_names: list[str], target_bin: np.ndarray, max_full_enum_n: int = 4, seed: int = 0,
) -> _RuleSearchResult:
    """Dispatch to exhaustive or structured search depending on TF count."""
    n_tfs = tf_bin.shape[1]
    if n_tfs <= max_full_enum_n:
        return _exhaustive_search(tf_bin, tf_names, target_bin, seed)
    return _structured_search(tf_bin, tf_names, target_bin, seed)


# ---------------------------------------------------------------------------
# Public proxy-label function, matching the interface of the correlation-
# based ones in scgrip.data (labels, known_mask, "corr"-shaped diagnostic).
# ---------------------------------------------------------------------------

def logic_proxy_labels(
    expr_df: pd.DataFrame,
    edge_index: EdgeIndex,
    min_cells: int = 30,
    min_f1_cv: float = 0.55,
    max_full_enum_n: int = 4,
    binarize: str = "detected",
    min_expressing_frac: float = 0.05,
    seed: int = 0,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Combinatorial-logic proxy labels (see module docstring).

    Parameters
    ----------
    min_cells : targets with fewer cells than this are left unsupervised
        (there's no reliable way to fit or cross-validate a Boolean rule
        on very few observations).
    min_f1_cv : the winning rule's cross-validated F1 must clear this bar
        for any of that target's edges to be treated as supervised — a
        rule that doesn't actually predict the target well shouldn't be
        trusted for its implied polarity either.
    binarize : "detected" (default) or "bimodal" — see `binarize_matrix`
        and this module's docstring for why "detected" is the default.

    Returns
    -------
    labels     : float tensor of {0, 1} for edges with `known_mask==True`.
    known_mask : bool array, one per edge — True where the target's TF
        count allowed a fit, had enough cells, the winning rule cleared
        `min_f1_cv`, AND this specific TF had a clean monotonic
        (activator/repressor) role in that rule.
    f1_cv      : each edge's target's winning-rule cross-validated F1,
        broadcast across that target's edges (named to line up with the
        other proxy-label functions' "corr" field, though it isn't one).
    """
    genes = edge_index.genes
    src, dst = edge_index.directed
    n_edges = edge_index.n_edges

    target_to_edges: dict[str, list[tuple[str, int]]] = {}
    for e, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
        target_to_edges.setdefault(genes[d], []).append((genes[s], e))

    known_mask = np.zeros(n_edges, dtype=bool)
    full_labels = np.zeros(n_edges, dtype=np.float32)
    f1_cv_per_edge = np.zeros(n_edges, dtype=np.float64)

    bin_df = binarize_matrix(expr_df, method=binarize, min_expressing_frac=min_expressing_frac)

    for target, tf_edge_pairs in target_to_edges.items():
        tfs = [tf for tf, _ in tf_edge_pairs]
        edges_here = [e for _, e in tf_edge_pairs]

        if len(expr_df) < min_cells:
            continue

        tf_bin = bin_df[tfs].to_numpy()
        target_bin = bin_df[target].to_numpy()
        if target_bin.sum() == 0 or target_bin.sum() == len(target_bin):
            continue  # target never varies — no rule can be evaluated meaningfully

        result = fit_boolean_roles(tf_bin, tfs, target_bin, max_full_enum_n=max_full_enum_n, seed=seed)
        f1_cv_per_edge[edges_here] = result.f1_cv
        if result.f1_cv < min_f1_cv:
            continue

        for tf, e in tf_edge_pairs:
            role = result.roles.get(tf)
            if role is None:
                continue
            known_mask[e] = True
            full_labels[e] = 1.0 if role == "activation" else 0.0

    labels = torch.tensor(full_labels[known_mask], dtype=torch.float32)
    return labels, known_mask, f1_cv_per_edge

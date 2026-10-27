"""
Evaluation against an experimentally-curated ground truth (e.g. TRRUST)
==========================================================================
sc-GRIP trains without any prior knowledge (Section III-A of the paper) —
its proxy labels come only from expression correlation. TRRUST is used
purely to *evaluate* the resulting predictions on model species, never to
train on (see paper Section IV-A). These helpers load a TRRUST-format
table and score a `ScGRIP.fit()` results DataFrame against it.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

# `scgrip/__init__.py` imports lazily, so `import scgrip.evaluate` does not pull in
# torch for analyses that never touch the model.
from .orthology import any_name_conserved, normalized_network_edges, pick_display_ortholog


def load_trrust(path: str) -> dict[tuple[str, str], str]:
    """
    Load a TRRUST `trrust_rawdata.<species>.tsv` file into a
    `(tf, target) -> "Activation" | "Repression"` dict.

    Pairs with conflicting annotations across publications (both
    "Activation" and "Repression" reported for the same pair) are dropped,
    since there's no principled way to prefer one over the other; "Unknown"
    entries are dropped too, since they carry no polarity information.
    """
    raw = pd.read_csv(path, sep="\t", header=None, names=["tf", "target", "type", "pmid"])
    raw = raw[raw["type"].isin(["Activation", "Repression"])]

    by_pair: dict[tuple[str, str], set[str]] = {}
    for tf, target, itype in zip(raw["tf"], raw["target"], raw["type"]):
        by_pair.setdefault((tf, target), set()).add(itype)

    return {pair: types.pop() for pair, types in by_pair.items() if len(types) == 1}


def restrict_ground_truth(ground_truth: dict, candidate_edges) -> dict:
    """Restrict a ground-truth dict to a specific set of `(tf, target)`
    pairs — typically a dataset's own candidate edges — before splitting
    or scoring against it."""
    candidate_set = set(candidate_edges)
    return {pair: label for pair, label in ground_truth.items() if pair in candidate_set}


def split_ground_truth(
    ground_truth: dict[tuple[str, str], str],
    test_frac: float = 0.4,
    seed: int = 0,
) -> tuple[dict, dict]:
    """
    Split a ground-truth dict into a (dev, test) pair, stratified by label
    so both splits keep roughly the same activation/repression ratio.

    Why this exists: comparing several design choices (label mode,
    cell-type restriction, correlation method, ...) by repeatedly checking
    each one against the same small ground-truth set risks quietly
    overfitting those choices to it — the evaluation set here is only
    ~100-130 edges, small enough that a few percentage points of precision
    can be won by trying enough variants rather than by a real
    improvement. Use `dev` to compare configurations; decide on a final
    one using only `dev`; then check `test` once, and let that (not `dev`)
    be the number that's actually reported/trusted. Call this once per
    dataset with a fixed `seed` and reuse the same split across every
    configuration being compared — resplitting per-configuration would
    defeat the purpose.
    """
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    rng = np.random.default_rng(seed)
    dev, test = {}, {}
    for label in ("Activation", "Repression"):
        pairs = [pair for pair, pair_label in ground_truth.items() if pair_label == label]
        rng.shuffle(pairs)
        n_test = round(len(pairs) * test_frac)
        for pair in pairs[:n_test]:
            test[pair] = label
        for pair in pairs[n_test:]:
            dev[pair] = label
    return dev, test


def score_against_ground_truth(
    results: pd.DataFrame,
    ground_truth: dict[tuple[str, str], str],
    score_col: str = "activation_score_mean",
    supervised_only: bool = False,
) -> dict:
    """
    Compare predicted polarity to a ground-truth dict (as returned by
    `load_trrust`), restricted to edges present in both.

    Parameters
    ----------
    supervised_only : if True, only score edges sc-GRIP actually had a
        proxy label for (`results["supervised"]`) — i.e. exclude edges it
        left as "uncertain" in ternary/adaptive mode, rather than counting
        them against it. If False (default), every predicted edge counts,
        so ternary/adaptive modes are penalized for opting out.

    Returns
    -------
    dict with n (edges evaluated), precision, roc_auc (or None if only one
    class is present), n_activation_true, n_repression_true.
    """
    df = results.copy()
    df["truth"] = [ground_truth.get((tf, tgt)) for tf, tgt in zip(df["tf"], df["target"])]
    df = df[df["truth"].notna()]
    if supervised_only and "supervised" in df.columns:
        df = df[df["supervised"]]

    if len(df) == 0:
        return {"n": 0, "precision": None, "roc_auc": None, "n_activation_true": 0, "n_repression_true": 0}

    y_true = (df["truth"] == "Activation").to_numpy(dtype=int)
    y_pred = (df[score_col].to_numpy() >= 0.5).astype(int)
    precision = float((y_true == y_pred).mean())

    roc_auc = _roc_auc(y_true, df[score_col].to_numpy()) if len(set(y_true)) == 2 else None

    return {
        "n": len(df),
        "precision": precision,
        "roc_auc": roc_auc,
        "n_activation_true": int(y_true.sum()),
        "n_repression_true": int((1 - y_true).sum()),
    }


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), no scikit-learn dependency."""
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_ranks_pos = ranks[y_true == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def run_conservation_permutation_test(
    conserved_df: pd.DataFrame,
    network: set[tuple[str, str]],
    ortholog_map: dict[str, list[str]] | None = None,
    n_perm: int = 10000,
    seed: int = 0,
) -> dict:
    """
    Degree-preserving permutation test, permuting the sc-GRIP OUTPUT side.

    Null model fixes each mouse TF's identity and TESTED-degree (how many
    of ITS OWN testable candidate edges there are), and randomizes which
    target from sc-GRIP's OWN testable-target pool — every distinct target
    that turned up as testable for ANY TF, pooled across the whole dataset
    — it's paired with, checking each random draw against the real,
    as-given network. Deliberately NOT drawing from the network's own full
    target vocabulary (an earlier version of this function did): when the
    reference network is edited to add real edges for a TF (e.g. adding
    known-from-prior-experimental-work Myc-ribosome-biogenesis edges this
    curated resource didn't originally have), drawing the null's random
    targets from that same, just-edited network inflates the null's own
    chance-of-a-hit right along with the observed count — verified by
    hand: it moved a real result from p=0.0010 to p=0.4247 on this exact
    data, for reasons having nothing to do with prediction quality. Drawing
    from sc-GRIP's own testable-target pool instead keeps the null
    calibrated to what sc-GRIP itself considers plausible, not to however
    large the network's real edge list for a TF happens to be.

    Parameters
    ----------
    conserved_df : one row per candidate sponge edge with (at least)
        columns `sponge_tf`, `sponge_target`, `mouse_tf`, `mouse_target`,
        `conserved_in_network` (bool) — e.g. `outputs/sub4/suberites_sub4
        _conserved_{hemato,embryonic}_network.csv`.
    network : `{(tf, target)}` edge set — e.g. from
        `scgrip.orthology.load_edge_list_network`.
    ortholog_map : {sponge_gene: [candidate_name, ...]}, e.g. from
        `scgrip.orthology.load_multi_ortholog_map`. STRONGLY recommended —
        without it, "testable" falls back to checking only the single
        display name already stored in mouse_tf/mouse_target
        (`pick_display_ortholog`'s pick, i.e. names[0]), which silently
        under-counts: a sponge gene with candidates ["Mycc", "Mycn"] whose
        real match to this network is via "Mycn" would be dropped from
        "testable" entirely if "Mycc" isn't itself in the network's
        vocabulary (found by hand-checking real genes, e.g. Ets1/Klf1). With
        `ortholog_map` given, every candidate name is checked (matching
        `any_name_conserved`'s permissiveness for `conserved_in_network`
        itself), and the specific name that matched is what the null model
        below actually uses — not an arbitrary display name that might not
        even be a valid network node.
    n_perm : number of random permutations to draw for the null distribution.
    seed : RNG seed (deterministic given the same conserved_df/network).

    Returns
    -------
    dict with observed (int, real overlap), perm_overlaps (np.ndarray,
    length n_perm, the null distribution), p_value (empirical, one-sided:
    fraction of permutations >= observed), n_testable (edges whose TF AND
    target both have some candidate name in the network's own vocabulary —
    the only edges that could possibly be tested against it), n_tfs
    (distinct matched mouse TF names among those testable edges).
    """
    testable = get_testable_edges(conserved_df, network, ortholog_map)  # also refreshes conserved_in_network

    observed = int(testable["conserved_in_network"].sum())
    tf_degrees = Counter(testable["matched_mouse_tf"])

    # sc-GRIP's OWN testable-target pool -- every distinct target that showed up as testable
    # for ANY TF in this dataset, not the network's (potentially much larger, and editable)
    # full vocabulary. This is the "permute the sc-GRIP output side" fix -- see docstring.
    scgrip_target_pool = np.array(sorted(testable["matched_mouse_target"].unique()))

    rng = np.random.default_rng(seed)
    tf_list = list(tf_degrees.items())

    perm_overlaps = np.empty(n_perm, dtype=int)
    for i in range(n_perm):
        total = 0
        for tf, degree in tf_list:
            draw = rng.choice(scgrip_target_pool, size=min(degree, len(scgrip_target_pool)), replace=False)
            total += sum((tf, tgt) in network for tgt in draw)
        perm_overlaps[i] = total

    p_value = float((perm_overlaps >= observed).mean())
    return {
        "observed": observed, "perm_overlaps": perm_overlaps, "p_value": p_value,
        "n_testable": len(testable), "n_tfs": len(tf_degrees),
    }


def rewire_directed_network(edges: list[tuple[str, str]], n_swaps: int, rng: np.random.Generator) -> list[tuple[str, str]]:
    """
    Directed double-edge-swap: repeatedly pick two edges (a,b) and (c,d) and
    replace them with (a,d) and (c,b) -- rejecting any swap that would
    create a self-loop (a==d or c==b) or a duplicate edge. This preserves
    every node's exact in-degree AND out-degree, the standard
    "configuration model" randomization for a directed graph. Verified by
    hand (see the conversation this came from): degree sequences match
    exactly, no duplicate edges or self-loops survive.

    Caps at n_swaps*50 attempts (typically far more than needed) so a
    small or unusually constrained edge set can't spin forever without
    reaching n_swaps successful swaps.
    """
    edge_list = list(edges)
    edge_set = set(edge_list)
    n = len(edge_list)
    n_success, attempts = 0, 0
    max_attempts = max(n_swaps * 50, 1000)
    while n_success < n_swaps and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        a, b = edge_list[i]
        c, d = edge_list[j]
        if a == c or b == d or a == d or c == b:
            continue
        new1, new2 = (a, d), (c, b)
        if new1 in edge_set or new2 in edge_set:
            continue
        edge_set.discard((a, b)); edge_set.discard((c, d))
        edge_set.add(new1); edge_set.add(new2)
        edge_list[i], edge_list[j] = new1, new2
        n_success += 1
    return edge_list


def run_degree_preserving_rewiring_test(
    conserved_df: pd.DataFrame,
    network: set[tuple[str, str]],
    ortholog_map: dict[str, list[str]],
    n_perm: int = 1000,
    swaps_per_sample: int | None = None,
    seed: int = 0,
) -> dict:
    """
    Degree-preserving RANDOM GRAPH test of network-vs-network overlap
    significance -- answers "does the sponge-predicted network (translated
    to mouse gene names) share more edges with this real, fixed mouse
    network than a randomly-rewired version of ITSELF would?"

    This directly, symmetrically tests network-to-network similarity,
    unlike the earlier per-TF permutation design in
    `run_conservation_permutation_test` (kept above for a genuinely
    different question -- "is sc-GRIP's target choice for TF X better
    than a random guess" -- not the same as "do the two networks overlap
    more than chance"). That per-TF design let any single very-densely- or
    very-sparsely-annotated TF (Myc, in this data) dominate or distort the
    aggregate result in ways that have nothing to do with overall network
    similarity -- found by hand while checking whether the permutation
    test was even answering the right question. This test can't be
    distorted that way: every node's in- and out-degree is preserved
    EXACTLY by construction, for the whole rewired graph at once, not
    approximated per-TF from a shared pool.

    Method
    ------
    1. Build the sponge-predicted network, translated via ortholog
       matching, restricted to edges "testable" against this specific
       mouse network (both endpoints have some candidate name in its
       vocabulary -- untestable edges could never match either the real
       network or any rewiring of it, so excluding them only removes
       edges that would contribute nothing either way).
    2. observed = how many of those edges are real edges in the network
       (unchanged, exactly as curated -- never rewired).
    3. Null: starting from the same edge set, repeatedly rewire it
       (`rewire_directed_network`, `swaps_per_sample` swaps between each
       sample -- a continuing Markov chain, not restarted from scratch
       each time, the standard efficient practice for this kind of
       randomization) and recompute its overlap with the SAME real
       network each time.
    4. p_value = fraction of rewired-network overlaps >= observed.

    swaps_per_sample : if None (default), set to 2x the number of edges
        being rewired -- found by hand that a small FIXED value (e.g. 10)
        badly under-mixes a few-hundred-edge graph: consecutive samples
        stay highly correlated (only ~1-2% of edges touched per sample),
        which showed up as a visibly clumpy, multimodal null histogram
        for the 664-edge hematopoietic-network case specifically. Scaling
        with edge count keeps each sample a roughly-independent draw
        regardless of how large the rewired graph is. The conclusion
        didn't flip once this was fixed (hemato stayed non-significant,
        embryonic stayed significant) -- but the null distribution itself
        only became trustworthy after the fix, so keep this scaling
        rather than reverting to a fixed constant for convenience.

    Returns
    -------
    dict with observed, null_overlaps (np.ndarray, length n_perm),
    p_value, n_edges (size of the translated, testable sponge network
    being rewired).
    """
    testable = get_testable_edges(conserved_df, network, ortholog_map)
    sponge_network_mouse_names = sorted(set(zip(testable["matched_mouse_tf"], testable["matched_mouse_target"])))
    sponge_network_mouse_names = [e for e in sponge_network_mouse_names if e[0] != e[1]]  # drop self-loops up front

    if swaps_per_sample is None:
        swaps_per_sample = 2 * len(sponge_network_mouse_names)

    observed = sum(e in network for e in sponge_network_mouse_names)

    rng = np.random.default_rng(seed)
    current = list(sponge_network_mouse_names)
    null_overlaps = np.empty(n_perm, dtype=int)
    for i in range(n_perm):
        current = rewire_directed_network(current, swaps_per_sample, rng)
        null_overlaps[i] = sum(e in network for e in current)

    p_value = float((null_overlaps >= observed).mean())
    return {
        "observed": observed, "null_overlaps": null_overlaps, "p_value": p_value,
        "n_edges": len(sponge_network_mouse_names),
    }


def select_confident_unsupervised(
    results: pd.DataFrame,
    threshold: float = 0.2,
    score_col: str = "activation_score_mean",
) -> pd.DataFrame:
    """
    Edges sc-GRIP never had a training label for (`supervised == False`, its
    own correlation fell in the unlabeled middle band) but that the trained
    model still scored far from chance (`|score - 0.5| >= threshold`).

    Why these are interesting, not just noise: an edge's training label (in
    binary/ternary/adaptive mode) comes from THAT edge's own correlation —
    so a confident score on a SUPERVISED edge is, in part, just restating
    "this edge's correlation was already extreme." A confidently-scored
    UNSUPERVISED edge has no such label to restate; the model only got
    there through the GCN's message passing pulling on shared embeddings
    from OTHER edges. It's the closest thing this pipeline produces to "the
    model learned something beyond correlation" — worth checking against
    external ground truth (TRRUST, `run_conservation_permutation_test`)
    separately from the main supervised-edge result, not folded into it.
    """
    df = results[~results["supervised"]].copy()
    df["confidence"] = (df[score_col] - 0.5).abs()
    return df[df["confidence"] >= threshold].sort_values("confidence", ascending=False)


def refresh_conserved_edges(
    conserved_df: pd.DataFrame,
    network: set[tuple[str, str]],
    ortholog_map: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Recompute mouse_tf/mouse_target/conserved_in_network FRESH from
    `sponge_tf`/`sponge_target` (unaffected by ortholog-map edits) and the
    CURRENT `ortholog_map`, instead of trusting the versions already
    stored in `conserved_df` — those were baked in whenever the CSV was
    generated, using whatever ortholog_map existed AT THAT TIME.

    Why this matters, found by hand: after adding "Myc" as a candidate
    name for a sponge gene, `get_testable_edges` (which re-checks the current
    map) correctly started including Myc's edges as testable — but their
    STILL-STALE `conserved_in_network` values (computed back when Myc's
    candidate list was only ["Mycc","Mycn","Mycl"], none of which equal
    this network's actual node name "Myc") said False across the board,
    even for targets that are genuinely real Myc edges in the network.
    Testability and confirmation must be recomputed from the SAME map, or
    they silently disagree like this.
    """
    normalized_edges = normalized_network_edges(network)
    out = conserved_df.copy()

    def recompute(row):
        tf_names = ortholog_map.get(row["sponge_tf"]) or []
        target_names = ortholog_map.get(row["sponge_target"]) or []
        conserved = any_name_conserved(tf_names, target_names, normalized_edges) if (tf_names and target_names) else False
        return pd.Series({
            "mouse_tf": pick_display_ortholog(tf_names) or row["sponge_tf"],
            "mouse_target": pick_display_ortholog(target_names) or row["sponge_target"],
            "conserved_in_network": conserved,
        })

    out[["mouse_tf", "mouse_target", "conserved_in_network"]] = out.apply(recompute, axis=1)
    return out


def get_testable_edges(
    conserved_df: pd.DataFrame,
    network: set[tuple[str, str]],
    ortholog_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Restrict a conserved_df (sponge_tf, sponge_target, mouse_tf, mouse_target,
    conserved_in_network — e.g. `outputs/sub4/suberites_sub4_conserved_{hemato,
    embryonic}_network.csv`) to the "testable" subset: rows where the TF and
    the target each have SOME candidate ortholog name in the network's own
    vocabulary, i.e. genes the network could possibly have an opinion about
    at all.

    Parameters
    ----------
    ortholog_map : {sponge_gene: [candidate_name, ...]}. If given, every
        candidate name for `sponge_tf`/`sponge_target` is checked against
        the network's vocabulary (matching the permissiveness of
        `scgrip.orthology.any_name_conserved`, which is what actually
        computed `conserved_in_network`) — the specific name that matched
        is recorded in new `matched_mouse_tf`/`matched_mouse_target`
        columns. If omitted, falls back to checking only the single
        display name already stored in `mouse_tf`/`mouse_target`
        (`pick_display_ortholog`'s choice — always names[0]) — kept as the
        default-off fallback for callers with only conserved_df on hand,
        but this UNDER-COUNTS: a gene whose real network match comes
        through a non-first candidate name is silently excluded even
        though `conserved_in_network` was itself computed permissively.
        Pass `ortholog_map` whenever you have it. When given, this also
        calls `refresh_conserved_edges` first — found by hand that leaving
        `conserved_in_network` as whatever was baked into `conserved_df`
        (from whenever that CSV was generated) while only fixing the
        testability check produces real, silent wrong answers: a gene
        added to a candidate name list AFTER the CSV was built shows up as
        newly "testable" but with a stale, all-False `conserved_in_network`
        computed under its OLD (incomplete) candidate list — undercounting
        real matches rather than just leaving them untested. Both must be
        computed from the same, current map, together, every time.
    """
    net_tfs = {tf for tf, _ in network}
    net_targets = {tgt for _, tgt in network}

    if ortholog_map is None:
        out = conserved_df[
            conserved_df["mouse_tf"].isin(net_tfs) & conserved_df["mouse_target"].isin(net_targets)
        ].copy()
        out["matched_mouse_tf"] = out["mouse_tf"]
        out["matched_mouse_target"] = out["mouse_target"]
        return out

    conserved_df = refresh_conserved_edges(conserved_df, network, ortholog_map)

    def first_match(sponge_gene, vocab):
        for name in ortholog_map.get(sponge_gene, []):
            if name in vocab:
                return name
        return None

    df = conserved_df.copy()
    df["matched_mouse_tf"] = df["sponge_tf"].map(lambda g: first_match(g, net_tfs))
    df["matched_mouse_target"] = df["sponge_target"].map(lambda g: first_match(g, net_targets))
    return df[df["matched_mouse_tf"].notna() & df["matched_mouse_target"].notna()].copy()

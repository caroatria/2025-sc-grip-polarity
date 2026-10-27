"""
Cross-species comparison via ortholog mapping
================================================
sc-GRIP's whole value proposition is inferring regulatory interactions in
non-model species where no curated ground truth exists. But that also
means predictions on such a species can't be scored against a TRRUST-like
database directly — the gene IDs don't match anything. This module
bridges that gap: translate a non-model species' predicted edges into a
model species' gene symbols via a reciprocal-BLAST ortholog table, then
compare against a curated network from that model species (existence of
the interaction, since these curated networks are typically unsigned —
no activation/repression annotation to compare polarity against).

This is deliberately generic (column names are parameters, not
hardcoded); the motivating case was a sponge-to-mouse reciprocal-best-hit
BLAST ortholog table.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import pandas as pd


def load_ortholog_map(
    path: str,
    source_col: str,
    target_col: str,
    sheet_name=0,
) -> dict[str, str]:
    """
    Load a `source_gene -> target_gene_symbol` mapping from a BLAST-based
    ortholog table (Excel or CSV). Rows with a missing `target_col` are
    dropped — no ortholog call for that gene.
    """
    path = str(path)
    df = pd.read_excel(path, sheet_name=sheet_name) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df = df[[source_col, target_col]].dropna()
    return dict(zip(df[source_col], df[target_col]))


def translate_edges(
    edges: list[tuple[str, str]],
    gene_map: dict[str, str],
) -> tuple[list[tuple[str, str]], int]:
    """
    Translate a list of (src, dst) gene-ID edges via `gene_map`. An edge
    is dropped if either endpoint has no ortholog call.

    Returns
    -------
    translated : list of (src_ortholog, dst_ortholog) pairs.
    n_dropped  : how many input edges were dropped for lacking an
        ortholog on at least one side (useful to report — most candidate
        genes in a non-model species typically won't have one).
    """
    translated = []
    dropped = 0
    for src, dst in edges:
        if src in gene_map and dst in gene_map:
            translated.append((gene_map[src], gene_map[dst]))
        else:
            dropped += 1
    return translated, dropped


def load_multi_ortholog_map(path: str) -> dict[str, list[str]]:
    """
    Load a `gene -> [ortholog_name, ...]` mapping from a JSON file of the form
    `{"gene_id": "Name1, Name2", ...}`. Unlike `load_ortholog_map`, this is
    not 1:1: each gene's value is a comma-joined string
    that can list more than one candidate name (e.g. both a mouse and a
    human BLAST hit, or paralog ambiguity) — kept as a list here rather
    than silently collapsed to one.
    """
    with open(path) as f:
        raw = json.load(f)
    return {gene: [n.strip() for n in str(names).split(",") if n.strip()] for gene, names in raw.items()}


def normalize_gene_name(name) -> str:
    """
    Strip a trailing numeric motif-variant suffix (`_1`, `_2`, ...) and
    uppercase — for case/variant-insensitive matching between TF motif
    column names, ortholog-table symbols, and reference-network gene
    symbols (which don't share a consistent case/suffix convention).
    """
    if not isinstance(name, str):
        return name
    return re.sub(r"_[0-9]+$", "", name).upper()


def invert_multi_ortholog_map(sponge_to_names: dict[str, list[str]]) -> dict[str, set[str]]:
    """
    Invert a `sponge_gene -> [name, ...]` map into
    `normalized_name -> {sponge_gene, ...}` — for looking up which sponge
    gene(s) a TF motif/symbol name corresponds to.
    """
    out: dict[str, set[str]] = defaultdict(set)
    for gene, names in sponge_to_names.items():
        for name in names:
            out[normalize_gene_name(name)].add(gene)
    return out


def pick_display_ortholog(names: list[str]) -> str | None:
    """The first candidate ortholog name, used as the single display
    symbol for a sponge gene that may carry several candidate names."""
    return names[0] if names else None


def normalized_network_edges(edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Normalize a `{(tf, target), ...}` network edge set for
    case/variant-insensitive membership checks."""
    return {(normalize_gene_name(a), normalize_gene_name(b)) for a, b in edges}


def any_name_conserved(
    tf_names: list[str],
    target_names: list[str],
    normalized_edges: set[tuple[str, str]],
) -> bool:
    """
    True if ANY combination of a sponge gene's candidate TF names and its
    target's candidate names forms an edge in `normalized_edges` (from
    `normalized_network_edges`). A sponge gene with several ortholog
    candidates only needs one of them to line up with the network.
    """
    return any(
        (normalize_gene_name(t), normalize_gene_name(g)) in normalized_edges
        for t in tf_names
        for g in target_names
    )


# --- permissive (family-level) matching -------------------------------------
#
# The "stringent" check above requires an EXACT (normalized) name match — the
# ortholog table's own BLAST/phylogeny call. That's the trustworthy default.
# But TF paralog families are common (Sox2/Sox4/Sox8/..., Klf1/Klf2/Klf4/...,
# Jun/Junb/Jund), and a sponge gene's specific ortholog call can land on a
# different family member than the one a given curated network happens to
# name — not a wrong call, just a different (often equally plausible) paralog
# pick. `any_name_conserved_permissive` additionally credits a match when a
# candidate name's gene-family root matches a network gene's root, even
# without an exact name match. This trades specificity for recall and should
# always be reported ALONGSIDE the strict result, never in place of it —
# treat it as a sensitivity check, not a replacement.
#
# Two false-positive traps found by manual review: a short/generic shared prefix can coincide by chance rather than
# reflect real homology (Cited2 vs Cit, Gps2 vs Gps1), and a gene named after
# another as a binding partner or induced target is not that gene's paralog
# (Myb vs Mybbp1a, Rb1 vs Rb1cc1, Tnf vs Tnfaip3/8). These specific roots are
# excluded by default; treat any NEW family match this turns up with the same
# scrutiny before trusting it.

KNOWN_FALSE_FAMILY_ROOTS = {"CIT", "MYBBP", "RB", "TNFAIP", "GPS"}


def gene_family_key(name, min_len: int = 3) -> str | None:
    """
    The leading letters of a normalized gene name up to its first digit
    (e.g. "SOX2" -> "SOX", "FOXC1" -> "FOXC", "JUND" -> "JUND" — a name with
    no digit at all just keeps its full form here; catching Jun/Jund-style
    no-digit paralogs is what the bidirectional-prefix check inside
    `any_name_conserved_permissive` is for). Returns None for a root shorter
    than `min_len` — too short/generic to trust as a family signal (e.g. the
    "T" in Brachyury, or "SP" alone) — and for any of `KNOWN_FALSE_FAMILY_ROOTS`.
    """
    n = normalize_gene_name(name)
    if not isinstance(n, str):
        return None
    m = re.match(r"^[A-Za-z]+", n)
    key = m.group() if m else n
    if len(key) < min_len or key in KNOWN_FALSE_FAMILY_ROOTS:
        return None
    return key


def _family_keys(names: list[str], min_len: int = 3) -> set[str]:
    keys = set()
    for n in names:
        k = gene_family_key(n, min_len)
        if k:
            keys.add(k)
        # also add the bidirectional-prefix form for no-digit letter-suffix
        # paralogs (Jun/Junb/Jund) that share no digit-delimited root
        norm = normalize_gene_name(n)
        if isinstance(norm, str) and len(norm) >= min_len and norm not in KNOWN_FALSE_FAMILY_ROOTS:
            keys.add(norm)
    return keys


def build_family_edge_index(normalized_edges: set[tuple[str, str]], min_len: int = 3) -> dict[str, set[str]]:
    """
    Precompute `tf_family_key -> {target_family_key, ...}` from a
    normalized network edge set, so `any_name_conserved_permissive` can look
    up family-level matches in O(1) per candidate instead of rescanning the
    whole network for every sponge edge.
    """
    index: dict[str, set[str]] = defaultdict(set)
    for tf, target in normalized_edges:
        tf_key = gene_family_key(tf, min_len)
        target_key = gene_family_key(target, min_len)
        if tf_key and target_key:
            index[tf_key].add(target_key)
    return index


def any_name_conserved_permissive(
    tf_names: list[str],
    target_names: list[str],
    normalized_edges: set[tuple[str, str]],
    family_index: dict[str, set[str]],
    min_len: int = 3,
) -> tuple[bool, str]:
    """
    Like `any_name_conserved`, but falls back to a family-level match when no
    exact match exists: true if the network has SOME edge whose TF shares a
    family root with one of `tf_names` AND whose target shares a family root
    with one of `target_names` (using a *bidirectional prefix* for no-digit
    letter-suffix families like Jun/Jund, in addition to the digit-delimited
    root). Returns `(matched, match_type)` with `match_type` one of
    `"exact"`, `"family"`, `"none"` — always keep this alongside the plain
    exact check in output, never report only the permissive result.
    """
    if any_name_conserved(tf_names, target_names, normalized_edges):
        return True, "exact"

    tf_keys = _family_keys(tf_names, min_len)
    target_keys = _family_keys(target_names, min_len)
    for tf_key in tf_keys:
        if target_keys & family_index.get(tf_key, set()):
            return True, "family"
    return False, "none"


def load_edge_list_network(
    path: str,
    src_col: str = "node1",
    dst_col: str = "node2",
    edge_type_col: str = "Edge_type",
    keep_type: str = "TF_Target",
    sep: str = "\t",
) -> set[tuple[str, str]]:
    """
    Load a plain directed edge-list network (e.g. a curated TF-target
    network export with mixed edge types) into a `{(src, dst)}` set,
    keeping only rows whose `edge_type_col` equals `keep_type`.
    """
    df = pd.read_csv(path, sep=sep)
    df = df[df[edge_type_col] == keep_type]
    return set(zip(df[src_col], df[dst_col]))

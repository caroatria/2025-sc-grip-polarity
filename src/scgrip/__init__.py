"""
sc-GRIP: Single-cell Gene Regulation Interaction Polarity
==========================================================
A graph-convolutional framework that scores transcription-factor-to-target
interactions from single-cell expression data, without training labels, for
use in non-model species. Based on Atria et al., ICDM 2025, but substantially
redesigned since (see CHANGELOG.md).

    from scgrip import ScGRIP, ScGRIPConfig

    model = ScGRIP.from_files("expression.csv", "tf_interactions.csv")
    results = model.fit()
    model.summary(results)

Submodules are imported lazily, so lightweight helpers (e.g. `scgrip.orthology`,
`scgrip.evaluate`) work without importing torch.
"""

from __future__ import annotations

import importlib

__version__ = "0.2.0"

# public name -> submodule that defines it
_EXPORTS = {
    # model and inference
    "ScGRIP": "inference", "ScGRIPConfig": "inference", "FitResult": "inference",
    "GAEModel": "model", "GCNEncoder": "model", "BilinearDecoder": "model",
    # edge existence (link prediction)
    "ExistenceConfig": "existence", "score_candidate_edges": "existence",
    "candidate_pairs": "existence", "clr_normalize": "existence",
    # data
    "read_expression_matrix": "data", "read_adjacency_matrix": "data", "align_genes": "data",
    "build_edge_index": "data", "build_cell_graphs": "data",
    "correlation_proxy_labels": "data", "correlation_proxy_labels_ternary": "data",
    "correlation_proxy_labels_adaptive": "data",
    "logic_proxy_labels": "logic_labels", "fit_boolean_roles": "logic_labels",
    "binarize_matrix": "logic_labels",
    # preprocessing
    "read_obs_categorical": "celltype", "restrict_to_cell_type": "celltype",
    "pseudobulk": "pseudobulk",
    # evaluation
    "load_trrust": "evaluate", "restrict_ground_truth": "evaluate",
    "split_ground_truth": "evaluate", "score_against_ground_truth": "evaluate",
    # orthology (non-model species)
    "load_ortholog_map": "orthology", "translate_edges": "orthology",
    "load_edge_list_network": "orthology", "load_multi_ortholog_map": "orthology",
    "normalize_gene_name": "orthology", "invert_multi_ortholog_map": "orthology",
    "pick_display_ortholog": "orthology", "normalized_network_edges": "orthology",
    "any_name_conserved": "orthology", "any_name_conserved_permissive": "orthology",
    "build_family_edge_index": "orthology", "gene_family_key": "orthology",
}

__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name: str):
    if name in _EXPORTS:
        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'scgrip' has no attribute {name!r}")


def __dir__():
    return __all__

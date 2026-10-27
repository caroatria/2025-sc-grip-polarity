import numpy as np
import pandas as pd

from scgrip.data import align_genes, build_edge_index


def test_edges_map_to_gene_nodes_not_matrix_positions():
    # TFs are not the first genes alphabetically, so matrix positions != node indices
    expr = pd.DataFrame(np.random.default_rng(0).random((5, 4)), columns=["A", "B", "ZTF", "YTF"])
    adj = pd.DataFrame([[1, 0], [0, 1]], index=["YTF", "ZTF"], columns=["A", "B"])
    expr, adj = align_genes(expr, adj)
    ei = build_edge_index(expr, adj)
    genes = ei.genes
    pairs = {(genes[s], genes[d]) for s, d in ei.directed.T.tolist()}
    assert pairs == {("YTF", "A"), ("ZTF", "B")}
    assert ei.names == ["YTF->A", "ZTF->B"]


def test_self_loops_are_dropped():
    expr = pd.DataFrame(np.ones((3, 2)), columns=["T", "G"])
    adj = pd.DataFrame([[1, 1]], index=["T"], columns=["T", "G"])
    expr, adj = align_genes(expr, adj)
    assert build_edge_index(expr, adj).names == ["T->G"]

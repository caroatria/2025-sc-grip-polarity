from scgrip import ScGRIP, ScGRIPConfig
from scgrip.synthetic import make_combinatorial_dataset


def _labels(mode):
    d = make_combinatorial_dataset(seed=0)
    m = ScGRIP(d.expr_df, d.edge_df, ScGRIPConfig(label_mode=mode, n_runs=1, num_epochs=1, verbose=False, seed=0))
    known = dict(zip(m.edge_index.names, m.known_mask))
    labelled = [n for n, k in known.items() if k]
    return known, dict(zip(labelled, m.labels.tolist()))


def test_logic_labels_recover_and_gate():
    known, labels = _labels("logic")
    assert labels == {"TF1->GeneAND": 1.0, "TF2->GeneAND": 1.0}
    assert not known["TF3->GeneAND"]  # decoy TF stays unlabelled

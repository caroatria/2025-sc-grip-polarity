import numpy as np

from scgrip.existence import ExistenceConfig, candidate_pairs, clr_normalize, proxy_labels, score_candidate_edges
from scgrip.synthetic import make_synthetic_dataset


def test_candidate_pairs_exclude_self_loops():
    src, dst = candidate_pairs(["A", "B", "C"], ["A", "C", "X"])
    assert len(src) == 4 and not (src == dst).any()


def test_per_tf_labels_are_balanced_across_regulators():
    rng = np.random.default_rng(0)
    src = np.repeat([0, 1, 2], 100)
    score = np.r_[rng.random(100) + 10, rng.random(100), rng.random(100)]  # regulator 0 is a "hub"
    pos_global, _ = proxy_labels(score, src, "global", 0.1, 0.5)
    pos_tf, neg_tf = proxy_labels(score, src, "per_tf", 0.1, 0.5)
    assert set(src[pos_global]) == {0}
    assert [int(pos_tf[src == r].sum()) for r in range(3)] == [10, 10, 10]
    assert not (pos_tf & neg_tf).any()


def test_clr_removes_row_offset():
    src, dst = np.array([0, 0, 1, 1]), np.array([2, 3, 2, 3])
    a = clr_normalize(np.array([1.0, 2.0, 11.0, 12.0]), src, dst, 4)
    assert np.isfinite(a).all() and a.shape == (4,)


def test_score_candidate_edges_runs_both_feature_modes():
    d = make_synthetic_dataset(n_cells=60, seed=0)
    tfs = list(d.edge_df.index)
    for features in ("per_cell", "profile"):
        cfg = ExistenceConfig(features=features, n_runs=1, epochs=1, profile_steps=5, pos_frac=0.2, clr=True)
        res = score_candidate_edges(d.expr_df, tfs, cfg)
        assert len(res) == len(tfs) * (d.expr_df.shape[1] - 1)
        assert {"tf", "target", "score", "score_raw", "abs_spearman", "train_positive"} <= set(res.columns)
        assert np.isfinite(res.score).all()

import numpy as np

from scgrip import ScGRIP, ScGRIPConfig
from scgrip.synthetic import make_synthetic_dataset


def test_polarity_separates_activation_from_repression():
    d = make_synthetic_dataset(n_cells=200, seed=0)
    res = ScGRIP(d.expr_df, d.edge_df, ScGRIPConfig(n_runs=1, num_epochs=10, seed=0, verbose=False)).fit()
    truth = res["edge"].map(d.truth)
    act = res.loc[truth == "activation", "activation_score_mean"]
    rep = res.loc[truth == "repression", "activation_score_mean"]
    assert len(res) == 6 and np.isfinite(res.activation_score_mean).all()
    assert act.min() > rep.max()

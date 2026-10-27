"""
sc-GRIP quick start on synthetic data (runs in about a minute on a laptop CPU).

1. Polarity: 4 TFs, 6 targets, known activating / repressing edges.
2. Edge existence: score every TF -> gene pair, label-free.
"""
from scgrip import ExistenceConfig, ScGRIP, ScGRIPConfig, score_candidate_edges
from scgrip.synthetic import make_synthetic_dataset

data = make_synthetic_dataset(n_cells=300, seed=0)

# 1. polarity on the known candidate network
model = ScGRIP(data.expr_df, data.edge_df, ScGRIPConfig(n_runs=2, num_epochs=20, seed=0, verbose=False))
res = model.fit()
res["truth"] = res["edge"].map(data.truth)
print(res[["edge", "activation_score_mean", "correlation", "predicted_polarity", "truth"]].to_string(index=False))

# 2. edge existence over all TF -> gene pairs
tfs = list(data.edge_df.index)
edges = score_candidate_edges(data.expr_df, tfs, ExistenceConfig(n_runs=1, epochs=5, pos_frac=0.2))
edges["true_edge"] = (edges.tf + "->" + edges.target).isin(data.truth)
print(edges.sort_values("score", ascending=False).head(8).to_string(index=False))

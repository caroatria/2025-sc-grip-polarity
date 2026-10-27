"""
sc-GRIP inference pipeline
============================
High-level, scikit-learn-style entry point for the sc-GRIP method
(Atria et al., ICDM 2025). Wraps data preparation (`scgrip.data`) and the
graph autoencoder (`scgrip.model`) into Algorithm 1 from the paper:

    for run in 1..R:
        for epoch in 1..T:
            for each cell graph (X_i, E):
                Z_i = Encoder(X_i, E)
                Y_hat_i = Decoder(Z_i, E)
                loss = BCE(Y_hat_i, Y)          # Y = correlation proxy labels
                update encoder/decoder
        Y_hat(run) = median over cells of the trained model's predictions
    Y_final = aggregate over runs (mean +/- std, and median)

Usage
-----
    from scgrip import ScGRIP, ScGRIPConfig

    model = ScGRIP.from_files("expression.csv", "tf_interactions.csv")
    results = model.fit()
    model.summary(results)
    results.to_csv("predictions.csv", index=False)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .data import (
    EdgeIndex,
    add_reverse_edges,
    align_genes,
    build_cell_graphs,
    build_edge_index,
    correlation_proxy_labels,
    correlation_proxy_labels_adaptive,
    correlation_proxy_labels_ternary,
    read_adjacency_matrix,
    read_expression_matrix,
)
from .logic_labels import logic_proxy_labels
from .model import GAEModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScGRIPConfig:
    # Architecture
    hidden_channels: int = 128
    out_channels: int = 64

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    batch_size: int = 16

    # Algorithm 1: multi-run training for a robust consensus score
    n_runs: int = 5
    cell_aggregation: str = "median"     # how per-cell predictions are pooled within a run

    # Proxy labels (paper Eq. 1, plus an adaptive variant — see data.py).
    # Default is "adaptive", not the paper's "binary": on sparse single-cell
    # data a fixed threshold (tau=0.5) labels almost every edge "repression",
    # because few TF-target correlations ever reach 0.5. "adaptive" thresholds
    # on the dataset's own correlation quantiles instead (docs/design_notes.md).
    # Cell-type restriction and TF-detection conditioning (celltype.py,
    # condition_on_tf_detected below) helped on some datasets and hurt on
    # others, so they stay opt-in.
    label_mode: str = "adaptive"         # "binary", "ternary", "adaptive", or "logic"
    tau: float = 0.5                     # binary: activation if corr > tau
    tau_low: float = -0.15               # ternary: repression if corr <= tau_low
    tau_high: float = 0.15               # ternary: activation if corr >= tau_high
    low_q: float = 0.3                   # adaptive: bottom low_q quantile -> repression
    high_q: float = 0.7                  # adaptive: top (1 - high_q) quantile -> activation
    corr_method: str = "pearson"
    condition_on_tf_detected: bool = False   # compute each edge's correlation only over
                                              # cells where its TF is actually detected
    min_cells_for_corr: int = 20             # below this many qualifying cells, treat the
                                              # edge's correlation as inestimable, not as 0

    # "logic" mode (scgrip.logic_labels): per-target combinatorial Boolean
    # rule fit over all of that target's regulating TFs at once, instead of
    # one TF at a time via correlation — see logic_labels.py module docstring.
    logic_min_cells: int = 30
    logic_min_f1_cv: float = 0.55
    logic_max_full_enum_n: int = 4
    logic_binarize: str = "detected"         # "detected" (default) or "bimodal" — see logic_labels.py
    logic_min_expressing_frac: float = 0.05

    # Graph construction
    use_reverse_edges: bool = True       # undirected message passing (see model.GAEModel)
    normalize_expression: bool = True

    device: str = "cpu"
    seed: Optional[int] = None
    verbose: bool = True

    def __post_init__(self):
        if self.label_mode not in ("binary", "ternary", "adaptive", "logic"):
            raise ValueError(
                f"label_mode must be 'binary', 'ternary', 'adaptive', or 'logic', got {self.label_mode!r}"
            )
        if self.cell_aggregation not in ("median", "mean"):
            raise ValueError(f"cell_aggregation must be 'median' or 'mean', got {self.cell_aggregation!r}")


@dataclass
class FitResult:
    """Raw artifacts from `ScGRIP.fit()`, in case more than the summary
    DataFrame is needed (e.g. for custom plots or further aggregation)."""
    edge_index: EdgeIndex
    known_mask: np.ndarray               # bool, one per edge; False where unsupervised/inestimable
    corr: np.ndarray                     # correlation value used to derive each edge's label
    per_run_scores: np.ndarray           # [n_runs, n_edges], within-run cell-aggregated scores
    dataframe: pd.DataFrame = field(repr=False)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ScGRIP:
    """
    Infer TF-target interaction polarity from single-cell expression data.

    Parameters
    ----------
    expr_df : cells x genes expression matrix (see `scgrip.data.read_expression_matrix`).
    edge_df : TFs x target genes adjacency matrix (see `scgrip.data.read_adjacency_matrix`).
    config  : `ScGRIPConfig`; defaults are used when omitted.
    """

    def __init__(self, expr_df: pd.DataFrame, edge_df: pd.DataFrame, config: Optional[ScGRIPConfig] = None):
        self.config = config or ScGRIPConfig()
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        expr_df, edge_df = align_genes(expr_df, edge_df)
        self.expr_df = expr_df
        self.edge_df = edge_df
        self.edge_index = build_edge_index(expr_df, edge_df)

        self._conv_edge_index = (
            add_reverse_edges(self.edge_index.directed)
            if self.config.use_reverse_edges
            else self.edge_index.directed
        )
        self._decode_edge_index = self.edge_index.directed

        self.dataset = build_cell_graphs(
            expr_df, self._conv_edge_index, normalize=self.config.normalize_expression
        )
        for data in self.dataset:
            data.decode_edge_index = self._decode_edge_index

        self.labels, self.known_mask, self.corr = self._compute_labels()
        self._fit_result: Optional[FitResult] = None

        if self.config.verbose:
            n_known = int(self.known_mask.sum())
            print(
                f"sc-GRIP: {len(self.dataset)} cells, {len(self.edge_index.genes)} genes, "
                f"{self.edge_index.n_edges} candidate edges "
                f"({n_known} with a supervised {self.config.label_mode} proxy label)."
            )

    @classmethod
    def from_files(
        cls,
        expression_path: str,
        adjacency_path: str,
        adata_path: Optional[str] = None,
        config: Optional[ScGRIPConfig] = None,
    ) -> "ScGRIP":
        """Convenience constructor that loads both inputs from disk."""
        expr_df = read_expression_matrix(expression_path, adata_path=adata_path)
        edge_df = read_adjacency_matrix(adjacency_path)
        return cls(expr_df, edge_df, config=config)

    def _compute_labels(self) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        cfg = self.config
        detection_kwargs = dict(
            condition_on_tf_detected=cfg.condition_on_tf_detected, min_cells=cfg.min_cells_for_corr
        )

        if cfg.label_mode == "binary":
            return correlation_proxy_labels(
                self.expr_df, self.edge_index, tau=cfg.tau, method=cfg.corr_method, **detection_kwargs
            )

        if cfg.label_mode == "ternary":
            return correlation_proxy_labels_ternary(
                self.expr_df, self.edge_index, tau_low=cfg.tau_low, tau_high=cfg.tau_high,
                method=cfg.corr_method, **detection_kwargs
            )

        if cfg.label_mode == "adaptive":
            return correlation_proxy_labels_adaptive(
                self.expr_df, self.edge_index, low_q=cfg.low_q, high_q=cfg.high_q,
                method=cfg.corr_method, **detection_kwargs
            )

        return logic_proxy_labels(
            self.expr_df, self.edge_index, min_cells=cfg.logic_min_cells,
            min_f1_cv=cfg.logic_min_f1_cv, max_full_enum_n=cfg.logic_max_full_enum_n,
            binarize=cfg.logic_binarize, min_expressing_frac=cfg.logic_min_expressing_frac,
            seed=cfg.seed or 0,
        )

    # -- training --------------------------------------------------------

    def _train_one_run(self) -> GAEModel:
        cfg = self.config
        n_edges = self.edge_index.n_edges
        known_idx = torch.tensor(np.where(self.known_mask)[0], dtype=torch.long)
        labels = self.labels.to(cfg.device)

        model = GAEModel(in_channels=1, hidden_channels=cfg.hidden_channels, out_channels=cfg.out_channels)
        model.to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss()
        loader = DataLoader(self.dataset, batch_size=cfg.batch_size, shuffle=True)

        for epoch in range(1, cfg.num_epochs + 1):
            model.train()
            total_loss, n_batches = 0.0, 0
            for batch in loader:
                batch = batch.to(cfg.device)
                logits = model(batch.x, batch.edge_index, batch.decode_edge_index)
                logits = logits.view(batch.num_graphs, n_edges)

                logits_known = logits[:, known_idx]
                targets = labels.unsqueeze(0).expand_as(logits_known)
                loss = criterion(logits_known, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if cfg.verbose and (epoch == 1 or epoch % max(1, cfg.num_epochs // 5) == 0):
                print(f"    epoch {epoch}/{cfg.num_epochs} - loss: {total_loss / max(n_batches, 1):.4f}")

        return model

    @torch.no_grad()
    def _predict_edge_scores(self, model: GAEModel) -> np.ndarray:
        """Per-cell activation probabilities, aggregated (median/mean) across cells."""
        cfg = self.config
        n_edges = self.edge_index.n_edges
        model.eval()
        loader = DataLoader(self.dataset, batch_size=cfg.batch_size, shuffle=False)

        all_probs = []
        for batch in loader:
            batch = batch.to(cfg.device)
            probs = model(batch.x, batch.edge_index, batch.decode_edge_index, as_probs=True)
            all_probs.append(probs.view(batch.num_graphs, n_edges).cpu())

        probs = torch.cat(all_probs, dim=0).numpy()  # [n_cells, n_edges]
        return np.median(probs, axis=0) if cfg.cell_aggregation == "median" else probs.mean(axis=0)

    def fit(self) -> pd.DataFrame:
        """
        Run Algorithm 1: `n_runs` independent training runs, each producing a
        per-edge activation score aggregated over cells; the final result
        aggregates those runs (mean +/- std, and the paper's median).

        Returns
        -------
        DataFrame with one row per candidate edge (see `to_dataframe`).
        """
        cfg = self.config
        per_run_scores = np.zeros((cfg.n_runs, self.edge_index.n_edges), dtype=np.float64)

        for run in range(cfg.n_runs):
            if cfg.verbose:
                print(f"=== Run {run + 1}/{cfg.n_runs} ===")
            model = self._train_one_run()
            per_run_scores[run] = self._predict_edge_scores(model)

        df = self._to_dataframe(per_run_scores)
        self._fit_result = FitResult(
            edge_index=self.edge_index,
            known_mask=self.known_mask,
            corr=self.corr,
            per_run_scores=per_run_scores,
            dataframe=df,
        )
        return df

    # -- reporting ---------------------------------------------------------

    def _to_dataframe(self, per_run_scores: np.ndarray) -> pd.DataFrame:
        tfs = [name.split("->")[0] for name in self.edge_index.names]
        targets = [name.split("->")[1] for name in self.edge_index.names]
        mean_score = per_run_scores.mean(axis=0)

        df = pd.DataFrame({
            "edge": self.edge_index.names,
            "tf": tfs,
            "target": targets,
            "activation_score_mean": mean_score,
            "activation_score_std": per_run_scores.std(axis=0),
            "activation_score_median_of_runs": np.median(per_run_scores, axis=0),
            # The value backing an edge's label: a correlation coefficient in
            # binary/ternary/adaptive mode, or the target's winning Boolean
            # rule's cross-validated F1 in "logic" mode (see logic_labels.py) —
            # named for the common case, kept generic on purpose so all label
            # modes share one results schema.
            "correlation": self.corr,
            "supervised": self.known_mask,
            "predicted_polarity": np.where(mean_score >= 0.5, "activation", "repression"),
        })
        df.loc[~df["supervised"], "predicted_polarity"] = "uncertain"
        return df

    def summary(self, results: Optional[pd.DataFrame] = None, top_n: int = 10) -> None:
        """Print a short, human-readable summary of a `fit()` result."""
        df = results if results is not None else (self._fit_result.dataframe if self._fit_result else None)
        if df is None:
            raise RuntimeError("Call fit() first, or pass its returned DataFrame explicitly.")

        n_supervised = int(df["supervised"].sum())
        n_activation = int((df["predicted_polarity"] == "activation").sum())
        n_repression = int((df["predicted_polarity"] == "repression").sum())

        print(f"\n{'=' * 72}")
        print("sc-GRIP SUMMARY")
        print(f"{'=' * 72}")
        print(f"Candidate edges scored : {len(df)}")
        print(f"Supervised by a proxy label : {n_supervised}")
        print(f"Predicted activation : {n_activation}")
        print(f"Predicted repression : {n_repression}")
        print(f"\nMost confident activation edges (top {top_n}):")
        print(
            df[df["supervised"]]
            .sort_values("activation_score_mean", ascending=False)
            .head(top_n)[["edge", "activation_score_mean", "activation_score_std"]]
            .to_string(index=False)
        )
        print(f"\nMost confident repression edges (top {top_n}):")
        print(
            df[df["supervised"]]
            .sort_values("activation_score_mean", ascending=True)
            .head(top_n)[["edge", "activation_score_mean", "activation_score_std"]]
            .to_string(index=False)
        )

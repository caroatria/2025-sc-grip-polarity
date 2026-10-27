"""
Edge existence (TF -> target link prediction), label-free
==========================================================
`scgrip.inference.ScGRIP` takes a candidate network as input and scores the
*polarity* of its edges. This module uses the same architecture (2-layer GCN
encoder + bilinear decoder, `scgrip.model.GAEModel`) to score whether an edge
*exists*, for every candidate regulator -> gene pair, still without any
ground-truth labels:

1. Proxy labels come from expression only: pairs with the highest |Spearman|
   correlation are positives (and form the message-passing graph), pairs with
   the lowest are negatives. ``label_scheme="per_tf"`` picks them per
   regulator, so a few highly correlated hub genes cannot take most of the
   training edges; ``"global"`` picks them over all pairs.
2. The model is trained to separate positives from sampled negatives, either
   on one graph per cell with a scalar expression feature per gene
   (``features="per_cell"``, as in the original sc-GRIP) or on one graph whose
   node features are each gene's expression profile across cells
   (``features="profile"``, much faster).
3. Scores are aggregated over cells (median) and runs (mean), and optionally
   CLR-normalised (`clr_normalize`) to remove "this gene scores high with
   everything" effects.

Benchmark results for these settings are in ``benchmarks/beeline/results``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .model import GAEModel


@dataclass
class ExistenceConfig:
    label_scheme: str = "per_tf"     # "per_tf" or "global"
    pos_frac: float = 0.05           # share of pairs labelled positive (per TF or overall)
    neg_frac: float = 0.5            # share of pairs labelled negative
    features: str = "per_cell"       # "per_cell" (original design) or "profile"
    clr: bool = False                # CLR-normalise the final scores
    n_runs: int = 3
    epochs: int = 30                 # per_cell: passes over all cells
    profile_steps: int = 300         # profile: full-batch training steps
    pca_dim: int = 50                # profile: PCA dimensions for node features (0 = none)
    batch_size: int = 16
    neg_ratio: int = 1               # negatives sampled per positive per step
    hidden_channels: int = 128
    out_channels: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    cell_aggregation: str = "median"  # "median" or "mean"
    device: str = "cpu"
    seed: int = 0
    verbose: bool = False

    def __post_init__(self):
        if self.label_scheme not in ("per_tf", "global"):
            raise ValueError("label_scheme must be 'per_tf' or 'global'")
        if self.features not in ("per_cell", "profile"):
            raise ValueError("features must be 'per_cell' or 'profile'")
        if self.cell_aggregation not in ("median", "mean"):
            raise ValueError("cell_aggregation must be 'median' or 'mean'")


# ---------------------------------------------------------------------------
# Candidates, correlation, proxy labels
# ---------------------------------------------------------------------------

def candidate_pairs(genes: Sequence[str], regulators: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """All regulator -> gene index pairs over `genes`, without self-loops."""
    g_idx = {g: i for i, g in enumerate(genes)}
    regs = [g_idx[r] for r in regulators if r in g_idx]
    src = np.repeat(regs, len(genes))
    dst = np.tile(np.arange(len(genes)), len(regs))
    keep = src != dst
    return src[keep], dst[keep]


def abs_correlation(X: np.ndarray, src: np.ndarray, dst: np.ndarray, method: str = "spearman") -> np.ndarray:
    """|correlation| for each (src, dst) pair; X is cells x genes."""
    Z = pd.DataFrame(X).rank().to_numpy() if method == "spearman" else np.asarray(X, float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    return np.abs((Z[:, src] * Z[:, dst]).mean(0))


def proxy_labels(score: np.ndarray, src: np.ndarray, scheme: str, pos_frac: float, neg_frac: float):
    """Boolean (positive, negative) masks from a pairwise association score."""
    if scheme == "global":
        return score >= np.quantile(score, 1 - pos_frac), score <= np.quantile(score, neg_frac)
    pos, neg = np.zeros(len(score), bool), np.zeros(len(score), bool)
    for r in np.unique(src):
        idx = np.where(src == r)[0]
        v = score[idx]
        pos[idx[v >= np.quantile(v, 1 - pos_frac)]] = True
        neg[idx[v <= np.quantile(v, neg_frac)]] = True
    return pos, neg & ~pos


def clr_normalize(score: np.ndarray, src: np.ndarray, dst: np.ndarray, n_genes: int) -> np.ndarray:
    """
    Context likelihood of relatedness (Faith et al. 2007): z-score each pair
    against its regulator's row and its target's column, clip negatives, and
    combine as sqrt(z_row^2 + z_col^2).
    """
    M = np.full((n_genes, n_genes), np.nan)
    M[src, dst] = score
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows/columns (genes that are never regulators)
        zr = (M - np.nanmean(M, 1, keepdims=True)) / (np.nanstd(M, 1, keepdims=True) + 1e-12)
        zc = (M - np.nanmean(M, 0, keepdims=True)) / (np.nanstd(M, 0, keepdims=True) + 1e-12)
    out = np.sqrt(np.clip(zr, 0, None) ** 2 + np.clip(zc, 0, None) ** 2)
    return np.nan_to_num(out[src, dst])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _pair_logits(model: GAEModel, z: torch.Tensor, s: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Bilinear decoder for a batch of embeddings z [B, N, d] -> logits [B, n_pairs].
    Same parameters as `model.decoder.bilinear`, written as a matmul (much faster
    backward pass on CPU than nn.Bilinear)."""
    W, b = model.decoder.bilinear.weight[0], model.decoder.bilinear.bias[0]
    return ((z[:, s, :] @ W) * z[:, d, :]).sum(-1) + b


def _conv_graph(src, dst, pos_mask, dev):
    ps, pd_ = src[pos_mask], dst[pos_mask]
    return torch.tensor(np.stack([np.r_[ps, pd_], np.r_[pd_, ps]]), dtype=torch.long, device=dev)


def _loss(lp, ln):
    return (F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp))
            + F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))) / 2


def _fit_per_cell(Xz, src, dst, pos_mask, neg_mask, cfg: ExistenceConfig, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = cfg.device
    X = torch.tensor(Xz, dtype=torch.float32, device=dev).unsqueeze(-1)  # [cells, genes, 1]
    conv = _conv_graph(src, dst, pos_mask, dev)
    pos_s = torch.tensor(src[pos_mask], device=dev)
    pos_d = torch.tensor(dst[pos_mask], device=dev)
    neg_idx = np.where(neg_mask)[0]
    n_neg = min(len(neg_idx), cfg.neg_ratio * int(pos_mask.sum()))

    model = GAEModel(1, cfg.hidden_channels, cfg.out_channels).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n_batches = 0.0, 0
        for cells in torch.randperm(X.shape[0], device=dev).split(cfg.batch_size):
            ni = rng.choice(neg_idx, n_neg, replace=False)
            z = model.encoder(X[cells], conv)
            loss = _loss(_pair_logits(model, z, pos_s, pos_d),
                         _pair_logits(model, z, torch.tensor(src[ni], device=dev), torch.tensor(dst[ni], device=dev)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n_batches = total + loss.item(), n_batches + 1
        if cfg.verbose and (epoch == 1 or epoch % max(1, cfg.epochs // 5) == 0):
            print(f"    epoch {epoch}/{cfg.epochs} loss {total / max(n_batches, 1):.4f}")

    model.eval()
    with torch.no_grad():
        Z = torch.cat([model.encoder(xb, conv) for xb in X.split(cfg.batch_size)], 0)  # [cells, genes, d]
        S, D = torch.tensor(src, device=dev), torch.tensor(dst, device=dev)
        out = np.empty(len(src), dtype=np.float32)
        step = max(1, int(2e8 // (Z.shape[0] * Z.shape[2] * 4)))  # keep blocks ~200 MB
        for i in range(0, len(src), step):
            lg = _pair_logits(model, Z, S[i:i + step], D[i:i + step])
            agg = lg.median(0).values if cfg.cell_aggregation == "median" else lg.mean(0)
            out[i:i + step] = torch.sigmoid(agg).cpu().numpy()
    return out


def _fit_profile(Xz, src, dst, pos_mask, neg_mask, cfg: ExistenceConfig, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = cfg.device
    feats = Xz.T  # genes x cells
    if cfg.pca_dim and min(feats.shape) > cfg.pca_dim:
        from sklearn.decomposition import PCA
        feats = PCA(cfg.pca_dim, random_state=seed).fit_transform(feats)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    X = torch.tensor(feats, dtype=torch.float32, device=dev)
    conv = _conv_graph(src, dst, pos_mask, dev)
    pos_s = torch.tensor(src[pos_mask], device=dev)
    pos_d = torch.tensor(dst[pos_mask], device=dev)
    neg_idx = np.where(neg_mask)[0]
    n_neg = min(len(neg_idx), cfg.neg_ratio * int(pos_mask.sum()))

    model = GAEModel(X.shape[1], cfg.hidden_channels, cfg.out_channels).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for step in range(1, cfg.profile_steps + 1):
        model.train()
        ni = rng.choice(neg_idx, n_neg, replace=False)
        z = model.encoder(X, conv).unsqueeze(0)
        loss = _loss(_pair_logits(model, z, pos_s, pos_d),
                     _pair_logits(model, z, torch.tensor(src[ni], device=dev), torch.tensor(dst[ni], device=dev)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if cfg.verbose and (step == 1 or step % max(1, cfg.profile_steps // 5) == 0):
            print(f"    step {step}/{cfg.profile_steps} loss {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        z = model.encoder(X, conv).unsqueeze(0)
        S, D = torch.tensor(src, device=dev), torch.tensor(dst, device=dev)
        return torch.sigmoid(_pair_logits(model, z, S, D)[0]).cpu().numpy()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_candidate_edges(
    expr_df: pd.DataFrame,
    regulators: Sequence[str],
    config: Optional[ExistenceConfig] = None,
    genes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Score every regulator -> gene pair for edge existence, label-free.

    Parameters
    ----------
    expr_df    : cells x genes expression matrix (log-normalised).
    regulators : candidate TFs (those not in `expr_df` are ignored).
    genes      : restrict to these genes (default: all columns of `expr_df`).

    Returns
    -------
    DataFrame with one row per pair: ``tf, target, score`` (final score,
    higher = more likely an edge), ``score_raw`` (before CLR),
    ``abs_spearman`` and ``train_positive`` (whether the pair was a
    proxy-label positive).
    """
    cfg = config or ExistenceConfig()
    genes = list(genes) if genes is not None else list(expr_df.columns)
    X = expr_df[genes].to_numpy(dtype=float)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-6)
    src, dst = candidate_pairs(genes, regulators)
    if len(src) == 0:
        raise ValueError("no regulator of `regulators` is among the expression genes")

    sp = abs_correlation(X, src, dst, "spearman")
    pos, neg = proxy_labels(sp, src, cfg.label_scheme, cfg.pos_frac, cfg.neg_frac)
    fit = _fit_profile if cfg.features == "profile" else _fit_per_cell
    raw = np.mean([fit(Xz, src, dst, pos, neg, cfg, cfg.seed + r) for r in range(cfg.n_runs)], axis=0)
    final = clr_normalize(raw, src, dst, len(genes)) if cfg.clr else raw

    return pd.DataFrame({
        "tf": [genes[i] for i in src],
        "target": [genes[j] for j in dst],
        "score": final,
        "score_raw": raw,
        "abs_spearman": sp,
        "train_positive": pos,
    })

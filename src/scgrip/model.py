"""
sc-GRIP model architecture
===========================
Graph-autoencoder for TF-target interaction polarity, as described in
sc-GRIP: a Graph Convolutional Approach to Infer Gene Interaction Polarity
from Single-cell Data (Atria et al., ICDM 2025), Section III-C.

    Encoder (2-layer GCN):
        H1 = ReLU(GCNConv(X,  A; W1))
        Z  = GCNConv(H1, A; W2)

    Decoder (bilinear):
        s_uv = z_u^T W z_v
        p_uv = sigmoid(s_uv)

`Z` is a per-cell latent embedding of every gene (TF or target); `p_uv` is
the model's belief that the edge (u -> v) is an *activating* interaction,
given that cell's expression pattern propagated through the fixed TF-target
graph structure.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(torch.nn.Module):
    """Two-layer Graph Convolutional encoder (paper Eq. 3-5).

    Parameters
    ----------
    in_channels : dimensionality of the per-gene input feature (1 if each
        gene contributes only its own expression level in that cell).
    hidden_channels : width of the first GCN layer's output.
    out_channels : dimensionality of the latent embedding Z.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)


class BilinearDecoder(torch.nn.Module):
    """Bilinear edge-polarity decoder (paper Eq. 6-7).

    Returns raw logits by default (pair with `BCEWithLogitsLoss`, which is
    numerically safer than applying sigmoid and then `BCELoss`). Call
    `.forward(..., as_probs=True)` to get sigmoid-activated probabilities.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.bilinear = torch.nn.Bilinear(latent_dim, latent_dim, 1)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, as_probs: bool = False) -> torch.Tensor:
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        logits = self.bilinear(src, dst).squeeze(-1)
        return torch.sigmoid(logits) if as_probs else logits


class GAEModel(torch.nn.Module):
    """Full sc-GRIP graph autoencoder: `GCNEncoder` + `BilinearDecoder`."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 64):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = BilinearDecoder(out_channels)

    def encode(self, x: torch.Tensor, conv_edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, conv_edge_index)

    def forward(
        self,
        x: torch.Tensor,
        conv_edge_index: torch.Tensor,
        decode_edge_index: torch.Tensor = None,
        as_probs: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        conv_edge_index : edges used for GCN message passing. May include
            reverse edges (TF<->target) to let information flow both ways,
            even though the decoder only ever scores the original directed
            TF->target edges — undirected propagation is a message-passing
            convenience, not a claim about regulatory directionality.
        decode_edge_index : the directed TF->target edges to score. Defaults
            to `conv_edge_index` when not given (i.e. no reverse edges used).
        """
        z = self.encode(x, conv_edge_index)
        if decode_edge_index is None:
            decode_edge_index = conv_edge_index
        return self.decoder(z, decode_edge_index, as_probs=as_probs)

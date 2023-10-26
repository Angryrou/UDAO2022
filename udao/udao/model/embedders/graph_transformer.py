from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import dgl
import torch as th
import torch.nn as nn

from .base_embedder import BaseEmbedder, EmbedderParams
from .layers.graph_transformer_layer import GraphTransformerLayer
from .layers.multi_head_attention import AttentionLayerName

ReadoutType = Literal["sum", "max", "mean"]


@dataclass
class GraphTransformerParams(EmbedderParams):
    pos_encoding_dim: int
    """Dimension of the position encoding."""
    n_layers: int
    """Number of GCN layers."""
    n_heads: int
    """Number of attention heads."""
    hidden_dim: int
    """Size of the hidden layers outputs."""
    readout: ReadoutType
    """Readout type: how the node embeddings are aggregated
    to form the graph embedding."""
    max_dist: Optional[int] = None
    """Maximum distance for QF attention."""
    non_siblings_map: Optional[Sequence[Sequence[int]]] = None
    """Non-siblings map for RAAL attention."""
    attention_layer_name: AttentionLayerName = "GTN"
    """Defines which attention layer to use (QF, RAAL, or GTN))"""
    dropout: float = 0.0
    """Dropout probability."""
    residual: bool = True
    """Whether to make the layer residual. Defaults to True."""
    use_bias: bool = False
    """Whether to use bias in the attention layer. Defaults to False."""
    batch_norm: bool = True
    """Whether to use batch normalization. Defaults to True."""
    layer_norm: bool = False
    """Whether to use layer normalization. Defaults to False."""


class GraphTransformer(BaseEmbedder):
    """Graph Transformer Network
    Computes graph embedding using attention mechanism
    (either QF, RAAL, or GTN)
    """

    def __init__(self, net_params: GraphTransformerParams) -> None:
        super().__init__(net_params=net_params)
        self.attention_layer_name = net_params.attention_layer_name
        self.embedding_lap_pos_enc = nn.Linear(
            net_params.pos_encoding_dim, net_params.hidden_dim
        )
        self.embedding_h = nn.Linear(self.input_size, net_params.hidden_dim)
        self.readout = net_params.readout
        self.attention_bias = None
        if self.attention_layer_name == "QF":
            if net_params.max_dist is None:
                raise ValueError("max_dist is required for QF")
            max_dist = net_params.max_dist
            self.attention_bias = nn.Parameter(th.zeros(max_dist))

        elif self.attention_layer_name == "RAAL":
            if net_params.non_siblings_map is None:
                raise ValueError("non_siblings_map is required for RAAL")
        elif self.attention_layer_name == "GTN":
            pass
        else:
            raise ValueError(self.attention_layer_name)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    in_dim=net_params.hidden_dim,
                    out_dim=out_dim,
                    n_heads=net_params.n_heads,
                    dropout=net_params.dropout,
                    layer_norm=net_params.layer_norm,
                    batch_norm=net_params.batch_norm,
                    residual=net_params.residual,
                    use_bias=net_params.use_bias,
                    attention_layer_name=net_params.attention_layer_name,
                    attention_bias=self.attention_bias,
                    non_siblings_map=net_params.non_siblings_map,
                )
                for out_dim in [
                    net_params.hidden_dim
                    if i < net_params.n_layers - 1
                    else net_params.embedding_size
                    for i in range(net_params.n_layers)
                ]
            ]
        )

    def _embed(
        self, g: dgl.DGLGraph, h: th.Tensor, h_lap_pos_enc: th.Tensor
    ) -> th.Tensor:
        h = self.embedding_h(h)
        h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc)
        h = h + h_lap_pos_enc

        # convnets
        for layer in self.layers:
            h = layer.forward(g, h)
        g.ndata["h"] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, "h")
        elif self.readout == "max":
            hg = dgl.max_nodes(g, "h")
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, "h")
        else:
            raise NotImplementedError
        return hg

    def forward(self, g: dgl.DGLGraph, h_lap_pos_enc: th.Tensor) -> th.Tensor:  # type: ignore[override]
        h = self.concatenate_op_features(g)
        return self.normalize_embedding(self._embed(g, h, h_lap_pos_enc))

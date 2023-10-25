# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: Graph Transformer Network implementation, adjusted based on
# https://github.com/graphdeeplearning/graphtransformer/blob/main/nets/molecules_graph_regression/graph_transformer_net.py
#
# Created at 03/01/2023

from dataclasses import dataclass
from typing import Optional, Sequence

import dgl
import torch as th
import torch.nn as nn
from udao.model.embedders.base_embedder import BaseEmbedder, EmbedderParams
from udao.model.embedders.layers.graph_transformer import GraphTransformerLayer


@dataclass
class TransformerParams(EmbedderParams):
    ped: int
    n_gcn_layers: int
    n_heads: int
    hidden_dim: int
    dropout: float
    residual: bool
    readout: str
    batch_norm: bool
    layer_norm: bool
    out_norm: str
    max_dist: Optional[int]
    non_siblings_map: Sequence[Sequence[int]]


class GraphTransformer(BaseEmbedder):
    def __init__(self, net_params: TransformerParams) -> None:
        super().__init__(net_params=net_params)

        ped = net_params.ped
        n_gcn_layers = net_params.n_gcn_layers
        n_heads = net_params.n_heads
        hidden_dim = net_params.hidden_dim
        dropout = net_params.dropout

        self.residual = net_params.residual
        self.readout = net_params.readout
        self.batch_norm = net_params.batch_norm
        self.layer_norm = net_params.layer_norm
        self.embedding_lap_pos_enc = nn.Linear(ped, hidden_dim)
        self.embedding_h = nn.Linear(self.in_feat_size_op, hidden_dim)

        if self.name == "QF":
            if net_params.max_dist is None:
                raise ValueError("max_dist is required for QF")
            max_dist = net_params.max_dist
            self.attention_bias = nn.Parameter(
                th.zeros(max_dist)
            )  # fixme: should be renamed to `attention_bias`
            self.layers = nn.ModuleList(
                [
                    GraphTransformerLayer(
                        self.name,
                        hidden_dim,
                        out_dim,
                        n_heads,
                        dropout,
                        self.layer_norm,
                        self.batch_norm,
                        self.residual,
                        attention_bias=self.attention_bias,
                    )
                    for out_dim in [
                        hidden_dim if i < n_gcn_layers - 1 else self.embedding_size
                        for i in range(n_gcn_layers)
                    ]
                ]
            )

        else:
            if self.name == "RAAL":
                self.add_misc = net_params.non_siblings_map
            else:
                raise ValueError(self.name)

            self.layers = nn.ModuleList(
                [
                    GraphTransformerLayer(
                        self.name,
                        hidden_dim,
                        out_dim,
                        n_heads,
                        dropout,
                        self.layer_norm,
                        self.batch_norm,
                        self.residual,
                        non_siblings_map=net_params.non_siblings_map,
                    )
                    for out_dim in [
                        hidden_dim if i < n_gcn_layers - 1 else self.embedding_size
                        for i in range(n_gcn_layers)
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
        for conv in self.layers:
            h = conv.forward(g, h)
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
        h = self.concatenate_op(g)
        return self.normalize_embedding(self._embed(g, h, h_lap_pos_enc))

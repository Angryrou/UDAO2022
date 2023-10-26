from typing import Optional, Sequence

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .multi_head_attention import ATTENTION_TYPES, AttentionLayerName


class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        layer_norm: bool = False,
        batch_norm: bool = True,
        residual: bool = True,
        use_bias: bool = False,
        attention_layer_name: AttentionLayerName = "GTN",
        non_siblings_map: Optional[Sequence[Sequence[int]]] = None,
        attention_bias: Optional[th.Tensor] = None,
    ) -> None:
        super().__init__()

        self.attention_layer_name = attention_layer_name
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.attention_bias = attention_bias
        self.non_siblings_map = non_siblings_map
        attention_config = ATTENTION_TYPES.get(self.attention_layer_name)
        if not attention_config:
            raise ValueError(f"Unknown attention type: {self.name}")

        required_attrs = attention_config["requires"]
        for attr in required_attrs:
            if self.__getattr__(attr) is None:
                raise ValueError(f"{self.name} requires {attr}")

        self.attention = attention_config["layer"](
            in_dim,
            out_dim // n_heads,
            n_heads,
            use_bias,
            **{attr: self.__getattr__(attr) for attr in required_attrs},
        )

        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g: dgl.DGLGraph, h: th.Tensor) -> th.Tensor:
        h_in1 = h  # for first residual connection
        # multi-head attention out
        attn_out = self.attention.forward(g, h)
        h = attn_out.view(-1, self.out_channels)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O(h)
        if self.residual:
            h = h_in1 + h  # residual connection
        if self.layer_norm:
            h = self.layer_norm1(h)
        if self.batch_norm:
            h = self.batch_norm1(h)
        h_in2 = h  # for second residual connection
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)
        if self.residual:
            h = h_in2 + h  # residual connection
        if self.layer_norm:
            h = self.layer_norm2(h)
        if self.batch_norm:
            h = self.batch_norm2(h)
        return h

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, heads={}, residual={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.n_heads,
            self.residual,
        )

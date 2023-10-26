from typing import Any, Callable, Dict, List, Literal, Sequence, Tuple

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn


def src_dot_dst(
    src_field: str, dst_field: str, out_field: str
) -> Callable[[Any], Dict[str, torch.Tensor]]:
    def func(edges: Any) -> Dict[str, torch.Tensor]:
        return {
            out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(
                -1, keepdim=True
            )
        }

    return func


def scaled_exp(
    field: str, scale_constant: float
) -> Callable[[Any], Dict[str, torch.Tensor]]:
    def func(edges: Any) -> Dict[str, torch.Tensor]:
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


def add_att_weights_bias(field: str) -> Callable[[Any], Dict[str, torch.Tensor]]:
    def func(edges: Any) -> Dict[str, torch.Tensor]:
        return {field: edges.data[field] + edges.data["att_weights"]}

    return func


class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Attention Layer for Graph

    Parameters
    ----------
        in_dim : int
            Input dimension
        out_dim : int
            Output dimension
        n_heads : int
            Number of attention heads
        use_bias : bool
            Whether to use bias
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int, use_bias: bool) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads

        self.Q = nn.Linear(in_dim, out_dim * n_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * n_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * n_heads, bias=use_bias)

    def _apply_attention(self, g: dgl.DGLGraph) -> torch.Tensor:
        edge_ids: Tuple[torch.Tensor, torch.Tensor] = g.edges()
        g.send_and_recv(
            edge_ids,
            fn.u_mul_e("V_h", "score", "V_h"),  # type: ignore
            fn.sum("V_h", "wV"),  # type: ignore
        )
        g.send_and_recv(
            edge_ids, fn.copy_e("score", "score"), fn.sum("score", "z")  # type: ignore
        )
        head_out = g.ndata["wV"] / (
            g.ndata["z"] + torch.full_like(g.ndata["z"], 1e-6)  # type: ignore
        )
        return head_out

    def compute_attention(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        g.apply_edges(src_dot_dst("K_h", "Q_h", "score"))  # , edges)
        g.apply_edges(scaled_exp("score", np.sqrt(self.out_dim)))
        return g

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata["Q_h"] = self.Q(h).view(-1, self.n_heads, self.out_dim)
        g.ndata["K_h"] = self.K(h).view(-1, self.n_heads, self.out_dim)
        g.ndata["V_h"] = self.V(h).view(-1, self.n_heads, self.out_dim)

        g = self.compute_attention(g)

        return self._apply_attention(g)


class RAALMultiHeadAttentionLayer(MultiHeadAttentionLayer):
    """MultiHead Attention using Resource-Aware Attentional LSTM
    proposed by "A Resource-Aware Deep Cost Model for Big Data Query Processing”
    https://ieeexplore.ieee.org/document/9835426


    Parameters
    ----------
    non_siblings_map : Sequence[Sequence[int]]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        use_bias: bool,
        non_siblings_map: Sequence[Sequence[int]],
    ) -> None:
        super().__init__(in_dim, out_dim, num_heads, use_bias)
        self.non_siblings_map = non_siblings_map

    def compute_attention(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        g_list = dgl.unbatch(g)
        gg_map: Dict[int, List[dgl.DGLGraph]] = {}
        for gg in g_list:
            sid = gg.ndata["sid"][0].cpu().item()
            if sid in gg_map:
                gg_map[sid].append(gg)
            else:
                gg_map[sid] = [gg]

        gb_list = []
        for sid, gg_list in gg_map.items():
            n_gg = len(gg_list)
            gb = dgl.batch(gg_list)
            Q = gb.ndata["Q_h"].reshape(n_gg, -1, self.n_heads, self.out_dim)  # type: ignore
            K = gb.ndata["K_h"].reshape(n_gg, -1, self.n_heads, self.out_dim)  # type: ignore
            QK = (
                torch.matmul(Q.transpose(1, 2), K.transpose(1, 2).transpose(2, 3))
                .transpose(1, 2)
                .clamp(-5, 5)
            )
            srcs, dsts, eids = gg_list[0].edges(form="all", order="srcdst")
            score_list = [
                QK[:, src, :, dst]
                / (
                    QK[:, src, :, self.non_siblings_map[sid][eid]].sum(-1)
                    + torch.full_like(QK[:, src, :, dst], 1e-6)
                )
                for src, dst, eid in zip(
                    srcs.cpu().numpy(), dsts.cpu().numpy(), eids.cpu().numpy()
                )
            ]
            gb.edata["score"] = torch.cat(score_list, dim=1).view(-1, self.n_heads, 1)
            gb_list.append(gb)
        return dgl.batch(gb_list)


class QFMultiHeadAttentionLayer(MultiHeadAttentionLayer):
    """MultiHead Attention using QueryFormer
    proposed by "QueryFormer: A Tree Transformer Model for Query Plan
    Representation"
    https://www.vldb.org/pvldb/vol15/p1658-zhao.pdf

    Parameters
    ----------
    attention_bias : torch.Tensor
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        use_bias: bool,
        attention_bias: torch.Tensor,
    ) -> None:
        super().__init__(in_dim, out_dim, num_heads, use_bias)
        self.attention_bias = attention_bias

    def compute_attention(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        g = super().compute_attention(g)
        g.edata["att_bias"] = torch.index_select(
            self.attention_bias, 0, g.edata["dist"] - 1  # type: ignore
        ).reshape(-1, 1, 1)
        g.edata["score"] = g.edata["score"] + g.edata["att_bias"]  # type: ignore
        return g


AttentionLayerName = Literal["QF", "GTN", "RAAL"]

ATTENTION_TYPES: Dict[AttentionLayerName, Dict] = {
    "QF": {"layer": QFMultiHeadAttentionLayer, "requires": ["attention_bias"]},
    "GTN": {"layer": MultiHeadAttentionLayer, "requires": []},
    "RAAL": {
        "layer": RAALMultiHeadAttentionLayer,
        "requires": ["non_siblings_map"],
    },
}

# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: Graph Transformer Layer implementation,
# credits to https://github.com/graphdeeplearning/graphtransformer/blob/main/layers/graph_transformer_layer.py
#
# Created at 03/01/2023


import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    Util functions
"""

def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func

def add_att_weights_bias(field):

    def func(edges):
        return {field: edges.data[field] + edges.data["att_weights"]}

    return func

"""
    Single Attention Head
"""


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, name, in_dim, out_dim, num_heads, use_bias, add_misc):
        super().__init__()

        self.name = name
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.add_misc = add_misc

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)


    def forward(self, g, h):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        if self.name == "GTN":
            g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)
            g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

            # Send weighted values to target nodes
            eids = g.edges()
            g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
            g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
            head_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))

        elif self.name == "RAAL":
            g_list = dgl.unbatch(g)
            gg_map = {}
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
                Q = gb.ndata["Q_h"].reshape(n_gg, -1, self.num_heads, self.out_dim)
                K = gb.ndata["K_h"].reshape(n_gg, -1, self.num_heads, self.out_dim)
                QK = torch.matmul(Q.transpose(1, 2), K.transpose(1, 2).transpose(2, 3)).transpose(1, 2).clamp(-5, 5)
                srcs, dsts, eids = gg_list[0].edges(form='all', order='srcdst')
                score_list = [
                    QK[:, src, :, dst] / (QK[:, src, :, self.add_misc[sid][eid]].sum(-1) +
                                          torch.full_like(QK[:, src, :, dst], 1e-6))
                    for src, dst, eid in zip(srcs.cpu().numpy(), dsts.cpu().numpy(), eids.cpu().numpy())
                ]
                gb.edata["score"] = torch.cat(score_list, dim=1).view(-1, self.num_heads, 1)
                gb_list.append(gb)
            g = dgl.batch(gb_list)

            # for gg in g_list:
            #     Q = gg.ndata["Q_h"] # (n_nodes, n_heads, n_dim)
            #     K = gg.ndata["K_h"] # (n_nodes, n_heads, n_dim)
            #     sid = gg.ndata["sid"][0].cpu().item()
            #     # (n_nodes, n_heads, n_nodes)
            #     QK = torch.matmul(Q.transpose(0, 1), K.transpose(0, 1).transpose(1, 2)).transpose(0, 1).clamp(-5, 5)
            #     srcs, dsts, eids = gg.edges(form='all', order='srcdst')
            #     gg.edata["score"] = torch.cat([
            #         QK[src:src+1, :, dst] / QK[src:src+1, :, self.add_misc[sid][eid]].sum(-1)
            #         for src, dst, eid in zip(srcs.cpu().numpy(), dsts.cpu().numpy(), eids.cpu().numpy())
            #     ]).unsqueeze(2)
            # g = dgl.batch(g_list)

            # Send weighted values to target nodes
            eids = g.edges()
            g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
            g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
            head_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))

        elif self.name == "QF":
            g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)
            g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))
            attention_bias = self.add_misc
            g.edata["att_bias"] = torch.index_select(attention_bias, 0, g.edata["dist"] - 1).reshape(-1, 1, 1)
            g.edata["score"] = g.edata["score"] + g.edata["att_bias"]

            # Send weighted values to target nodes
            eids = g.edges()
            g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
            g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
            head_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        else:
            ValueError(self.name)

        return head_out


class GraphTransformerLayer(nn.Module):
    """
        Param:
    """

    def __init__(self, name, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False, add_misc=None):
        super().__init__()

        self.name = name
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(name, in_dim, out_dim // num_heads,
                                                 num_heads, use_bias, add_misc)

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

    def forward(self, g, h):
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

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)
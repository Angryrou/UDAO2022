# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 28/02/2023

import os

from model.architecture.mlp_readout_layer import MLPReadout
os.environ['DGLBACKEND'] = 'pytorch'
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GATv2Conv


class GATv2(nn.Module):

    def __init__(self, net_params):
        super().__init__()
        name = net_params["name"]
        in_feat_size_op = net_params["in_feat_size_op"]
        in_feat_size_inst = net_params["in_feat_size_inst"]
        out_feat_size = net_params["out_feat_size"]
        n_gcn_layers = net_params["L_gtn"]
        n_mlp_layers = net_params["L_mlp"]
        num_heads = net_params["n_heads"]
        hidden_dim = net_params["hidden_dim"]
        out_dim = net_params["out_dim"]
        dropout = net_params["dropout"]
        dropout2 = 0. if "dropout2" not in net_params else net_params["dropout2"]
        op_groups = net_params["op_groups"]

        self.name = name
        self.op_type = ("ch1_type" in op_groups)
        self.op_cbo = ("ch1_cbo" in op_groups)
        self.op_enc = ("ch1_enc" in op_groups)
        if self.op_type:
            self.op_embedder = nn.Embedding(net_params["n_op_types"], net_params["ch1_type_dim"])
        self.residual = net_params["residual"]
        self.readout = net_params["readout"]
        self.embedding_h = nn.Linear(in_feat_size_op, hidden_dim)
        self.layer_norm = net_params["layer_norm"]

        if self.layer_norm:
            self.ln = nn.ModuleList([nn.LayerNorm(hidden_dim // num_heads) for _ in range(n_gcn_layers - 1)])
        else:
            self.ln = None

        self.layers = nn.ModuleList(
            [GATv2Conv(hidden_dim, hidden_dim // num_heads, num_heads, feat_drop=dropout, attn_drop=dropout,
                       residual=self.residual)
             for _ in range(n_gcn_layers - 1)])
        self.layers.append(
            GATv2Conv(hidden_dim, out_dim // num_heads, num_heads, feat_drop=dropout, attn_drop=dropout,
                      residual=self.residual))
        if "agg_dim" not in net_params or net_params["agg_dim"] is None:
            agg_dim = None
        else:
            agg_dim = net_params["agg_dim"]
            assert agg_dim != "None"
        self.MLP_layer = MLPReadout(
            input_dim=out_dim + in_feat_size_inst, hidden_dim=net_params["mlp_dim"], output_dim=out_feat_size,
            L=n_mlp_layers, dropout=dropout2, agg_dim=agg_dim)

    def forward(self, g, inst_feat):
        op_list = []
        if self.op_type:
            op_list.append(self.op_embedder(g.ndata["op_gid"]))
        if self.op_cbo:
            op_list.append(g.ndata["cbo"])
        if self.op_enc:
            op_list.append(g.ndata["enc"])
        h = th.cat(op_list, dim=1) if len(op_list) > 1 else op_list[0]
        if self.embedding_h is not None:
            h = self.embedding_h(h)
            h = F.leaky_relu(h)

        for i, conv in enumerate(self.layers):
            h = conv.forward(g, h)
            if self.ln is not None and i < len(self.ln):
                h = self.ln[i](h)
            h = h.flatten(1)
            h = F.leaky_relu(h)

        g.ndata["h"] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, "h")
        elif self.readout == "max":
            hg = dgl.max_nodes(g, "h")
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, "h")
        else:
            raise NotImplementedError

        if inst_feat is None:
            return hg
        else:
            return self.mlp_forward(hg, inst_feat)

    def mlp_forward(self, hg, inst_feat):
        hgi = th.cat([hg, inst_feat], dim=1)
        return th.exp(self.MLP_layer(hgi))
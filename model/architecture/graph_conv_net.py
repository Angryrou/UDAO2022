# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 27/02/2023

import os

from model.architecture.mlp_readout_layer import MLPReadout
os.environ['DGLBACKEND'] = 'pytorch'
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GraphConv


class GCN(nn.Module):

    def __init__(self, net_params):
        super(GCN, self).__init__()
        name = net_params["name"]
        in_feat_size_op = net_params["in_feat_size_op"]
        in_feat_size_inst = net_params["in_feat_size_inst"]
        out_feat_size = net_params["out_feat_size"]
        n_gcn_layers = net_params["L_gtn"]
        n_mlp_layers = net_params["L_mlp"]
        dropout2 = 0. if "dropout2" not in net_params else net_params["dropout2"]
        op_groups = net_params["op_groups"]
        hidden_dim = net_params["hidden_dim"]
        out_dim = net_params["out_dim"]

        self.name = name
        self.op_type = ("ch1_type" in op_groups)
        self.op_cbo = ("ch1_cbo" in op_groups)
        self.op_enc = ("ch1_enc" in op_groups)
        if self.op_type:
            self.op_embedder = nn.Embedding(net_params["n_op_types"], net_params["ch1_type_dim"])
        self.readout = net_params["readout"]

        layers = []
        if in_feat_size_op < hidden_dim:
            self.embedding_h = nn.Linear(in_feat_size_op, hidden_dim)
        else:
            self.embedding_h = None
            layers.append(GraphConv(in_feat_size_op, hidden_dim))

        for i in range(n_gcn_layers - 1):
            layers.append(GraphConv(hidden_dim, hidden_dim // 2))
            hidden_dim = hidden_dim // 2

        assert hidden_dim > out_dim
        layers.append(GraphConv(hidden_dim, out_dim))
        self.convs = nn.ModuleList(layers)

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
            h = F.relu(h)

        for conv in self.convs:
            h = conv(g, h)
            h = F.relu(h)

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

# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: Graph Transformer Network implementation, adjusted based on
# https://github.com/graphdeeplearning/graphtransformer/blob/main/nets/molecules_graph_regression/graph_transformer_net.py
#
# Created at 03/01/2023

import torch as th
import torch.nn as nn
import dgl

from .graph_transformer_layer import GraphTransformerLayer
from .mlp_readout_layer import MLPReadout


class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        name = net_params["name"]
        ped = net_params["ped"]
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
        self.batch_norm = net_params["batch_norm"]
        self.layer_norm = net_params["layer_norm"]
        self.embedding_lap_pos_enc = nn.Linear(ped, hidden_dim)
        self.embedding_h = nn.Linear(in_feat_size_op, hidden_dim)

        if name == "QF":
            max_dist = net_params["max_dist"]
            self.attention_weights = nn.Parameter(th.zeros(max_dist)) # fixme: should be renamed to `attention_bias`
            self.layers = nn.ModuleList(
                [GraphTransformerLayer(name, hidden_dim, hidden_dim, num_heads, dropout,
                                       self.layer_norm, self.batch_norm, self.residual,
                                       add_misc=self.attention_weights)
                 for _ in range(n_gcn_layers - 1)])
            self.layers.append(
                GraphTransformerLayer(name, hidden_dim, out_dim, num_heads, dropout,
                                      self.layer_norm, self.batch_norm, self.residual,
                                      add_misc=self.attention_weights))
        else:
            if name == "GTN":
                self.add_misc = None
            elif name == "RAAL":
                self.add_misc = net_params["non_siblings_map"]
            else:
                raise ValueError(name)

            self.layers = nn.ModuleList(
                [GraphTransformerLayer(name, hidden_dim, hidden_dim, num_heads, dropout,
                                       self.layer_norm, self.batch_norm, self.residual,
                                       add_misc=self.add_misc)
                 for _ in range(n_gcn_layers - 1)])
            self.layers.append(
                GraphTransformerLayer(name, hidden_dim, out_dim, num_heads, dropout,
                                      self.layer_norm, self.batch_norm, self.residual,
                                      add_misc=self.add_misc))

        self.MLP_layer = MLPReadout(
            input_dim=out_dim + in_feat_size_inst, hidden_dim=net_params["mlp_dim"], output_dim=out_feat_size,
            L=n_mlp_layers, dropout=dropout2)

    def forward(self, g, h_lap_pos_enc, inst_feat=None):
        """
        g: stage graphs
        h_lap_pos_enc: flipped g.ndata["lap_pos_enc"]
        inst_feat: tabular feats
        """
        # input embedding
        op_list = []
        if self.op_type:
            op_list.append(self.op_embedder(g.ndata["op_gid"]))
        if self.op_cbo:
            op_list.append(g.ndata["cbo"])
        if self.op_enc:
            op_list.append(g.ndata["enc"])
        h = th.cat(op_list, dim=1) if len(op_list) > 1 else op_list[0]
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

        if inst_feat is None:
            return hg
        else:
            return self.mlp_forward(hg, inst_feat)

    def mlp_forward(self, hg, inst_feat):
        hgi = th.cat([hg, inst_feat], dim=1)
        return th.exp(self.MLP_layer(hgi))
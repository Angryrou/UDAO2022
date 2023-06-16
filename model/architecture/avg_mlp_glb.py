# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: just a temporary impl for Qi
#
# Created at 16/06/2023

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl

from model.architecture.IsoBN import IsoBN
from model.architecture.mlp_readout_layer import MLPReadout


class AVGMLP_GLB(nn.Module):

    def __init__(self, net_params):  # L=nb_hidden_layers
        super(AVGMLP_GLB, self).__init__()
        self.name = "AVGMLP_GLB"

        # in_feat_size_op -> out_dim
        in_feat_size_op = net_params["in_feat_size_op"]
        # theta_s_dim + out_dim -> hidden_dim
        theta_s_dim = net_params["theta_s_dim"]
        hidden_dim = net_params["hidden_dim"]
        # hidden_dim + in_feat_size_inst -> out_feat_size
        in_feat_size_inst = net_params["in_feat_size_inst"]
        out_feat_size = net_params["out_feat_size"]
        n_mlp_layers = net_params["L_mlp"]
        out_dim = net_params["out_dim"]
        mlp_dim = net_params["mlp_dim"]
        dropout2 = net_params["dropout2"]

        op_groups = net_params["op_groups"]
        self.op_type = ("ch1_type" in op_groups)
        self.op_cbo = ("ch1_cbo" in op_groups)
        self.op_enc = ("ch1_enc" in op_groups)

        assert self.op_type
        self.op_embedder = nn.Embedding(net_params["n_op_types"], net_params["ch1_type_dim"])
        self.emb = nn.Sequential(
            nn.Linear(in_feat_size_op, net_params["out_dim"]),
            nn.ReLU()) \
            if self.op_cbo or self.op_enc else None

        self.inner_xfer = nn.Sequential(
            nn.Linear(out_dim + theta_s_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        if "agg_dim" not in net_params or net_params["agg_dim"] is None:
            agg_dim = None
        else:
            agg_dim = net_params["agg_dim"]
        assert agg_dim != "None"
        self.MLP_layers = MLPReadout(
            input_dim=hidden_dim + in_feat_size_inst, hidden_dim=mlp_dim, output_dim=out_feat_size,
            L=n_mlp_layers, dropout=dropout2, agg_dim=agg_dim)

        if "out_norm" not in net_params or net_params["out_norm"] is None:
            self.out_norm = None
        elif net_params["out_norm"] == "BN":
            self.out_norm = nn.BatchNorm1d(out_dim)
        elif net_params["out_norm"] == "LN":
            self.out_norm = nn.LayerNorm(out_dim)
        elif net_params["out_norm"] == "IsoBN":
            self.out_norm = IsoBN(out_dim)
        else:
            raise ValueError(net_params["out_norm"])

    def forward(self, g_stage, g_op, theta_s, inst_feat):
        op_list = []
        if self.op_type:
            op_list.append(self.op_embedder(g_op.ndata["op_gid"]))
        h = th.cat(op_list, dim=1) if len(op_list) > 1 else op_list[0]
        if self.emb is not None:
            h = self.emb(h)
        g_op.ndata["h"] = h
        hg = dgl.mean_nodes(g_op, "h")
        if self.out_norm is not None:
            hg = self.out_norm(hg)
        hgs = th.cat([hg, theta_s], dim=1)
        hgs = self.inner_xfer(hgs)
        g_stage.ndata["h"] = hgs
        hg_stage = dgl.mean_nodes(g_stage, "h")
        hgi = th.cat([hg_stage, inst_feat], dim=1)
        out = self.MLP_layers.forward(hgi)
        return th.exp(out)

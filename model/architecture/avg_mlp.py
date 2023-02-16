# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 16/02/2023
import torch as th
import torch.nn as nn
import dgl

from model.architecture.IsoBN import IsoBN
from model.architecture.mlp_readout_layer import MLPReadout

class AVGMLP(nn.Module):

    def __init__(self, net_params): # L=nb_hidden_layers
        super(AVGMLP, self).__init__()
        self.name = "AVGMLP"
        in_feat_size_op = net_params["in_feat_size_op"]
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
        if self.op_type:
            self.op_embedder = nn.Embedding(net_params["n_op_types"], net_params["ch1_type_dim"])

        self.emb = nn.Sequential(
            nn.Linear(in_feat_size_op, out_dim),
            nn.ReLU()
        )

        self.MLP_layers = MLPReadout(
            input_dim=out_dim + in_feat_size_inst, hidden_dim=mlp_dim, output_dim=out_feat_size,
            L=n_mlp_layers, dropout=dropout2)

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

    def forward(self, g, inst_feat):
        op_list = []
        if self.op_type:
            op_list.append(self.op_embedder(g.ndata["op_gid"]))
        if self.op_cbo:
            op_list.append(g.ndata["cbo"])
        if self.op_enc:
            op_list.append(g.ndata["enc"])
        h = th.cat(op_list, dim=1) if len(op_list) > 1 else op_list[0]
        h = self.emb(h)
        g.ndata["h"] = h
        hg = dgl.mean_nodes(g, "h")
        if self.out_norm is not None:
            hg = self.out_norm(hg)

        hgi = th.cat([hg, inst_feat], dim=1)
        return th.exp(self.MLP_layers.forward(hgi))
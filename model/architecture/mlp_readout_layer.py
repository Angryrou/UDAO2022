# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 03/01/2023

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""


class MLPReadout_old(nn.Module):

    def __init__(self, input_dim, output_dim, L=2, dropout=0.0):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.dropout_list = [nn.Dropout(p=dropout) for l in range(L)]
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
            y = self.dropout_list[l](y)
        y = self.FC_layers[self.L](y)
        return y


class MLPReadout(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, L=2, dropout=0.0, agg_dim=None):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim, hidden_dim)]
        for l in range(L - 1):
            list_FC_layers.append(nn.Linear(hidden_dim, hidden_dim))

        if agg_dim is None:
            list_FC_layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            try:
                a_list = [int(a) for a in agg_dim.split(",")]
            except:
                raise ValueError(agg_dim)
            old_d = hidden_dim
            for a in a_list:
                list_FC_layers.append(nn.Linear(old_d, a))
                old_d = a
            list_FC_layers.append(nn.Linear(old_d, output_dim))

        self.FC_layers = nn.ModuleList(list_FC_layers)

        if dropout == 0:
            list_BN_layers =[nn.BatchNorm1d(hidden_dim)]
            for l in range(L - 1):
                list_BN_layers.append(nn.BatchNorm1d(hidden_dim))
            if agg_dim is not None:
                for a in a_list:
                    list_BN_layers.append(nn.BatchNorm1d(a))
            self.BN_layers = nn.ModuleList(list_BN_layers)

        self.dropout = dropout
        self.L = L

    def forward(self, x):
        y = x
        for l in range(len(self.FC_layers) - 1):
            y = self.FC_layers[l](y)
            y = F.relu(y)
            if self.dropout == 0:
                y = self.BN_layers[l](y)
            else:
                y = F.dropout(y, self.dropout, training=self.training)
        y = self.FC_layers[-1](y)
        return y

class PureMLP(nn.Module):

    def __init__(self, net_params):  # L=nb_hidden_layers
        super().__init__()
        n_mlp_layers = net_params["L_mlp"]
        hidden_dim = net_params["hidden_dim"]
        mlp_dim = net_params["mlp_dim"]
        in_feat_size_inst = net_params["in_feat_size_inst"]
        out_feat_size = net_params["out_feat_size"]
        # self.MLP_layers = MLP_Dropout(in_feat_size_inst, h_dim_list, dropout, out_dim=out_feat_size)
        self.emb = nn.Sequential(
            nn.Linear(in_feat_size_inst, hidden_dim),
            nn.ReLU()
        )
        dropout2 = net_params["dropout2"]
        if "agg_dim" not in net_params or net_params["agg_dim"] is None:
            agg_dim = None
        else:
            agg_dim = net_params["agg_dim"]
            assert agg_dim != "None"
        self.MLP_layers = MLPReadout(hidden_dim, mlp_dim, out_feat_size, L=n_mlp_layers,
                                     dropout=dropout2, agg_dim=agg_dim)

    def forward(self, x):
        x = self.emb(x)
        return torch.exp(self.MLP_layers.forward(x))
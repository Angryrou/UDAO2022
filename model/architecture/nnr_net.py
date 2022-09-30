import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.model.model_utils as ut

class NNRNet(nn.Module):
    def __init__(self, input_dim, output_dim, cap_list):
        super(NNRNet, self).__init__()
        np.random.seed(ut.SEED)
        torch.manual_seed(ut.SEED)

        hidden_fc_list = []
        in_dim = input_dim
        for cap in cap_list:
            fc = nn.Linear(in_dim, cap)
            in_dim = cap
            hidden_fc_list.append(fc)

        self.out_layer = nn.Linear(in_dim, output_dim)
        self.hidden_fc_list = nn.ModuleList(hidden_fc_list)
        self.in_dim = input_dim
        self.out_dim = output_dim

    def forward(self, X):
        assert X.shape[1] == self.in_dim
        for fc in self.hidden_fc_list:
            X = F.relu(fc(X))
        l = torch.exp(self.out_layer(X))
        if l.shape[1] == 1:
            l = l.squeeze(dim=1)
        return l


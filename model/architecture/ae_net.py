import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.model.model_utils as ut

class AENet(nn.Module):
    def __init__(self, W_dim=12, C_dim=12, O_dim=358, cap_list=None, cap_from=512, cap_to=32, cap_facotr=0.25):
        super(AENet, self).__init__()
        np.random.seed(ut.SEED)
        torch.manual_seed(ut.SEED)

        z_dim = W_dim + C_dim

        if cap_list is None:
            cap_list = [O_dim]
            tmp_dim = cap_from
            while tmp_dim > cap_to:
                cap_list.append(tmp_dim)
                tmp_dim = int(cap_facotr * tmp_dim)
            cap_list.append(tmp_dim)
            cap_list = cap_list + [z_dim] + cap_list[::-1]

        n_layers = len(cap_list)
        assert cap_list[n_layers//2] == z_dim

        cap_in_list = cap_list[:-1]
        cap_out_list = cap_list[1:]

        encode_fc_list = []
        decode_fc_list = []
        for cap_idx, (cap_in, cap_out) in enumerate(zip(cap_in_list, cap_out_list)):
            fc = nn.Linear(cap_in, cap_out)
            if cap_idx < n_layers/2-1:
                encode_fc_list.append(fc)
            else:
                decode_fc_list.append(fc)

        self.encode_fc_list = nn.ModuleList(encode_fc_list)
        self.decode_fc_list = nn.ModuleList(decode_fc_list)
        self.W_dim = W_dim
        self.C_dim = C_dim
        self.O_dim = O_dim

    def forward(self, O):
        assert(O.shape[1] == self.O_dim)
        X = O
        for fc in self.encode_fc_list:
            X = F.relu(fc(X))
        z = X
        for fc in self.decode_fc_list[:-1]:
            X = F.relu(fc(X))
        O_pred = self.decode_fc_list[-1](X)
        C_pred = z[:, :self.C_dim]
        W_pred = z[:, self.C_dim:]
        assert C_pred.shape[1] == self.C_dim
        assert W_pred.shape[1] == self.W_dim
        assert O_pred.shape[1] == self.O_dim

        return O_pred, C_pred, W_pred

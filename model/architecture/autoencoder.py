import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
import numpy as np
import utils.model.model_utils as ut
import utils.model.utils as utils
from .ae_net import AENet

class AutoEncoder:
    def __init__(self, W_dim, cap_list=None, O_dim=358):
        self.dtype = ut.DTYPE
        self.device = ut.DEVICE
        self.model = AENet(W_dim=W_dim, C_dim=12, O_dim=O_dim, cap_list=cap_list, cap_from=512, cap_to=32,
                           cap_facotr=0.25).to(self.device)

    def fit(self, tr_np, val_np, lr=1e-3, bs=64, epochs=1, patience=10,
            verbose=False, weight_list=None, to_plot=True):
        np.random.seed(ut.SEED)
        torch.manual_seed(ut.SEED)

        if weight_list is None:
            weight_list = [1, 1, 1, 1]
        dtype_ = [torch.long, torch.long, None, None, None]
        A_tr, temp_tr, C_tr, O_tr, L_tr = [utils.get_tensor(var_np, dtype=var_dtype) for var_np, var_dtype in
                                           zip(tr_np, dtype_)]
        A_val, temp_val, C_val, O_val, L_val = [utils.get_tensor(var_np, dtype=var_dtype) for var_np, var_dtype in
                                                zip(val_np, dtype_)]

        dataset = TensorDataset(A_tr, temp_tr, C_tr, O_tr, L_tr)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        best_epoch = -1
        best_loss = float("inf")
        best_epoch_1 = -1
        best_epoch_2 = -1
        best_epoch_3 = -1
        best_epoch_4 = -1
        best_loss_1 = float("inf")
        best_loss_2 = float("inf")
        best_loss_3 = float("inf")
        best_loss_4 = float("inf")

        best_model_state = None

        loss_tr_list = []
        loss_val_list = []

        early_stop = False
        iter_idx = 0
        print("epoch, batch, loss_tr, loss_val, lv_1, lv_2, lv_3, lv_4")
        for e in range(epochs):
            if early_stop:
                break
            for batch_idx, (A_batch, temp_batch, C_batch, O_batch, L_batch) in enumerate(loader):
                loss, loss_1, loss_2, loss_3, loss_4 = self.objective(A_batch, temp_batch, C_batch, O_batch,
                                                                      weight_list)

                if iter_idx % 50 == 0:
                    with torch.no_grad():
                        loss_val, loss_1_val, loss_2_val, loss_3_val, loss_4_val = self.objective(A_val, temp_val,
                                                                                                  C_val, O_val,
                                                                                                  weight_list)
                        if to_plot:
                            # for plot
                            loss_tr_list.append([l.item() for l in [loss_1, loss_2, loss_3, loss_4]])
                            loss_val_list.append([l.item() for l in [loss_1_val, loss_2_val, loss_3_val, loss_4_val]])
                        if verbose:
                            print(f"{e}, {batch_idx}, {loss:.5f}, {loss_val:.5f}, {loss_1_val:.5f}, {loss_2_val:.5f}, {loss_3_val:.5f}, {loss_4_val:.5f}")
                        if loss_val < best_loss:
                            best_epoch = e
                            best_loss = loss_val
                            best_model_state = copy.deepcopy(self.model.state_dict())
                        if loss_1_val < best_loss_1:
                            best_epoch_1 = e
                            best_loss_1 = loss_1_val
                        if loss_2_val < best_loss_2:
                            best_epoch_2 = e
                            best_loss_2 = loss_2_val
                        if loss_3_val < best_loss_3:
                            best_epoch_3 = e
                            best_loss_3 = loss_3_val
                        if loss_4_val < best_loss_4:
                            best_epoch_4 = e
                            best_loss_4 = loss_4_val
                        if e > max(best_epoch, best_epoch_1, best_epoch_2, best_epoch_3, best_epoch_4) + patience:
                            print(f'early stopped at epoch {e}, batch {batch_idx}')
                            early_stop = True
                            break

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_idx += 1

        print(f'--> at epoch {best_epoch}, got best_val_loss = {best_loss}')
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return np.array(loss_tr_list), np.array(loss_val_list)

    def objective(self, A, temp, C, O, weight_list):
        O_pred, C_pred, W_pred = self.model.forward(O)
        loss_1 = utils.get_tensor(0) if weight_list[0] == 0 else utils.get_l2_loss(input=O_pred, target=O, mode="mean")
        loss_2 = utils.get_tensor(0) if weight_list[1] == 0 else utils.get_l2_loss(input=C_pred, target=C, mode="mean")
        loss_3 = utils.get_tensor(0) if weight_list[2] == 0 else utils.get_loss_self_dist(A, W_pred)
        if weight_list[3] == 0:
            loss_4 = utils.get_tensor(0)
        else:
            _, temp_unique_list, W_unique_list = self.get_tensor_unique_wl(A, temp, W_pred)
            if torch.numel(temp_unique_list) < 2:
                loss_4 = utils.get_tensor(0)
            else:
                loss_4 = utils.get_loss_tri(W_unique_list, temp_unique_list)
        loss = weight_list[0] * loss_1 + weight_list[1] * loss_2 + weight_list[2] * loss_3 + weight_list[3] * loss_4
        return loss, loss_1, loss_2, loss_3, loss_4

    def get_tensor_unique_wl(self, A, temp, W):
        unique_len = len(A.unique())
        # A_list = torch.ones(unique_len, dtype=torch.long, device=self.device)
        A_list = []
        temp_list = torch.ones(unique_len, dtype=torch.long, device=self.device)
        W_list = torch.rand((unique_len, W.shape[1]), dtype=torch.float, device=self.device)
        ind = 0
        for a, t in zip(A, temp):
            if a in A_list:
                continue
            else:
                A_list.append(a)
                temp_list[ind] = t
                a_ind = torch.where(A == a)[0]
                w_a = W[a_ind]
                W_list[ind] = w_a.mean(dim=0)
                ind += 1
        A_list = torch.tensor(A_list, dtype=torch.long, device=self.device)
        return A_list, temp_list, W_list

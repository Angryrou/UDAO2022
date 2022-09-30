import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
import numpy as np
from .nnr_net import NNRNet
import utils.model.utils as utils
import utils.model.model_utils as ut

class NeuralNetworkRegressor:
    def __init__(self, cap_list, W_dim=12, C_dim=12, l_dim=1):
        if len(cap_list) < 1:
            raise Exception('cap_list should has at least one layer')
        self.dtype = ut.DTYPE
        self.device = ut.DEVICE

        in_dim = W_dim + C_dim
        self.model = NNRNet(input_dim = in_dim, output_dim=l_dim, cap_list=cap_list).to(self.device)
        self.in_dim = in_dim
        self.out_dim = l_dim

    def fit(self, tr_np, val_np, lr=1e-3, bs=64, epochs=1, patience=10,
            verbose=False, to_plot=True):
        np.random.seed(ut.SEED)
        torch.manual_seed(ut.SEED)
        X_tr, y_tr, X_val, y_val = [utils.get_tensor(var_np) for var_np in
                                    [tr_np['X'], tr_np['y'], val_np['X'], val_np['y']]]
        dataset = TensorDataset(X_tr, y_tr)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        best_epoch = -1
        best_loss = float("inf")
        best_model_state = None

        loss_tr_list = []
        loss_val_list = []

        early_stop = False
        iter_idx = 0
        for e in range(epochs):
            if early_stop:
                break
            for batch_idx, (X_batch, y_batch) in enumerate(loader):
                loss = self.objective(X_batch, y_batch)

                if iter_idx % 50 == 0:
                    with torch.no_grad():
                        loss_val = self.objective(X_val, y_val)
                        if to_plot:
                            loss_tr_list.append(loss.item())
                            loss_val_list.append(loss_val.item())
                        if verbose:
                            print(f'epoch {e}, batch {batch_idx}, loss_tr: {loss:.5f}, loss_val: {loss_val:.5f}')
                    if loss_val < best_loss:
                        best_epoch = e
                        best_loss = loss_val
                        best_model_state = copy.deepcopy(self.model.state_dict())
                    elif e > best_epoch + patience:
                        print(f'early stopped at epoch {e}, batch {batch_idx}')
                        early_stop = True
                        break

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_idx += 1

        print(f'--> at epoch {best_epoch}, got best_val_loss = {best_loss}')
        self.model.load_state_dict(best_model_state)
        return np.array(loss_tr_list), np.array(loss_val_list)

    def objective(self, X, y):
        y_pred = self.model.forward(X)
        loss = torch.abs(y_pred - y) / y
        return torch.mean(loss)

    def predict(self, X):
        with torch.no_grad():
            y_pred = self.model.forward(X)
        return y_pred

    def get_mape(self, X, y, reduction="mean"):
        """get numpy or scalar"""
        with torch.no_grad():
            y_pred = self.model.forward(X)
            mape_list = torch.abs(y_pred - y) / y
            if reduction == "mean":
                mape = torch.mean(mape_list).item()
            elif reduction == "none":
                mape = mape_list.detach().cpu().numpy()
            elif reduction == "all":
                mape = [mape_list.detach().cpu().numpy(), torch.mean(mape_list).item()]
        return mape

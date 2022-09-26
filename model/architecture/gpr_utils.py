import torch
import numpy as np

class GPRPT(object):
    def __init__(self, gp_obj_list, length_scale=1.0, magnitude=1.0):

        assert np.isscalar(length_scale)
        assert np.isscalar(magnitude)
        assert length_scale > 0 and magnitude > 0

        self.gp_obj_list = gp_obj_list
        self.length_scale = length_scale
        self.magnitude = magnitude

        torch.set_num_threads(1)
        np.random.seed(0)
        self.device, self.dtype = torch.device('cpu'), torch.float32

    def gs_dist(self, x1, x2):
        # x1.shape = (m, 12)
        # x2.shape = (n, 12)
        # K(x1, x2).shape = (m, n)
        assert x1.shape[1] == x2.shape[1]
        comm_dim = x1.shape[1]
        dist = torch.norm(x1.reshape(-1, 1, comm_dim) - x2.reshape(1, -1, comm_dim), dim=2)
        return dist

    def fit(self, X_train, y_dict, ridge=1.0):
        if X_train.ndim != 2:
            raise Exception("X_train should have 2 dimensions! X_dim:{}"
                            .format(X_train.ndim))
        X_train = self._get_tensor(X_train)
        sample_size = X_train.shape[0]
        if np.isscalar(ridge):
            ridge = np.ones(sample_size) * ridge
        assert isinstance(ridge, np.ndarray)
        assert ridge.ndim == 1
        ridge = self._get_tensor(ridge)
        K = self.magnitude * torch.exp(-self.gs_dist(X_train, X_train) / self.length_scale) + torch.diag(ridge)
        K_inv = K.inverse()

        y_tensor_dict = {}
        for obj in self.gp_obj_list:
            y_tensor_dict[obj] = self._get_tensor(y_dict[obj])
        return [X_train, y_tensor_dict, K_inv]

    def get_kernel(self, x1, x2):
        return self.magnitude * torch.exp(-self.gs_dist(x1, x2) / self.length_scale)

    def objective_std(self, X_test, X_train, K_inv, y_scale):
        K_tete = self.get_kernel(X_test, X_test) # (1,1)
        K_tetr = self.get_kernel(X_test, X_train) # (1, N)
        K_trte = K_tetr.t() # (N,1)
        var = K_tete - torch.matmul(torch.matmul(K_tetr, K_inv), K_trte) # (1,1)
        var_diag = var.diag()
        try:
            std = torch.sqrt(var_diag)
        except:
            std = var_diag - var_diag
            print('!!! var < 0')
        return std * y_scale

    def objective(self, X_test, X_train, y_train, K_inv):
        # X_test is a tensor
        # y_train is a tensor
        if X_test.ndimension() == 1:
            X_test = X_test.reshape(1, -1)
        length_scale = self.length_scale
        K2 = self.magnitude * torch.exp(-self.gs_dist(X_train, X_test) / length_scale)
        K2_trans = K2.t()
        yhat = torch.matmul(K2_trans, torch.matmul(K_inv, y_train))
        return yhat

    def predict(self, X_test, obj, X_train, y_dict, K_inv):
        assert obj in self.gp_obj_list
        X_test = self._get_tensor(X_test)
        y_train = y_dict[obj]
        yhat = self.objective(X_test, X_train, y_train, K_inv)
        yhat_np = yhat.numpy()
        GPRPT.check_output(yhat_np)
        if yhat_np.ndim == 1:
            yhat_np = np.expand_dims(yhat_np, axis=1)
        return yhat_np[0, 0]

    def _get_tensor(self, x, dtype=None, device=None):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return torch.tensor(x, dtype=dtype, device=device)

    @staticmethod
    def check_output(X):
        finite_els = np.isfinite(X)
        if not np.all(finite_els):
            raise Exception("Input contains non-finite values: {}"
                            .format(X[~finite_els]))

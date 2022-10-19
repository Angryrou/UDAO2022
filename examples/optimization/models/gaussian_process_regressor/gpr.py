# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#            Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: An example of Gaussian Process Regressor (GPR) model (reuse code in ICDE paper)
#
# Created at 17/10/2022
from utils.optimization.configs_parser import ConfigsParser
from optimization.model.base_model import BaseModel

import numpy as np
import torch as th

class GPR(BaseModel):
    def __init__(self, objs: list, training_vars: np.ndarray):
        '''
        :param objs: list, name of objectives
        :param training_vars: ndarray(n_training_samples, n_vars), input to train GPR models
        '''
        super().__init__()
        self.initialize(objs, training_vars)

    def initialize(self, objs, training_vars):
        '''
        initialize parameters and fit GPR models for objectives separately
        :param objs: list, name of objectives
        :param training_vars: ndarray(n_training_samples, n_vars), input to train GPR models
        :return:
        '''
        self.objs = objs

        self.model_params = ConfigsParser().parse_details(option="model")
        self.n_training = training_vars.shape[0]
        self.magnitude = self.model_params["magnitude"]
        self.ridge = self.model_params["ridge"]
        self.length_scale = self.model_params["length_scale"]
        self.device, self.dtype = th.device('cpu'), th.float32

        self.models = self.fit(training_vars)

    def gs_dist(self, x1, x2):
        '''
        calculate distances between each training sample
        :param x1: Tensor(n_training_samples, n_vars), training input data
        :param x2: Tensor(n_training_samples or n_x, n_vars), training input data or test input data
        :return:
                dist: Tensor(n_training_samples, n_training_samples), each element shows distance between each training sample
        '''
        # e.g.
        # x1.shape = (m, 12)
        # x2.shape = (n, 12)
        # K(x1, x2).shape = (m, n)
        assert x1.shape[1] == x2.shape[1]
        comm_dim = x1.shape[1]
        dist = th.norm(x1.reshape(-1, 1, comm_dim) - x2.reshape(1, -1, comm_dim), dim=2)
        return dist

    def get_y_dict(self, vars):
        '''
        get training data of true objective values with example functions
        :param vars: ndarray(n_training_samples, n_vars), input to train GPR models
        :return: y_dict: dict, key is objective names, values are training data of true objective values
        '''
        y_dict = {}
        # objective functions are the same as that in HCF
        for obj in self.objs:
            if obj == "obj_1":
                y = 4 * vars[:, 0] * vars[:, 0] + 4 * vars[:, 1] * vars[:, 1]
            elif obj == "obj_2":
                y = (vars[:, 0] - 5) * (vars[:, 0] - 5) + (vars[:, 1] - 5) * (vars[:, 1] - 5)
            else:
                raise Exception(f"Objective {obj} is not configured in the configuration file!")
            y_dict[obj] = y
        return y_dict

    def fit(self, X_train):
        '''
        fit GPR models
        :param X_train: ndarray(n_training_samples, n_vars), input to train GPR models
        :return:
                X_train: Tensor(n_training_samples, n_vars), input to train GPR models
                y_tensor_dict: dict, key is objective names, values are training data of true objective values with Tensor(n_training_samples,) format
                K_inv: Tensor(n_training_samples, n_training_samples), inversion of the covariance matrix
        '''
        y_dict = self.get_y_dict(X_train)

        if X_train.ndim != 2:
            raise Exception("x_train should have 2 dimensions! X_dim:{}"
                            .format(X_train.ndim))
        X_train = self._get_tensor(X_train)

        sample_size = self.n_training
        if np.isscalar(self.ridge):
            ridge = np.ones(sample_size) * self.ridge
        else:
            raise Exception(f"rideg is not set properly!")

        assert isinstance(ridge, np.ndarray)
        assert ridge.ndim == 1

        ridge = self._get_tensor(ridge)
        K = self.magnitude * th.exp(-self.gs_dist(X_train, X_train) / self.length_scale) + th.diag(ridge)
        K_inv = K.inverse()

        y_tensor_dict = {}
        for obj in self.objs:
            y_tensor_dict[obj] = self._get_tensor(y_dict[obj])
        return [X_train, y_tensor_dict, K_inv]

    def get_kernel(self, x1, x2):
        '''
        get kernel
        :param x1: Tensor(n_training_samples, n_vars), training data
        :param x2: Tensor(n_x, n_vars),
        :return:
        '''
        return self.magnitude * th.exp(-self.gs_dist(x1, x2) / self.length_scale)

    def objective_std(self, X_test, X_train, K_inv, y_scale):
        '''
        [todo] used for loss of inaccurate models in MOGD
        :param X_test: Tensor(n_x, n_vars), input of the predictive model, where n_x shows the number of input variables
        :param X_train: Tensor(n_training_samples, n_vars), input to train GPR models
        :param K_inv: Tensor(n_training_samples, n_training_samples), inversion of the covariance matrix
        :param y_scale:
        :return:
        '''
        K_tete = self.get_kernel(X_test, X_test) # (1,1)
        K_tetr = self.get_kernel(X_test, X_train) # (1, N)
        K_trte = K_tetr.t() # (N,1)
        var = K_tete - th.matmul(th.matmul(K_tetr, K_inv), K_trte) # (1,1)
        var_diag = var.diag()
        try:
            std = th.sqrt(var_diag)
        except:
            std = var_diag - var_diag
            print('!!! var < 0')
        return std * y_scale

    def objective(self, X_test, X_train, y_train, K_inv):
        '''
        call GPR model to get objective values
        :param X_test: Tensor(n_x, n_vars), input of the predictive model, where n_x shows the number of input variables
        :param X_train: Tensor(n_training_samples, n_vars), input to train GPR models
        :param y_train: Tensor(n_training_samples,), training data of true objective values
        :param K_inv: Tensor(n_training_samples, n_training_samples), inversion of the covariance matrix
        :return:
                yhat: Tensor(n_x,), the prediction of the objective
        '''
        if X_test.ndimension() == 1:
            X_test = X_test.reshape(1, -1)
        length_scale = self.length_scale
        K2 = self.magnitude * th.exp(-self.gs_dist(X_train, X_test) / length_scale)
        K2_trans = K2.t()
        yhat = th.matmul(K2_trans, th.matmul(K_inv, y_train))
        return yhat

    def predict(self, obj, X_test):
        '''
        get prediction of the objective based on the GPR predictive model
        :param obj: str, name of the objective
        :param X_test: ndarray(n_x, n_vars), input of the predictive model, where n_x shows the number of input variables
        :return:
                yhat: Tensor(n_x,), the prediction of the objective
        '''
        assert obj in self.objs
        X_test = self._get_tensor(X_test)
        X_train, y_dict, K_inv = self.models
        y_train = y_dict[obj]
        yhat = self.objective(X_test, X_train, y_train, K_inv)
        yhat_np = yhat.numpy()
        GPR.check_output(yhat_np)
        return yhat_np

    def _get_tensor(self, x, dtype=None, device=None):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return th.tensor(x, dtype=dtype, device=device)

    @staticmethod
    def check_output(X):
        finite_els = np.isfinite(X)
        if not np.all(finite_els):
            raise Exception("Input contains non-finite values: {}"
                            .format(X[~finite_els]))
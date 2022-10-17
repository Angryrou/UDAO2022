# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example of Gaussian Process Regressor (GPR) model
#
# Created at 17/10/2022
from utils.optimization.configs_parser import ConfigsParser
from utils.optimization.base_model import BaseModel

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# an example of noise-free model by following: https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html
class GPR(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_params = ConfigsParser().parse_details(option="model")
        self.initialize()

    def initialize(self):
        random_state = self.model_params["random_state"]
        self.rng = np.random.RandomState(random_state)
        self.n_training = self.model_params["n_training"]
        self.n_restarts_optimizer = self.model_params["n_restarts_optimizer"]
        self.length_scale = self.model_params["length_scale"]
        self.length_scale_bounds = self.model_params["length_scale_bounds"]

    def fit(self, obj, vars):
        # objective functions are the same as that in HCF
        if obj == "obj_1":
            y = 4 * vars[:, 0] * vars[:, 0] + 4 * vars[:, 1] * vars[:, 1]
        elif obj == "obj_2":
            y = (vars[:, 0] - 5) * (vars[:, 0] - 5) + (vars[:, 1] - 5) * (vars[:, 1] - 5)
        else:
            raise Exception(f"Objective {obj} is not configured in the configuration file!")

        training_indices = self.rng.choice(np.arange(y.size), size=self.n_training, replace=False)
        X_train, y_train = vars[training_indices], y[training_indices]

        assert X_train.shape[0] == y_train.shape[0]
        assert X_train.shape[1] == vars.shape[1]

        kernel = 1 * RBF(length_scale=self.length_scale, length_scale_bounds=self.length_scale_bounds)
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=self.n_restarts_optimizer)
        self.gpr.fit(X_train, y_train)

    def predict(self, obj, vars):
        self.fit(obj, vars)
        mean_prediction, std_prediction = self.gpr.predict(vars, return_std=True)

        return mean_prediction
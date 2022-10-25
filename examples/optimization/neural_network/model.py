# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example of Neural Network (NN) model
#              and pre-define functions (objectives and constraints) based on NN
#
# Created at 17/10/2022

from utils.optimization.configs_parser import ConfigsParser
from optimization.model.base_model import BaseModel

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# an example by following: https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
class NN(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_params = ConfigsParser().parse_details(option="model")
        self.initialize()

    def initialize(self):
        self.in_features = self.model_params["in_features"]

    def fit(self):
        pass

    def predict(self, name, vars):
        # one-layer NN
        th.manual_seed(1)
        vars = th.Tensor(vars)
        if name == "obj_1":
            fc = nn.Linear(self.in_features, 1)
            X = F.relu(fc(vars))
            value = np.squeeze(X.detach().numpy())
        elif name == "const_1":
            fc = nn.Linear(self.in_features, 1)
            X = F.relu(fc(vars))
            value = np.squeeze(X.detach().numpy())
        else:
            raise Exception(f"Objective/constraint {name} is not configured in the configuration file!")
        return value

class NNPredictiveModels:
    # initialize model
    nn = NN()

    @classmethod
    def predict_obj1(cls, vars):
        obj = "obj_1"
        value = cls.nn.predict(obj, vars)
        return value

    @classmethod
    def predict_obj2(cls, vars):
        obj = "obj_2"
        value = (vars[:, 0] - 5) * (vars[:, 0] - 5) + (vars[:, 1] - 5) * (vars[:, 1] - 5)
        return value

    # constraints support left hand side (LHS) e.g. g(x1, x2, ...) - C <= 0
    @classmethod
    def const_func1(cls, vars):
        C = 10
        const = "const_1"
        value = cls.nn.predict(const, vars) - C
        return value

    @staticmethod
    def const_func2(vars):
        value = (vars[:, 0] - 8) * (vars[:, 0] - 8) + (vars[:, 1] + 3) * (vars[:, 1] + 3) - 7.7
        return value

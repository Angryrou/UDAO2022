# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example of Neural Network (NN) model
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

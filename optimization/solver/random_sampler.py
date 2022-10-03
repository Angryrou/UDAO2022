# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: random sampler
#
# Created at 15/09/2022

import numpy as np
from optimization.solver.base_solver import BaseSolver

class RandomSampler(BaseSolver):
    def __init__(self, rs_params):
        '''

        :param rs_params: int, the number of samples per variable
        '''
        super().__init__()
        self.n_samples_per_param = rs_params["n_samples"]
        self.seed = rs_params["seed"]

    def _rand_float(self, lower, upper, n_samples):
        '''
        generate n_samples random float values within the lower and upper bounds
        :param lower: int, lower bound
        :param upper: int upper bound
        :param n_samples: int, the number of samples
        :return: ndarray(n_samples, ), n_samples random float
        '''
        if lower > upper:
            return None
        else:
            scale = upper - lower
            out = np.random.rand(n_samples) * scale + lower
            return out

    def _get_input(self, bounds, var_types):
        '''
        generate samples of variables
        :param bounds: array (n_vars, 2), 2 refers to the lower and upper bounds
        :param var_types: list, type of each variable
        :return: array, variables (n_samples * n_vars)
        '''
        n_vars = bounds.shape[0]
        x = np.zeros([self.n_samples_per_param, n_vars])
        np.random.seed(self.seed)
        for i, values in enumerate(bounds):
            upper, lower = values[1], values[0]
            if (lower - upper) > 0:
                print("ERROR: lower bound is greater than the upper bound!")
                raise ValueError(bounds)
            # randomly sample n_samples within the range
            if var_types[i] == "FLOAT":
                x[:, i] = self._rand_float(lower, upper, self.n_samples_per_param)
            elif var_types[i] == "INTEGER" or var_types[i] == "BINARY":
                x[:, i] = np.random.randint(lower, upper + 1, size=self.n_samples_per_param)
            elif var_types[i] == "ENUM":
                inds = np.random.randint(0, len(values), size=self.n_samples_per_param)
                x[:, i] = np.array(values)[inds]
            else:
                raise ValueError(var_types[i])
        return x

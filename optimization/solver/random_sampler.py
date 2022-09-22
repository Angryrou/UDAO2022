# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: random sampler
#
# Created at 15/09/2022

import numpy as np
from optimization.solver.base_solver import BaseSolver

class RandomSampler(BaseSolver):
    def __init__(self, rs_params):
        super().__init__()
        self.n_samples_per_param = rs_params
        self.seed = 0

    def _rand_float(self, lower, upper, n_samples):
        ## generate n_samples random float values within the lower and upper bounds
        if lower > upper:
            return None
        else:
            scale = upper - lower
            out = np.random.rand(n_samples) * scale + lower
            return out

    def _get_input(self, bounds, var_types):
        '''

        :param bounds: array (2 * n_vars), 2 refers to the lower and upper bounds
        :param var_types: list, type of each variable
        :return: array, vararibles (n_samples * n_vars)
        '''
        if any((bounds[:, 0] - bounds[:, 1]) > 0):
            print("ERROR: lower bound is greater than the upper bound!")
            raise ValueError(bounds)
        n_vars = bounds.shape[0]
        x = np.zeros([self.n_samples_per_param, n_vars])
        np.random.seed(self.seed)
        for i, [lower, upper] in enumerate(bounds):
            # randomly sample n_samples within the range
            if var_types[i] == "FLOAT":
                x[:, i] = self._rand_float(lower, upper, self.n_samples_per_param)
            elif var_types[i] == "INTEGER" or var_types[i] == "BINARY":
                x[:, i] = np.random.randint(lower, upper + 1, size=self.n_samples_per_param)
            else:
                raise ValueError(var_types[i])
        return x

## a test on _get_input
if __name__ == '__main__':
    ## for variables
    n_vars = 3
    lower = np.array([[1], [2], [3]])
    upper = np.array([[6], [6], [8]])
    bounds = np.hstack([lower, upper])
    var_types = ["FLOAT", "FLOAT", "INTEGER"]

    rs_params = 1000

    test_vars = RandomSampler(rs_params)
    vars = test_vars._get_input(bounds, var_types)
    print(vars)
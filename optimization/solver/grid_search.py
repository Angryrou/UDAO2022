# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Grid-Search
#
# Created at 15/09/2022

import numpy as np
import itertools

from optimization.solver.base_solver import BaseSolver

class GridSearch(BaseSolver):
    def __init__(self, gs_params):
        '''

        :param gs_params: int, the number of grids for each variable
        '''
        super().__init__()
        self.n_grids_per_param = gs_params

    def _get_input(self, bounds, var_types):
        '''
        grid search on each variable
        :param bounds: ndarray (n_vars, 2), 2 refers to the lower and upper bounds
        :param var_types: list, type of each variable
        :return: array, variables (n_grids * n_vars)
        '''

        if any((bounds[:, 0] - bounds[:, 1]) > 0):
            print("ERROR: lower bound is greater than the upper bound!")
            raise ValueError(bounds)

        ## generate grids for each variable
        grids_list = []
        for i, [lower, upper] in enumerate(bounds):
            # the number of points generated for each variable
            n_grids_per_var = self.n_grids_per_param

            # make sure the grid point is the same with the type
            # e.g., if int x.min=0, x.max=5, n_grids_per_var=10, ONLY points[0, 1, 2, 3, 4, 5] are feasible
            if var_types[i] == "INTEGER" or var_types[i] == "BINARY":
                if self.n_grids_per_param > (upper - lower + 1):
                    n_grids_per_var = upper - lower + 1

            grids_per_var = np.linspace(lower, upper, num=n_grids_per_var, endpoint=True)
            grids_list.append(grids_per_var)

        ## generate cartesian product of grids_list
        x = np.array([list (i) for i in itertools.product(*grids_list)])

        return x

## a test on _get_input
if __name__ == '__main__':
    ## for variables
    n_vars = 3
    lower = np.array([[1], [2], [3]])
    upper = np.array([[6], [6], [8]])
    bounds = np.hstack([lower, upper])
    var_types = ["FLOAT", "FLOAT", "INTEGER"]

    gs_params = 10

    test_vars = GridSearch(gs_params)
    vars = test_vars._get_input(bounds, var_types)
    print(vars)

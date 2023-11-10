import itertools
from dataclasses import dataclass
from typing import List

import numpy as np

from ..concepts.variable import EnumVariable, IntegerVariable, NumericVariable, Variable
from .base_solver import BaseSolver


class GridSearch(BaseSolver):
    @dataclass
    class Params:
        n_grids_per_var: List[int]

    def __init__(self, gs_params: Params) -> None:
        """
        :param gs_params: dict, the parameters used in grid_search
        """
        super().__init__()
        self.n_grids_per_var = gs_params.n_grids_per_var

    def _get_input(self, variables: List[Variable]) -> np.ndarray:
        """
        generate grids for each variable
        :param var_ranges: ndarray (n_vars,),
            the lower and upper var_ranges of non-ENUM variables,
            and values of ENUM variables
        :param var_types: list, type of each variable
        :return: array, variables (n_grids * n_vars)
        """

        grids_list = []
        for i, var in enumerate(variables):
            # the number of points generated for each variable
            n_grids_per_var = self.n_grids_per_var[i]

            if isinstance(var, NumericVariable):
                # make sure the grid point is the same with the type
                # e.g., if int x.min=0, x.max=5, n_grids_per_var=10,
                # ONLY points[0, 1, 2, 3, 4, 5] are feasible
                if isinstance(var, IntegerVariable):
                    if n_grids_per_var > (var.upper - var.lower + 1):
                        n_grids_per_var = int(var.upper - var.lower + 1)

                grids_per_var = np.linspace(
                    var.lower, var.upper, num=n_grids_per_var, endpoint=True
                )
                if isinstance(var, IntegerVariable):
                    grids_per_var = np.round(grids_per_var).astype(int)
            elif isinstance(var, EnumVariable):
                grids_per_var = np.array(var.values)
            else:
                raise NotImplementedError(
                    f"ERROR: variable type {type(var)} is not supported!"
                )
            grids_list.append(grids_per_var)

        ## generate cartesian product of grids_list
        x = np.array([list(i) for i in itertools.product(*grids_list)])

        return x

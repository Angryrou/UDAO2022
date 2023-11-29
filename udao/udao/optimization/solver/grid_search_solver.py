import itertools
from dataclasses import dataclass
from typing import Dict, List, Mapping

import numpy as np

from ..concepts import EnumVariable, IntegerVariable, NumericVariable, Variable
from .sampler_solver import SamplerSolver


class GridSearch(SamplerSolver):
    @dataclass
    class Params:
        n_grids_per_var: List[int]

    def __init__(self, gs_params: Params) -> None:
        """
        :param gs_params: dict, the parameters used in grid_search
        """
        super().__init__()
        self.n_grids_per_var = gs_params.n_grids_per_var

    def _process_variable(self, var: Variable, n_grids: int) -> np.ndarray:
        if isinstance(var, NumericVariable):
            # make sure the grid point is the same with the type
            # e.g., if int x.min=0, x.max=5, n_grids_per_var=10,
            # ONLY points[0, 1, 2, 3, 4, 5] are feasible
            if isinstance(var, IntegerVariable):
                if n_grids > (var.upper - var.lower + 1):
                    n_grids = int(var.upper - var.lower + 1)

            var_grid = np.linspace(var.lower, var.upper, num=n_grids, endpoint=True)
            if isinstance(var, IntegerVariable):
                return np.round(var_grid).astype(int)
            return var_grid
        elif isinstance(var, EnumVariable):
            return np.array(var.values)
        else:
            raise NotImplementedError(
                f"ERROR: variable type {type(var)} is not supported!"
            )

    def _get_input(self, variables: Mapping[str, Variable]) -> Dict[str, np.ndarray]:
        """
        generate grids for each variable
        :param var_ranges: ndarray (n_vars,),
            the lower and upper var_ranges of non-ENUM variables,
            and values of ENUM variables
        :param var_types: list, type of each variable
        :return: array, variables (n_grids * n_vars)
        """
        grids_list = []
        variable_names = list(variables.keys())

        for i, var_name in enumerate(variable_names):
            var = variables[var_name]
            var_n_grids = self.n_grids_per_var[i]
            grids_list.append({var_name: self._process_variable(var, var_n_grids)})

        values_list = [list(d.values())[0] for d in grids_list]
        cartesian_product = np.array([list(i) for i in itertools.product(*values_list)])
        result_dict = {
            var_name: cartesian_product.T[i]
            for i, var_name in enumerate(variable_names)
        }

        return result_dict

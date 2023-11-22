import itertools
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..concepts import (
    Constraint,
    EnumVariable,
    IntegerVariable,
    NumericVariable,
    Objective,
    Variable,
)
from ..utils.exceptions import NoSolutionError
from ..utils.moo_utils import Point
from .base_solver import BaseSolver
from .utils import filter_on_constraints


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
            var_n_grids = self.n_grids_per_var[i]

            grids_list.append(self._process_variable(var, var_n_grids))

        ## generate cartesian product of grids_list
        x = np.array([list(i) for i in itertools.product(*grids_list)])

        return x

    def solve(
        self,
        objective: Objective,
        constraints: List[Constraint],
        variables: List[Variable],
        wl_id: Optional[str],
    ) -> Point:
        filtered_vars = filter_on_constraints(
            wl_id, self._get_input(variables), constraints
        )
        if filtered_vars.shape[0] == 0:
            raise NoSolutionError("No feasible solution found!")
        objective_value = np.array(
            objective.function(filtered_vars, wl_id=wl_id)
        ).reshape(-1, 1)
        op_ind = int(np.argmin(objective_value))

        return Point(objs=objective_value[op_ind], vars=filtered_vars[op_ind])

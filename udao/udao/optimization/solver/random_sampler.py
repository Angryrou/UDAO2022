from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..concepts import (
    Constraint,
    EnumVariable,
    FloatVariable,
    IntegerVariable,
    Objective,
    Variable,
)
from ..utils.exceptions import NoSolutionError
from ..utils.moo_utils import Point
from .base_solver import BaseSolver
from .utils import filter_on_constraints


class RandomSampler(BaseSolver):
    @dataclass
    class Params:
        n_samples_per_param: int
        "the number of samples per variable"

        seed: Optional[int] = None
        "random seed for generatino of samples"

    def __init__(self, params: Params) -> None:
        """
        :param params: RandomSampler.Params
        """
        super().__init__()
        self.n_samples_per_param = params.n_samples_per_param
        self.seed = params.seed

    def _process_variable(self, var: Variable) -> np.ndarray:
        """Generate samples of a variable"""
        if isinstance(var, FloatVariable):
            return np.random.uniform(var.lower, var.upper, self.n_samples_per_param)
        elif isinstance(var, IntegerVariable):
            return np.random.randint(
                var.lower, var.upper + 1, size=self.n_samples_per_param
            )
        elif isinstance(var, EnumVariable):
            inds = np.random.randint(0, len(var.values), size=self.n_samples_per_param)
            return np.array(var.values)[inds]
        else:
            raise NotImplementedError(
                f"ERROR: variable type {type(var)} is not supported!"
            )

    def _get_input(self, variables: List[Variable]) -> np.ndarray:
        """
        generate samples of variables

        Parameters:
        -----------
        variables: List[Variable],
            lower and upper var_ranges of variables(non-ENUM),
            and values of ENUM variables
        Returns:
        np.ndarray,
            variables (n_samples * n_vars)
        """
        n_vars = len(variables)
        x = np.zeros([self.n_samples_per_param, n_vars])
        np.random.seed(self.seed)
        for i, var in enumerate(variables):
            x[:, i] = self._process_variable(var)
        return x

    def solve(
        self,
        objective: Objective,
        constraints: List[Constraint],
        variables: List[Variable],
        wl_id: str | None,
    ) -> Point:
        filtered_vars = filter_on_constraints(
            wl_id, self._get_input(variables), constraints
        )
        if not filtered_vars.size:
            raise NoSolutionError("No feasible solution.")

        objective_value = np.array(
            objective.function(filtered_vars, wl_id=wl_id)
        ).reshape(-1, 1)
        op_ind = int(np.argmin(objective_value))
        return Point(objs=np.array(objective_value[op_ind]), vars=filtered_vars[op_ind])

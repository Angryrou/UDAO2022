from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..concepts import Constraint, Objective, Variable
from ..utils.exceptions import NoSolutionError
from ..utils.moo_utils import Point
from .base_solver import BaseSolver
from .utils import filter_on_constraints


class SamplerSolver(BaseSolver, ABC):
    @abstractmethod
    def _get_input(self, variables: Mapping[str, Variable]) -> Dict[str, np.ndarray]:
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
        pass

    def solve(
        self,
        objective: Objective,
        variables: Mapping[str, Variable],
        constraints: Optional[List[Constraint]] = None,
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> Point:
        """Solve a single-objective optimization problem

        Parameters
        ----------
        objective : Objective
            Objective to be optimized
        variables : Dict[str, Variable]
            Variables to be optimized
        constraints : Optional[List[Constraint]], optional
            List of constraints to comply with, by default None
        input_parameters : Optional[Dict[str, Any]], optional
            Fixed parameters input to objectives and/or constraints,
            by default None

        Returns
        -------
        Point
            A point that satisfies the constraints
            and optimizes the objective

        Raises
        ------
        NoSolutionError
            If no feasible solution is found
        """
        if constraints is None:
            constraints = []
        variable_values = self._get_input(variables)
        filtered_vars = filter_on_constraints(
            input_parameters=input_parameters,
            input_vars=variable_values,
            constraints=constraints,
        )
        if any([len(v) == 0 for v in filtered_vars.values()]):
            raise NoSolutionError("No feasible solution found!")
        objective_value = np.array(
            objective.function(filtered_vars, input_parameters=input_parameters)
        ).reshape(-1, 1)
        op_ind = int(np.argmin(objective_value * objective.direction))

        return Point(
            objs=objective_value[op_ind],
            vars={k: v[op_ind] for k, v in filtered_vars.items()},
        )

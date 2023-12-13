from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..concepts import Constraint, Objective, Variable
from ..utils.exceptions import NoSolutionError
from .base_solver import BaseSolver


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
        constraints: Optional[Sequence[Constraint]] = None,
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
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
        filtered_vars = self.filter_on_constraints(
            input_parameters=input_parameters,
            input_vars=variable_values,
            constraints=constraints,
        )
        if any([len(v) == 0 for v in filtered_vars.values()]):
            raise NoSolutionError("No feasible solution found!")
        th_value = objective.function(
            input_variables=filtered_vars, input_parameters=input_parameters
        )
        objective_value = th_value.numpy().reshape(-1, 1)
        op_ind = int(np.argmin(objective_value * objective.direction))

        return (
            objective_value[op_ind],
            {k: v[op_ind] for k, v in filtered_vars.items()},
        )

    @staticmethod
    def filter_on_constraints(
        input_parameters: Optional[Dict[str, Any]],
        input_vars: Dict[str, np.ndarray],
        constraints: Sequence[Constraint],
    ) -> Dict[str, np.ndarray]:
        """Keep only input variables that don't violate constraints

        Parameters:
        -----------
        wl_id : str | None
            workload id
        input_vars : np.ndarray
            input variables
        constraints : List[Constraint]
            constraint functions
        """
        if not constraints:
            return input_vars

        available_indices = np.arange(len(next(iter(input_vars.values()))))
        for constraint in constraints:
            const_values = constraint.function(
                input_variables=input_vars, input_parameters=input_parameters
            )
            if constraint.upper is not None:
                available_indices = np.intersect1d(
                    available_indices, np.where(const_values <= constraint.upper)
                )
            if constraint.lower is not None:
                available_indices = np.intersect1d(
                    available_indices, np.where(const_values >= constraint.lower)
                )
        filtered_vars = {k: v[available_indices] for k, v in input_vars.items()}
        return filtered_vars

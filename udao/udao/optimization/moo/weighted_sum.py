import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch as th

from ..concepts import Objective
from ..concepts.problem import MOProblem, SOProblem
from ..concepts.utils import InputParameters, InputVariables
from ..soo.base_solver import SOSolver
from ..utils import moo_utils as moo_ut
from ..utils.exceptions import NoSolutionError
from ..utils.moo_utils import Point
from .base_moo import MOSolver


class WeightedSumObjective(Objective):
    """Weighted Sum Objective"""

    def __init__(
        self,
        objectives: Sequence[Objective],
        ws: List[float],
        allow_cache: bool = False,
    ) -> None:
        self.objectives = objectives
        self.ws = ws
        super().__init__(
            name="weighted_sum", function=self.function, direction_type="MIN"
        )
        self._cache: Dict[str, np.ndarray] = {}
        self.allow_cache = allow_cache

    def _function(
        self, input_variables: InputVariables, input_parameters: InputParameters = None
    ) -> np.ndarray:
        hash_var = ""
        if self.allow_cache:
            hash_var = json.dumps(vars)
            if hash_var in self._cache:
                return self._cache[hash_var]
        objs: List[np.ndarray] = []
        for objective in self.objectives:
            obj = (
                objective.function(input_variables, input_parameters)
                * objective.direction
            )
            objs.append(obj.numpy().squeeze())

        # shape (n_feasible_samples/grids, n_objs)
        objs_array = np.array(objs).T
        if self.allow_cache:
            self._cache[hash_var] = objs_array
        return objs_array

    def function(
        self, input_variables: InputVariables, input_parameters: InputParameters = None
    ) -> th.Tensor:
        """Sum of weighted normalized objectives"""
        objs_array = self._function(input_variables, input_parameters)
        objs_norm = self._normalize_objective(objs_array)
        return th.tensor(np.sum(objs_norm * self.ws, axis=1))

    def _normalize_objective(self, objs_array: np.ndarray) -> np.ndarray:
        """Normalize objective values to [0, 1]

        Parameters
        ----------
        objs_array : np.ndarray
            shape (n_feasible_samples/grids, n_objs)

        Returns
        -------
        np.ndarray
            shape (n_feasible_samples/grids, n_objs)

        Raises
        ------
        NoSolutionError
            if lower bounds of objective values are
            higher than their upper bounds
        """
        objs_array = objs_array
        objs_min, objs_max = objs_array.min(0), objs_array.max(0)

        if any((objs_min - objs_max) > 0):
            raise NoSolutionError(
                "Cannot do normalization! Lower bounds of "
                "objective values are higher than their upper bounds."
            )
        return (objs_array - objs_min) / (objs_max - objs_min)


class WeightedSum(MOSolver):
    """
    Weighted Sum (WS) algorithm for MOO

    Parameters
    ----------
    ws_pairs: np.ndarray,
        weight settings for all objectives, of shape (n_weights, n_objs)
    inner_solver: BaseSolver,
        the solver used in Weighted Sum
    objectives: List[Objective],
        objective functions
    constraints: List[Constraint],
        constraint functions

    """

    def __init__(
        self,
        ws_pairs: np.ndarray,
        so_solver: SOSolver,
        allow_cache: bool = False,
    ):
        super().__init__()
        self.so_solver = so_solver
        self.ws_pairs = ws_pairs
        self.allow_cache = allow_cache

    def solve(self, problem: MOProblem) -> Tuple[np.ndarray, np.ndarray]:
        """solve MOO problem by Weighted Sum (WS)

        Parameters
        ----------
        variables : List[Variable]
            List of the variables to be optimized.
        input_parameters : Optional[Dict[str, Any]]
            Fixed input parameters expected by
            the objective functions.

        Returns
        -------
        Tuple[Optional[np.ndarray],Optional[np.ndarray]]
            Pareto solutions and corresponding variables.
        """
        candidate_points: List[Point] = []
        objective = WeightedSumObjective(
            problem.objectives, self.ws_pairs[0], self.allow_cache
        )
        for ws in self.ws_pairs:
            objective.ws = ws
            _, soo_vars = self.so_solver.solve(
                SOProblem(
                    objective,
                    constraints=problem.constraints,
                    variables=problem.variables,
                    input_parameters=problem.input_parameters,
                )
            )

            objective_values = objective._function(
                soo_vars, input_parameters=problem.input_parameters  # type: ignore
            )

            candidate_points.append(Point(objective_values, soo_vars))

        return moo_ut.summarize_ret(
            [point.objs for point in candidate_points],
            [point.vars for point in candidate_points],
        )

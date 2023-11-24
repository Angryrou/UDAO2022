import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ..concepts import Constraint, NumericVariable, Objective, Variable
from ..solver.base_solver import BaseSolver
from ..utils import moo_utils as moo_ut
from ..utils.exceptions import NoSolutionError
from ..utils.moo_utils import Point
from .base_moo import BaseMOO


class WeightedSumObjective(Objective):
    """Weighted Sum Objective"""

    def __init__(
        self, objectives: List[Objective], ws: List[float], allow_cache: bool = False
    ) -> None:
        self.objectives = objectives
        self.ws = ws
        super().__init__(
            name="weighted_sum", function=self.function, direction_type="MIN"
        )
        self._cache: Dict[str, np.ndarray] = {}
        self.allow_cache = allow_cache

    def _function(self, vars: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        hash_var = ""
        if self.allow_cache:
            hash_var = hashlib.md5(vars.data.tobytes()).hexdigest()
            if hash_var in self._cache:
                return self._cache[hash_var]
        objs: List[np.ndarray] = []
        for objective in self.objectives:
            obj = objective.function(vars, **kwargs) * objective.direction
            objs.append(obj.numpy().squeeze())

        # shape (n_feasible_samples/grids, n_objs)
        objs_array = np.array(objs).T
        if self.allow_cache:
            self._cache[hash_var] = objs_array
        return objs_array

    def function(self, vars: np.ndarray, *args: Any, **kwargs: Any) -> th.Tensor:
        """Sum of weighted normalized objectives"""
        objs_array = self._function(vars, *args, **kwargs)
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


class WeightedSum(BaseMOO):
    """
    Weighted Sum (WS) algorithm for MOO

    Parameters:
    ------------
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
        inner_solver: BaseSolver,
        objectives: List[Objective],
        constraints: List[Constraint],
        allow_cache: bool = False,
    ):
        super().__init__()
        self.inner_solver = inner_solver
        self.ws_pairs = ws_pairs
        self.objectives = objectives
        self.constraints = constraints
        self.allow_cache = allow_cache

    def solve(
        self, wl_id: Optional[str], variables: List[Variable]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """solve MOO problem by Weighted Sum (WS)

        Parameters
        ----------
        wl_id : Optional[str]
            workload id
        variables : List[Variable]
            List of the variables to be optimized.

        Returns
        -------
        Tuple[Optional[np.ndarray],Optional[np.ndarray]]
            Pareto solutions and corresponding variables.
        """
        candidate_points: List[Point] = []
        objective = WeightedSumObjective(
            self.objectives, self.ws_pairs[0], self.allow_cache
        )
        for ws in self.ws_pairs:
            objective.ws = ws
            point = self.inner_solver.solve(
                objective, self.constraints, variables, wl_id
            )
            objective_values = objective._function(np.array([point.vars]), wl_id=wl_id)  # type: ignore
            point.objs = objective_values
            candidate_points.append(point)

        return moo_ut.summarize_ret(
            [point.objs for point in candidate_points],
            [point.vars for point in candidate_points],
        )


def solve_ws(
    job_ids: List[str],
    solver: BaseSolver,
    n_probes: int,
    n_objs: int,
    objectives: List[Objective],
    constraints: List[Constraint],
    variables: List[Variable],
    wl_ranges: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]], list[float]]:
    """Temporary solve function for WS,
    replacing the call to solve() in GenericMOO"""
    po_objs_list: List[Optional[np.ndarray]] = []
    po_vars_list: List[Optional[np.ndarray]] = []
    time_cost_list: list[float] = []

    ws_steps = 1 / (n_probes - n_objs - 1)
    ws_pairs = moo_ut.even_weights(ws_steps, n_objs)
    ws = WeightedSum(
        ws_pairs=ws_pairs,
        inner_solver=solver,
        objectives=objectives,
        constraints=constraints,
    )

    for wl_id in job_ids:
        # fixme: to be generalized further
        if wl_ranges is not None and wl_id is not None:
            vars_max, vars_min = wl_ranges[wl_id]
            for variable, var_max, var_min in zip(variables, vars_max, vars_min):
                if isinstance(variable, NumericVariable):
                    variable.lower = var_min
                    variable.upper = var_max
        else:
            pass
        time0 = time.time()
        try:
            po_objs, po_vars = ws.solve(wl_id, variables)
        except NoSolutionError:
            po_objs, po_vars = None, None
        time_cost_list.append(time.time() - time0)
        po_objs_list.append(po_objs)
        po_vars_list.append(po_vars)

    return po_objs_list, po_vars_list, time_cost_list

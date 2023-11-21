import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..concepts import Constraint, NumericVariable, Objective, Variable
from ..solver.base_solver import BaseSolver
from ..utils import moo_utils as moo_ut
from ..utils.exceptions import NoSolutionError
from .base_moo import BaseMOO


class WeightedSum(BaseMOO):
    """
    Weighted Sum (WS) algorithm for MOO

    Parameters:
    ------------
    ws_pairs: ndarray(n_weights, n_objs),
        weight settings for all objectives
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
    ):
        super().__init__()
        self.inner_solver = inner_solver
        self.ws_pairs = ws_pairs
        self.objectives = objectives
        self.constraints = constraints

    def _filter_on_constraints(
        self, wl_id: Optional[str], input_vars: np.ndarray
    ) -> np.ndarray:
        """Keep only input variables that don't violate constraints"""
        if not self.constraints:
            return input_vars

        available_indices = np.array(range(len(input_vars)))
        for constraint in self.constraints:
            const_values = constraint.function(input_vars, wl_id=wl_id)
            if constraint.type == "<=":
                compliant_indices = np.where(const_values <= 0)
            elif constraint.type == ">=":
                compliant_indices = np.where(const_values >= 0)
            else:
                compliant_indices = np.where(const_values == 0)
            available_indices = np.intersect1d(compliant_indices, available_indices)
        filtered_vars = input_vars[available_indices]
        return filtered_vars

    def _normalize_objective(self, objs_array: np.ndarray) -> np.ndarray:
        objs_min, objs_max = objs_array.min(0), objs_array.max(0)

        if any((objs_min - objs_max) > 0):
            raise NoSolutionError(
                "Cannot do normalization! Lower bounds of "
                "objective values are higher than their upper bounds."
            )
        return (objs_array - objs_min) / (objs_max - objs_min)

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
        # TODO: we will compare the current WS implementation with
        # the existing WS numerical solver in the future,
        # and the one with better performance will be kept in the package.
        filtered_vars = self._filter_on_constraints(
            wl_id, self.inner_solver._get_input(variables)
        )

        if filtered_vars.size == 0:
            raise NoSolutionError(
                "No feasible solutions found. All candidate points violate constraints."
            )

        po_obj_list: List[np.ndarray] = []
        po_var_list: List[np.ndarray] = []

        objs: List[np.ndarray] = []
        for objective in self.objectives:
            obj = objective.function(filtered_vars, wl_id=wl_id) * objective.direction
            objs.append(obj.squeeze())

        # shape (n_feasible_samples/grids, n_objs)
        objs_array = np.array(objs).T

        objs_norm = self._normalize_objective(objs_array)
        for ws in self.ws_pairs:
            po_ind = self.get_soo_index(objs_norm, ws)
            po_obj_list.append(objs_array[po_ind])
            po_var_list.append(filtered_vars[po_ind])

        return moo_ut.summarize_ret(po_obj_list, po_var_list)

    def get_soo_index(self, objs: np.ndarray, ws_coeffs: List[float]) -> int:
        """
        Find argmin of single objective optimization
        problem corresponding to weighted sum of objectives

        Parameters
        ----------
        objs: np.ndarray,
            Shape(n_sample, n_objectives)
        ws_coeffs: List[float]
            One weight per objectives

        Returns
        -------
        int
            Argmin of weighted sum of objectives
        """
        obj = np.sum(objs * ws_coeffs, axis=1)
        return int(np.argmin(obj))


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

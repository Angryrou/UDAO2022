from abc import ABC
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from ....utils.interfaces import VarTypes
from ....utils.logging import logger
from ...concepts import Constraint, Objective
from ...concepts.problem import MOProblem, SOProblem
from ...soo.base_solver import SOSolver
from ...utils import solver_utils as solver_ut
from ...utils.exceptions import NoSolutionError
from ...utils.moo_utils import Point
from ..base_moo import MOSolver


class BaseProgressiveFrontier(MOSolver, ABC):
    """Base class for Progressive Frontier.
    Includes the common methods for Progressive Frontier.
    """

    @dataclass
    class Params:
        """Parameters for Progressive Frontier"""

        constraint_stress: float = 1e5
        objective_stress: float = 10.0

    def __init__(
        self,
        solver: SOSolver,
        params: Params,
    ) -> None:
        super().__init__()
        self.solver = solver
        self.constraint_stress = params.constraint_stress
        self.objective_stress = params.objective_stress
        self.opt_obj_ind = 0

    def get_anchor_point(
        self,
        problem: MOProblem,
        obj_ind: int,
    ) -> Point:
        """
        Find the anchor point for the given objective,
        by unbounded single objective optimization

        Parameters:
        ----------
        wl_id : str
            workload id
        obj_ind : int
            index of the objective to be optimized
        anchor_option : str
            choice for anchor points calculation

        Returns:
        -------
        Point
            anchor point for the given objective
        """
        try:
            _, soo_vars = self.solver.solve(
                SOProblem(
                    variables=problem.variables,
                    objective=problem.objectives[obj_ind],
                    constraints=problem.constraints,
                    input_parameters=problem.input_parameters,
                )
            )
        except NoSolutionError:
            raise NoSolutionError("Cannot find anchor points.")
        else:
            objs = self._compute_objectives(problem, soo_vars)

        # If the current objective type is Integer,
        # further find the optimal value for other objectives with float type
        if problem.objectives[obj_ind].type == "int":
            utopia_init = np.array(
                [0 if i != obj_ind else objs[obj_ind] for i in problem.objectives]
            )
            utopia_tmp, nadir_tmp = Point(objs=utopia_init), Point(objs=objs)
            # select the first objective with float type
            float_obj_ind = [
                i
                for i, objective in enumerate(problem.objectives)
                if objective.type == VarTypes.FLOAT
            ][0]
            obj_bounds_dict_so = self._form_obj_bounds_dict(
                problem, utopia_tmp, nadir_tmp
            )
            so_problem = self._so_problem_from_bounds_dict(
                problem, obj_bounds_dict_so, problem.objectives[float_obj_ind]
            )
            try:
                _, soo_vars_update = self.solver.solve(so_problem)
            except NoSolutionError:
                raise NoSolutionError("Cannot find anchor points.")
            else:
                objs = self._compute_objectives(problem, soo_vars)

                return Point(objs, soo_vars_update)
        else:
            return Point(objs, soo_vars)

    def _form_obj_bounds_dict(
        self, problem: MOProblem, utopia: Point, nadir: Point
    ) -> dict[str, list]:
        """
        form the dict used in the constrained optimization
        e.g. the format:
        obj_bounds_dict = {"latency": [solver_ut._get_tensor(0),
                      solver_ut._get_tensor(10000000)],
                      "cores": [solver_ut._get_tensor(0),
                      solver_ut._get_tensor(58)]
                      }
        Parameters:
        ----------
        utopia: Point
            the utopia point
        nadir: Point
            the nadir point
        opt_obj_ind: int
            the index of objective to be optimized

        Returns:
        -------
            dict with upper and lower bound for each objective
        """
        bounds = {}
        for i, objective in enumerate(problem.objectives):
            if objective.direction < 0:
                bounds[objective.name] = [
                    solver_ut.get_tensor(nadir.objs[i] * objective.direction),
                    solver_ut.get_tensor(utopia.objs[i] * objective.direction),
                ]
            else:
                bounds[objective.name] = [
                    solver_ut.get_tensor(utopia.objs[i] * objective.direction),
                    solver_ut.get_tensor(nadir.objs[i] * objective.direction),
                ]

        return bounds

    def _so_problem_from_bounds_dict(
        self,
        problem: MOProblem,
        obj_bounds_dict: dict[str, list],
        primary_obj: Objective,
    ) -> SOProblem:
        """

        Parameters
        ----------
        obj_bounds_dict : dict[str, list]
            A lower and upper bound for each objective
        primary_obj : Objective
            The objective to be optimized

        Returns
        -------
        Tuple[Objective, Sequence[Constraint]]
            The objective and constraints for the single-objective optimization
        """
        soo_constraints = list(problem.constraints)

        for obj in problem.objectives:
            obj_name = obj.name
            if obj_name != primary_obj.name:
                soo_constraints.append(
                    Constraint(
                        lower=obj_bounds_dict[obj_name][0],
                        upper=obj_bounds_dict[obj_name][1],
                        function=obj.function,
                        stress=self.objective_stress,
                    )
                )
        soo_objective = Objective(
            name=primary_obj.name,
            function=primary_obj.function,
            direction_type=primary_obj.direction_type,
            lower=obj_bounds_dict[primary_obj.name][0],
            upper=obj_bounds_dict[primary_obj.name][1],
        )
        so_problem = SOProblem(
            objective=soo_objective,
            variables=problem.variables,
            constraints=soo_constraints,
            input_parameters=problem.input_parameters,
        )
        return so_problem

    @staticmethod
    def get_utopia_and_nadir(points: list[Point]) -> Tuple[Point, Point]:
        """
        get the utopia and nadir points from a list of points
        Parameters:
        ----------
        points: list[Point],
            each element is a Point (defined class).

        Returns:
        -------
        Tuple[Point, Point]
            utopia and nadir point
        """
        if len(points) == 0:
            raise ValueError("The input list of points is empty.")
        n_objs = points[0].n_objs
        if any([point.n_objs != n_objs for point in points]):
            raise Exception("The number of objectives is not consistent among points.")
        best_objs = [min([point.objs[i] for point in points]) for i in range(n_objs)]
        worst_objs = [max([point.objs[i] for point in points]) for i in range(n_objs)]
        logger.debug(f"best_objs {best_objs}")
        utopia = Point(np.array(best_objs))
        nadir = Point(np.array(worst_objs))

        return utopia, nadir

    def _compute_objectives(
        self,
        problem: MOProblem,
        variable_values: dict[str, Any],
    ) -> np.ndarray:
        """Compute an array of objective for a given point.
        (variable_values is a dict of variable name and single value)

        Parameters
        ----------
        variable_values : dict[str, Any]
            Name: value of variables
        input_parameters : Optional[dict[str, Any]]
            Name: value of other fixed input parameters

        Returns
        -------
        np.ndarray
            _description_
        """
        obj_list = []
        for obj in problem.objectives:
            obj_value = (
                obj(
                    input_parameters=problem.input_parameters,
                    input_variables=variable_values,
                )
                * obj.direction
            ).squeeze()
            obj_list.append(obj_value)
        return np.array(obj_list)

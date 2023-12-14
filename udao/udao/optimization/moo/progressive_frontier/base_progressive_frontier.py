from abc import ABC
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ....utils.interfaces import VarTypes
from ....utils.logging import logger
from ...concepts import Constraint, Objective, Variable
from ...solver.mogd import MOGD
from ...utils import solver_utils as solver_ut
from ...utils.exceptions import NoSolutionError
from ...utils.moo_utils import Point
from ..base_moo import BaseMOO


class BaseProgressiveFrontier(BaseMOO, ABC):
    """Base class for Progressive Frontier.
    Includes the common methods for Progressive Frontier.
    """

    def __init__(
        self,
        solver_params: dict,
        variables: Dict[str, Variable],
        objectives: Sequence[Objective],
        constraints: Sequence[Constraint],
        constraint_stress: float = 1e5,
        objective_stress: float = 10.0,
    ) -> None:
        super().__init__()
        self.objectives = objectives
        self.constraint_stress = constraint_stress

        self.constraints = [
            Constraint(
                function=constraint.function,
                lower=constraint.lower,
                upper=constraint.upper,
                stress=self.constraint_stress,
            )
            for constraint in constraints
        ]

        self.variables = variables
        self.mogd = MOGD(MOGD.Params(**solver_params))

        self.objective_stress = objective_stress
        self.opt_obj_ind = 0

    def get_anchor_point(
        self,
        obj_ind: int,
        input_parameters: Optional[Dict[str, Any]] = None,
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
            _, soo_vars = self.mogd.solve(
                variables=self.variables,
                objective=self.objectives[obj_ind],
                constraints=self.constraints,
                input_parameters=input_parameters,
            )
        except NoSolutionError:
            raise NoSolutionError("Cannot find anchor points.")
        else:
            objs = self._compute_objectives(soo_vars, input_parameters)

        # If the current objective type is Integer,
        # further find the optimal value for other objectives with float type
        if self.objectives[obj_ind].type == "int":
            utopia_init = np.array(
                [0 if i != obj_ind else objs[obj_ind] for i in self.objectives]
            )
            utopia_tmp, nadir_tmp = Point(objs=utopia_init), Point(objs=objs)
            # select the first objective with float type
            float_obj_ind = [
                i
                for i, objective in enumerate(self.objectives)
                if objective.type == VarTypes.FLOAT
            ][0]
            obj_bounds_dict_so = self._form_obj_bounds_dict(utopia_tmp, nadir_tmp)
            soo_objective, soo_constraints = self._soo_params_from_bounds_dict(
                obj_bounds_dict_so, self.objectives[float_obj_ind]
            )
            try:
                _, soo_vars_update = self.mogd.solve(
                    objective=soo_objective,
                    constraints=soo_constraints,
                    input_parameters=input_parameters,
                    variables=self.variables,
                )
            except NoSolutionError:
                raise NoSolutionError("Cannot find anchor points.")
            else:
                objs = self._compute_objectives(soo_vars, input_parameters)

                return Point(objs, soo_vars_update)
        else:
            return Point(objs, soo_vars)

    def _form_obj_bounds_dict(self, utopia: Point, nadir: Point) -> dict[str, list]:
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
        return {
            objective.name: [
                solver_ut.get_tensor(utopia.objs[i]),
                solver_ut.get_tensor(nadir.objs[i]),
            ]
            for i, objective in enumerate(self.objectives)
        }

    def _soo_params_from_bounds_dict(
        self, obj_bounds_dict: dict[str, list], primary_obj: Objective
    ) -> Tuple[Objective, Sequence[Constraint]]:
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
        soo_constraints = list(self.constraints)

        for obj in self.objectives:
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
        return soo_objective, soo_constraints

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
        variable_values: dict[str, Any],
        input_parameters: Optional[dict[str, Any]],
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
        for obj in self.objectives:
            obj_value = (
                obj(input_parameters=input_parameters, input_variables=variable_values)
                * obj.direction
            ).squeeze()
            obj_list.append(obj_value)
        return np.array(obj_list)

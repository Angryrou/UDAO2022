from abc import ABC
from typing import Callable, List, Optional, Sequence, Tuple

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
    def __init__(
        self,
        solver_params: dict,
        variables: Sequence[Variable],
        objectives: Sequence[Objective],
        constraints: Sequence[Constraint],
        accurate: bool,
        std_func: Optional[Callable],
        alpha: float,
        precision_list: List[int],
    ) -> None:
        super().__init__()
        self.objectives = objectives
        self.constraints = constraints
        self.variables = variables
        self.mogd = MOGD(MOGD.Params(**solver_params))
        self.mogd.problem_setup(
            variables=variables,
            std_func=std_func,
            objectives=objectives,
            constraints=constraints,
            precision_list=precision_list,
            accurate=accurate,
            alpha=alpha,
        )

        self.opt_obj_ind = 0

    def get_anchor_point(
        self,
        wl_id: str,
        obj_ind: int,
        anchor_option: str = "2_step",
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
        obj_, vars = self.mogd.optimize_constrained_so(
            wl_id=wl_id,
            objective_name=self.objectives[obj_ind].name,
            obj_bounds_dict=None,
            batch_size=16,
        )
        if obj_ is None or vars is None:
            raise NoSolutionError("Cannot find anchor points.")
        objs = np.array(
            [obj_[i] * obj.direction for i, obj in enumerate(self.objectives)]
        )

        # If the current objective type is Integer,
        # further find the optimal value for other objectives with float type
        if anchor_option == "2_step" and self.objectives[obj_ind].type == "int":
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
            logger.debug(f"obj_bounds are: {obj_bounds_dict_so}")
            objs_update, vars_update = self.mogd.optimize_constrained_so(
                wl_id,
                objective_name=self.objectives[float_obj_ind].name,
                obj_bounds_dict=obj_bounds_dict_so,
            )
            if objs_update is None or vars_update is None:
                raise NoSolutionError("Cannot find anchor points.")
            return Point(np.array(objs_update), vars_update)
        else:
            if anchor_option not in ["1_step", "2_step"]:
                raise Exception(f"anchor_option {anchor_option} is not valid!")
            return Point(objs, vars)

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
        bounds = {}
        for i, objective in enumerate(self.objectives):
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

        utopia = Point(np.array(best_objs))
        nadir = Point(np.array(worst_objs))

        return utopia, nadir

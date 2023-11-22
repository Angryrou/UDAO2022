import heapq
import itertools
from abc import abstractmethod
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ...utils.logging import logger
from ..concepts import Constraint, Objective, Variable
from ..solver.mogd import MOGD
from ..utils import moo_utils as moo_ut
from ..utils import solver_utils as solver_ut
from ..utils.exceptions import NoSolutionError
from ..utils.moo_utils import Point, Rectangle
from ..utils.parameters import VarTypes
from .base_moo import BaseMOO


class AbstractProgressiveFrontier(BaseMOO):
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
        if (
            anchor_option == "2_step"
            and self.objectives[obj_ind].type == VarTypes.INTEGER
        ):
            utopia_init = np.array(
                [0 if i != obj_ind else objs[obj_ind] for i in self.objectives]
            )
            utopia_tmp, nadir_tmp = Point(objs=utopia_init), Point(objs=objs)
            # select the first objective with float type
            float_obj_ind = [
                i
                for i, objective in enumerate(self.objectives)
                if objective == VarTypes.FLOAT
            ][0]
            obj_bounds_dict_so = self._form_obj_bounds_dict(
                utopia_tmp, nadir_tmp, float_obj_ind
            )
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

    @abstractmethod
    def _form_obj_bounds_dict(
        self, utopia: Point, nadir: Point, opt_obj_ind: int
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
        pass

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


class SequentialProgressiveFrontier(AbstractProgressiveFrontier):
    """
    MOO by Progressive Frontier

    Parameters:
    ----------
    solver_params : dict
        parameters used in solver
    variables : Sequence[Variable]
        variables
    objectives : Sequence[Objective]
        objectives
    constraints : Sequence[Constraint]
        constraints
    accurate : bool
        whether the predictive model is accurate
    std_func : Optional[Callable]
        used in in-accurate predictive models
    alpha : float
        the value used in loss calculation of the inaccurate model
    precision_list : List
        precision for all variables
    """

    def solve(
        self,
        wl_id: str,
        n_probes: int,
        anchor_option: str = "2_step",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Solve MOO by Progressive Frontier

        Parameters:
        ----------
        wl_id : str
            workload id
        n_probes : int
            number of probes
        anchor_option : str
            choice for anchor points calculation

        Returns:
        -------
        Tuple[np.ndarray | None, np.ndarray | None]
            optimal objectives and variables
            None, None if no solution is found
        """

        rectangle_queue: List[Rectangle] = []
        n_objs = len(self.objectives)
        plans = [
            self.get_anchor_point(
                wl_id=wl_id,
                obj_ind=i,
                anchor_option=anchor_option,
            )
            for i in range(n_objs)
        ]
        utopia, nadir = self.get_utopia_and_nadir(plans)
        rectangle = Rectangle(utopia, nadir)
        heapq.heappush(rectangle_queue, rectangle)
        for count in range(n_probes - n_objs):
            if not rectangle_queue:
                logger.info("No more uncertainty space to explore further!")
                break
            rectangle = heapq.heappop(rectangle_queue)
            middle_point, subrectangles = self._find_local_optimum(rectangle, wl_id)
            if middle_point is not None:
                plans.append(middle_point)
            for sub_rect in subrectangles:
                if sub_rect.volume != 0:
                    heapq.heappush(rectangle_queue, sub_rect)

        ## filter dominated points
        po_objs_list = [point.objs.tolist() for point in plans]
        po_vars_list = [
            point.vars.tolist() if point.vars is not None else [] for point in plans
        ]

        po_objs, po_vars = moo_ut.summarize_ret(po_objs_list, po_vars_list)

        return po_objs, po_vars

    def _find_local_optimum(
        self, rectangle: Rectangle, wl_id: str
    ) -> Tuple[Optional[Point], List[Rectangle]]:
        """
        Find the local optimum in the given rectangle and
        the subrectangles in which to continue the search.
        If no optimum is found for the rectangle,
        return None and the upper subrectangles.
        If an optimum is found, return the optimum and
        the subrectangles in which to continue the search.

        Parameters
        ----------
        rectangle : Rectangle
            Rectangle in which to find the local optimum
        wl_id : str | None
            workload id

        Returns
        -------
        Tuple[Point | None, List[Rectangle]]
            The local optimum and the subrectangles in which to continue the search
        """
        current_utopia, current_nadir = Point(rectangle.lower_bounds), Point(
            rectangle.upper_bounds
        )
        middle_objs = np.array(
            [
                current_nadir.objs[i]
                if i == self.opt_obj_ind
                else (current_utopia.objs[i] + current_nadir.objs[i]) / 2
                for i in range(len(self.objectives))
            ]
        )
        middle_point = Point(middle_objs)
        obj_bounds_dict = self._form_obj_bounds_dict(
            current_utopia, middle_point, self.opt_obj_ind
        )
        logger.debug(f"obj_bounds are: {obj_bounds_dict}")
        obj, vars = self.mogd.optimize_constrained_so(
            wl_id=wl_id,
            objective_name=self.objectives[self.opt_obj_ind].name,
            obj_bounds_dict=obj_bounds_dict,
        )

        if obj is not None and vars is not None:
            middle_objs = np.array(
                [
                    obj[i] * objective.direction
                    for i, objective in enumerate(self.objectives)
                ]
            )
            middle_point = Point(middle_objs, vars)
            return middle_point, self.generate_sub_rectangles(
                current_utopia, current_nadir, middle_point
            )

        else:
            logger.debug(
                "This is an empty area \n "
                "don't have pareto points, only "
                "divide current uncertainty space"
            )
            middle_point = Point((current_utopia.objs + current_nadir.objs) / 2)
            rectangles = self.generate_sub_rectangles(
                current_utopia, current_nadir, middle_point, successful=False
            )
            return None, rectangles

    def generate_sub_rectangles(
        self, utopia: Point, nadir: Point, middle: Point, successful: bool = True
    ) -> List[Rectangle]:
        """

        Generate uncertainty space to be explored:
        - if starting from successful optimum as middle, excludes the dominated
        space (middle as utopia and nadir as nadir)
        - if starting from unsuccessful optimum as middle, excludes the space where
        all constraining objectives are lower than the middle point.

        Parameters:
        ----------
        utopia: Point
            the utopia point
        nadir: Point
            the nadir point
        middle: Point
            the middle point generated by
            the constrained single objective optimization
        successful: bool
            whether the middle point is from a successful optimization

        Returns:
        -------
        List[Rectangle]
            sub rectangles to be explored
        """

        rectangles = []
        corner_points = self._get_corner_points(utopia, nadir)
        for point in corner_points:
            # space explored (lower half of constraining objectives)
            is_explored_unconclusive = not successful and np.all(
                middle.objs[1:] - point.objs[1:] > 0
            )
            # nadir point
            is_dominated = successful and np.all(middle.objs - point.objs < 0)
            if is_dominated or is_explored_unconclusive:
                continue
            sub_rect_u, sub_rect_n = self.get_utopia_and_nadir([point, middle])
            rectangles.append(Rectangle(sub_rect_u, sub_rect_n))

        return rectangles

    def _get_corner_points(self, utopia: Point, nadir: Point) -> List[Point]:
        """
        get the corner points that can form a hyper_rectangle
        from utopia and nadir points.

        Parameters:
        ----------
        utopia: Points (defined by class), the utopia point
        nadir: Points (defined by class), the nadir point

        Returns:
        -------
        List[Point]
            2^n_objs corner points
        """
        n_objs = utopia.n_objs
        u_obj_values, n_obj_values = utopia.objs.reshape(
            [n_objs, 1]
        ), nadir.objs.reshape([n_objs, 1])
        grids_list = np.hstack([u_obj_values, n_obj_values])

        ## generate cartesian product of grids_list
        objs_corner_points = np.array([list(i) for i in itertools.product(*grids_list)])
        corner_points = [Point(objs=obj_values) for obj_values in objs_corner_points]

        return corner_points

    def _form_obj_bounds_dict(
        self, utopia: Point, nadir: Point, opt_obj_ind: int
    ) -> dict[str, list]:
        obj_bounds_dict = {}

        for i, objective in enumerate(self.objectives):
            # lower and upper var_ranges per objective
            lower = int(utopia.objs[i])

            if i == opt_obj_ind:
                upper = int(nadir.objs[i])
            else:
                # fixme: to be the same as in Java
                upper = int((utopia.objs[i] + nadir.objs[i]) / 2)

            obj_bounds_dict[objective.name] = [
                solver_ut.get_tensor(lower),
                solver_ut.get_tensor(upper),
            ]

        return obj_bounds_dict

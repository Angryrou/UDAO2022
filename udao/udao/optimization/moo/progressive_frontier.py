import heapq
import itertools
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


class ProgressiveFrontier(BaseMOO):
    def __init__(
        self,
        solver_params: dict,
        variables: Sequence[Variable],
        objectives: Sequence[Objective],
        constraints: Sequence[Constraint],
        accurate: bool,
        std_func: Optional[Callable],
        alpha: float,
        precision_list: List,
    ) -> None:
        """
        initialize parameters in Progressive Frontier method
        :param pf_option: str, internal algorithm ("pf_as" or "pf_ap")
        :param inner_solver: str, solver name
        :param solver_params: dict, parameters used in solver
        :param obj_names: list, objective names
        :param obj_funcs: list, objective functions
        :param opt_type: list, objectives to minimize or maximize
        :param const_funcs: list, constraint functions
        :param const_types: list, constraint types
            ("<=" "==" or ">=", e.g. g1(x1, x2, ...) - c <= 0)
        :param opt_obj_ind: int, the index of the objective
            to be optimized
        :param wl_list: None/list, each element is a string
            to indicate workload id
        :param wl_ranges: function, provided by users, to return upper
            and lower bounds of variables (used in MOGD)
        :param vars_constraints: dict, keys are "conf_min" and
            "conf_max" to indicate the variable range (only used in MOGD)
        :param accurate: bool, to indicate whether the predictive
            model is accurate (True) (used in MOGD)
        :param std_func: function, used in in-accurate predictive models
        """
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

    def solve(
        self,
        wl_id: str,
        n_probes: int,
        anchor_option: str = "2_step",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        solve MOO by Progressive Frontier
        :param wl_id: str, workload id, e.g. '1-7'
        :param accurate: bool, whether the predictive model
            is accurate (True) or not (False), used in MOGD
        :param alpha: float, the value used in loss calculation
            of the inaccurate model
        :param var_bounds: ndarray (n_vars,),
            the lower and upper var_ranges of non-ENUM variables,
            and values of ENUM variables
        :param var_types: list, variable types (float, integer, binary, enum)
        :param precision_list: list, precision for all variables
        :param n_probes: int, the upper bound of number of solutions
        :param n_grids: int, the number of cells set in pf-ap
        :param max_iters: int, the number of iterations in pf-ap
        :return:
                po_objs: ndarray(n_solutions, n_objs), Pareto solutions
                po_vars: ndarray(n_solutions, n_vars),
                    corresponding variables of Pareto solutions
        """

        rectangle_queue: List[Rectangle] = []
        n_objs = len(self.objectives)
        plans = [
            self.get_anchor_points(
                wl_id=wl_id,
                obj_ind=i,
                anchor_option=anchor_option,
            )
            for i in range(n_objs)
        ]
        utopia, nadir = self.get_utopia_and_nadir(plans, n_objs)
        rectangle = Rectangle(utopia, nadir)
        heapq.heappush(rectangle_queue, rectangle)
        count = 0
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
            count += 1

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
        """Find the local optimum in the given rectangle and
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

    def get_anchor_points(
        self,
        wl_id: str,
        obj_ind: int,
        anchor_option: str = "2_step",
    ) -> Point:
        """
        get anchor points
        :param wl_id: str, workload id, e.g. '1-7'
        :param obj_names: list, objective names
        :param obj_ind: int, objective index
        :param accurate:  bool, whether the predictive model
        is accurate (True) or not (False), used in MOGD
        :param alpha: float, the value used in loss calculation of the inaccurate model
        :param var_types: list, variable types (float, integer, binary, enum)
        :param var_bounds: ndarray (n_vars,), the lower and upper var_ranges
        of non-ENUM variables, and values of ENUM variables
        :param precision_list: list, precision for all variables
        :param anchor_option: str, a choice for anchor points calculation
        :param verbose: bool, to indicate whether to print information
        :return:
                objs: ndarray(n_objs,), objective values
                vars: ndarray(1, n_vars), variable values
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

    def get_utopia_and_nadir(
        self, plans: list[Point], n_objs: int
    ) -> Tuple[Point, Point]:
        """
        get the utopia and nadir points
        :param plans: list, each element is a Point (defined class).
        :param n_objs: int, the number of objectives
        :return:
                utopia: Points, utopia point
                nadir: Points, nadir point
        """
        best_objs = [min([point.objs[i] for point in plans]) for i in range(n_objs)]
        worst_objs = [max([point.objs[i] for point in plans]) for i in range(n_objs)]

        utopia = Point(np.array(best_objs))
        nadir = Point(np.array(worst_objs))

        return utopia, nadir

    def generate_sub_rectangles(
        self, utopia: Point, nadir: Point, middle: Point, successful: bool = True
    ) -> List[Rectangle]:
        """
        generate uncertainty space to be explored
        :param utopia: Points (defined by class), the utopia point
        :param nadir: Points (defined by class), the nadir point
        :param middle: Points (defined by class), the middle point
            generated by the constrained single objective optimization
        :param flag: str, to indicate whether
        :return:
                rectangles, list, uncertainty space (Rectangles)
                divided by the middle point
        """
        rectangles = []

        n_objs = utopia.n_objs
        corner_points = self._get_corner_points(utopia, nadir)
        for point in corner_points:
            # space explored (lower half of constraining objectives)
            explored_unconclusive = not successful and np.all(
                middle.objs[1:] - point.objs[1:] > 0
            )
            # nadir point
            dominated = successful and np.all(middle.objs - point.objs < 0)

            if dominated or explored_unconclusive:
                continue
            sub_rect_u, sub_rect_n = self.get_utopia_and_nadir([point, middle], n_objs)
            rectangles.append(Rectangle(sub_rect_u, sub_rect_n))

        return rectangles

    def _get_corner_points(self, utopia: Point, nadir: Point) -> List[Point]:
        """
        get the corner points that can form a hyper_rectangle
        :param utopia: Points (defined by class), the utopia point
        :param nadir: Points (defined by class), the nadir point
        :return:
                corner_points: list, each element is a point
                (Points class) that can form a hyper_rectangle
                ## total 2^n_objs corner points
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
        """
        form the dict used in the constrained optimization
        e.g. the format:
        obj_bounds_dict = {"latency": [solver_ut._get_tensor(0),
                      solver_ut._get_tensor(10000000)],
                      "cores": [solver_ut._get_tensor(0),
                      solver_ut._get_tensor(58)]
                      }
        :param utopia: Points (defined by class), the utopia point
        :param nadir: Points (defined by class), the nadir point
        :param obj_names: list, objective names
        :param opt_obj_ind: int, the index of objective to be optimized
        :return:
                obj_bounds_dict: dict, the dict includes var_ranges of all objectives
        """
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

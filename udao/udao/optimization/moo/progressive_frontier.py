import heapq
import itertools
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ..solver.mogd import MOGD
from ..utils import moo_utils as moo_ut
from ..utils import solver_utils as solver_ut
from ..utils.moo_utils import Points, Rectangles
from ..utils.parameters import VarTypes
from .base_moo import BaseMOO


class ProgressiveFrontier(BaseMOO):
    def __init__(
        self,
        pf_option: str,
        inner_solver: str,
        solver_params: dict,
        obj_names: list,
        obj_funcs: list,
        opt_type: list,
        obj_types: list,
        const_funcs: list,
        const_types: list,
        opt_obj_ind: int,
        wl_list: List[str],
        wl_ranges: Callable,
        vars_constraints: Dict,
        accurate: bool,
        std_func: Callable,
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
        self.pf_option = pf_option
        self.inner_solver = inner_solver
        self.obj_names = obj_names
        self.obj_funcs = obj_funcs
        self.opt_type = opt_type
        self.const_funcs = const_funcs
        self.const_types = const_types
        self.wl_ranges = wl_ranges
        self.obj_types = obj_types
        if self.inner_solver == "mogd":
            self.mogd = MOGD(MOGD.Params(**solver_params))
            self.mogd._problem(
                wl_list,
                wl_ranges,
                vars_constraints,
                accurate,
                std_func,
                obj_funcs,
                obj_names,
                opt_type,
                const_funcs,
                const_types,
            )
        else:
            raise Exception(f"Solver {inner_solver} is not supported!")

        self.opt_obj_ind = opt_obj_ind

    def solve(
        self,
        wl_id: str,
        accurate: bool,
        alpha: float,
        var_bounds: np.ndarray,
        var_types: list,
        precision_list: list,
        n_probes: int,
        n_grids: Optional[int] = None,
        max_iters: Optional[int] = None,
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
        if self.pf_option == "pf_as":
            po_objs, po_vars = self.solve_pf_as(
                wl_id,
                accurate,
                alpha,
                self.obj_names,
                var_bounds,
                self.opt_obj_ind,
                var_types,
                n_probes,
                precision_list,
                anchor_option=anchor_option,
            )
        elif self.pf_option == "pf_ap":
            if n_grids is None or max_iters is None:
                raise Exception("n_grids and max_iters should be provided")
            po_objs, po_vars = self.solve_pf_ap(
                wl_id,
                accurate,
                alpha,
                self.obj_names,
                var_bounds,
                self.opt_obj_ind,
                var_types,
                precision_list,
                n_grids,
                max_iters,
                anchor_option=anchor_option,
            )
        else:
            raise Exception(f"{self.pf_option} is not supported in PF!")

        return po_objs, po_vars

    def solve_pf_as(
        self,
        wl_id: str,
        accurate: bool,
        alpha: float,
        obj_names: List[str],
        var_bounds: np.ndarray,
        opt_obj_ind: int,
        var_types: List,
        n_probes: int,
        precision_list: List,
        anchor_option: str = "2_step",
        verbose: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Progressive Frontier(PF)-Approximation Sequential
        (AS) algorithm, get MOO solutions sequentially
        :param wl_id: str, workload id, e.g. '1-7'
        :param accurate: bool, whether the predictive model
            is accurate (True) or not (False), used in MOGD
        :param alpha: float, the value used in loss
            calculation of the inaccurate model
        :param obj_names: list, objective names
        :param var_bounds: ndarray (n_vars,), the lower and
            upper var_ranges of non-ENUM variables,
            and values of ENUM variables
        :param opt_obj_ind: int, index of objective to be optimized
        :param var_types: list, variable types
            (float, integer, binary, enum)
        :param n_probes: int, the upper bound of number of solutions
        :param precision_list: list, precision for all variables
        :return:
                po_objs: ndarray(n_solutions, n_objs),
                    Pareto solutions
                po_vars: ndarray(n_solutions, n_vars),
                    corresponding variables of Pareto solutions
        """
        # PF-S do not support multi-threading as
        # the solutions are searched within the hyperrectangle with maximum volume,
        # which is generated sequentially

        ## setup the priority queue sorted by hyperrectangle volume
        ## the initial is empty
        pq: List[Rectangles] = []
        ## get initial plans/form a intial hyperrectangle
        plans: List[Points] = []

        n_objs = len(self.opt_type)

        for i in range(n_objs):
            objs, vars = self.get_anchor_points(
                wl_id,
                obj_names,
                i,
                accurate,
                alpha,
                var_types,
                var_bounds,
                precision_list,
                anchor_option=anchor_option,
            )
            if objs is None:
                raise Exception("Cannot find anchor points.")
            plans.append(Points(objs, vars))
        ## compute initial utopia and nadir point
        utopia, nadir = self.get_utopia_and_nadir(plans, n_objs)
        if utopia is None or nadir is None:
            print("Cannot find utopia/nadir points")
            return None, None
        seg = Rectangles(utopia, nadir)
        heapq.heappush(pq, seg)
        count = n_objs
        while count < n_probes:
            if len(pq) == 0:
                print("No more uncertainty space to explore further!")
                break
            seg = heapq.heappop(pq)
            current_utopia, current_nadir = Points(seg.lower_bounds), Points(
                seg.upper_bounds
            )
            middle_objs = (
                np.ones(
                    [
                        n_objs,
                    ]
                )
                * np.inf
            )
            for i in range(n_objs):
                if i == opt_obj_ind:
                    middle_objs[i] = current_nadir.objs[i]
                else:
                    middle_objs[i] = (
                        current_utopia.objs[i] + current_nadir.objs[i]
                    ) / 2
            middle = Points(middle_objs)

            obj_bounds_dict = self._form_obj_bounds_dict(
                current_utopia, middle, obj_names, opt_obj_ind
            )
            if verbose:
                print("obj_bounds are:")
                print(obj_bounds_dict)

            obj, vars = self.mogd.constraint_so_opt(
                wl_id,
                obj=obj_names[opt_obj_ind],
                accurate=accurate,
                alpha=alpha,
                opt_obj_ind=opt_obj_ind,
                var_types=var_types,
                var_range=var_bounds,
                obj_bounds_dict=obj_bounds_dict,
                precision_list=precision_list,
            )

            if (obj is not None) & (vars is not None):
                objs_co = np.ones(
                    [
                        n_objs,
                    ]
                )
                for j in range(n_objs):
                    objs_co[j] = self.obj_funcs[j](vars, wl_id) * moo_ut._get_direction(
                        opt_type=self.opt_type, obj_index=j
                    )
                middle = Points(np.array(objs_co), vars)
                plans.append(middle)
                rectangles = self.generate_sub_rectangles(
                    current_utopia, current_nadir, middle
                )
                for sub_rect in rectangles:
                    if sub_rect.volume == 0:
                        continue
                    else:
                        heapq.heappush(pq, sub_rect)

            else:
                middle = Points((current_utopia.objs + current_nadir.objs) / 2)
                if verbose:
                    print("This is an empty area")
                    print(
                        "don't have pareto points, only "
                        "divide current uncertainty space"
                    )

                rectangles = self.generate_sub_rectangles(
                    current_utopia, current_nadir, middle, flag="bad"
                )

                for i, sub_rect in enumerate(rectangles):
                    # remove the first two sub-rectangles
                    if i == 0 or i == 1:
                        continue
                    else:
                        if sub_rect.volume == 0:
                            continue
                        else:
                            heapq.heappush(pq, sub_rect)

            count += 1

        ## filter dominated points
        po_objs_list = [point.objs.tolist() for point in plans]
        po_vars_list = [
            point.vars.tolist() if point.vars is not None else [] for point in plans
        ]
        sorted_inds = np.argsort(np.array(po_objs_list)[:, 0])
        sorted_po_objs_list = np.array(po_objs_list)[sorted_inds].tolist()
        sorted_po_vars_list = np.array(po_vars_list)[sorted_inds].tolist()

        po_objs, po_vars = moo_ut.summarize_ret(
            sorted_po_objs_list, sorted_po_vars_list
        )

        return po_objs, po_vars

    def solve_pf_ap(
        self,
        wl_id: str,
        accurate: bool,
        alpha: float,
        obj_names: List[str],
        var_bounds: np.ndarray,
        opt_obj_ind: int,
        var_types: List,
        precision_list: List,
        n_grids: int,
        max_iters: int,
        anchor_option: str = "2_step",
        verbose: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Progressive Frontier(PF)-Approximation Parallel (AP) algorithm,
        get MOO solutions parallely
        :param wl_id: str, workload id, e.g. '1-7'
        :param accurate: bool, whether the predictive model is
        accurate (True) or not (False), used in MOGD
        :param alpha: float, the value used in loss
            calculation of the inaccurate model
        :param obj_names: list, objective names
        :param var_bounds: ndarray (n_vars,), the lower and upper
        var_ranges of non-ENUM variables, and values of ENUM variables
        :param opt_obj_ind: int, index of objective to be optimized
        :param var_types: list, variable types (float, integer, binary, enum)
        :param precision_list: list, precision for all variables
        :param n_grids: int, the number of grids per objective
        :param max_iters: int, the number of iterations in pf-ap
        :return:
                po_objs: ndarray(n_solutions, n_objs), Pareto solutions
                po_vars: ndarray(n_solutions, n_vars), corresponding
                variables of Pareto solutions
        """
        # create initial rectangle
        # get initial plans/form a intial hyperrectangle
        plans: List[Points] = []
        n_objs = len(obj_names)

        all_objs_list: List[List] = []
        all_vars_list: List[List] = []
        for i in range(n_objs):
            objs, vars = self.get_anchor_points(
                wl_id,
                obj_names,
                i,
                accurate,
                alpha,
                var_types,
                var_bounds,
                precision_list,
                anchor_option=anchor_option,
            )
            if objs is None:
                raise Exception("Cannot find anchor points.")
            plans.append(Points(objs, vars))
            all_objs_list.append(objs.tolist())
            all_vars_list.append(vars.tolist() if vars is not None else [])

        if verbose:
            # fixme: to be the same as Java PF
            if n_objs == 2:
                opt_obj_ind = 0
            elif n_objs == 3:
                opt_obj_ind = 1
            else:
                raise Exception(f"{n_objs} objectives are not supported for now!" f"")
        iter = 0
        while iter < max_iters:
            if verbose:
                print(f"the number of iteration is {iter}")
            # choose the cell with max volume to explore
            max_volume = -1
            input_ind = -1
            for i in range(len(all_objs_list) - 1):
                current_volume = abs(
                    np.prod(np.array(all_objs_list)[i] - np.array(all_objs_list)[i + 1])
                )
                if current_volume > max_volume:
                    max_volume = current_volume
                    input_ind = i

            plan = [
                Points(objs=np.array(all_objs_list)[input_ind]),
                Points(objs=np.array(all_objs_list)[input_ind + 1]),
            ]
            utopia, nadir = self.get_utopia_and_nadir(plan, n_objs=n_objs)
            if utopia is None or nadir is None:
                print("Cannot find utopia/nadir points")
                return None, None
            # create uniform n_grids ^ (n_objs) grid cells based on the rectangle
            grid_cells_list = self._create_grid_cells(utopia, nadir, n_grids, n_objs)

            obj_bound_cells = []
            for cell in grid_cells_list:
                obj_bound_dict = self._form_obj_bounds_dict(
                    cell.utopia, cell.nadir, obj_names, opt_obj_ind
                )
                obj_bound_cells.append(obj_bound_dict)

            if verbose:
                print("the cells are:")
                print(obj_bound_cells)
            ret_list = self.mogd.constraint_so_parallel(
                wl_id,
                obj=obj_names[opt_obj_ind],
                opt_obj_ind=opt_obj_ind,
                accurate=accurate,
                alpha=alpha,
                var_types=var_types,
                var_ranges=var_bounds,
                cell_list=obj_bound_cells,
                precision_list=precision_list,
            )

            po_objs_list, po_vars_list = [], []
            for solution in ret_list:
                if solution[0] is None:
                    if verbose:
                        print("This is an empty area!")
                    continue
                else:
                    if verbose:
                        print(f"the objective values are:{solution[0]} ")
                    po_objs_list.append(solution[0])
                    if solution[1] is None:
                        raise Exception("Unexpected vars None for objective value.")
                    po_vars_list.append(solution[1].tolist())

            if verbose:
                print("the po_objs_list is: ")
                print(po_objs_list)
                print("the po_vars_list is: ")
                print(po_vars_list)

            all_objs_list.extend(po_objs_list)
            all_vars_list.extend(po_vars_list)

            po_objs = np.array(all_objs_list)
            sorted_inds = np.argsort(po_objs[:, 0])
            sorted_po_objs = np.array(all_objs_list)[sorted_inds].tolist()
            sorted_po_vars = np.array(all_vars_list)[sorted_inds].tolist()

            all_objs, all_vars = moo_ut.summarize_ret(sorted_po_objs, sorted_po_vars)
            all_objs_list = all_objs.tolist() if all_objs is not None else []
            all_vars_list = all_vars.tolist() if all_vars is not None else []
            iter = iter + 1
            if verbose:
                print("the sorted_po_objs is: ")
                print(sorted_po_objs)
                print(f"objs after filtering are: {all_objs_list}")
                print(f"all objs are: {np.array(all_objs_list)}")
                print(f"all_vars are: {np.array(all_vars_list)}")
        return np.array(all_objs_list), np.array(all_vars_list)

    def get_anchor_points(
        self,
        wl_id: str,
        obj_names: List[str],
        obj_ind: int,
        accurate: bool,
        alpha: float,
        var_types: List,
        var_bounds: np.ndarray,
        precision_list: List,
        anchor_option: str = "2_step",
        verbose: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
        obj, vars = self.mogd.single_objective_opt(
            wl_id,
            obj=obj_names[obj_ind],
            accurate=accurate,
            alpha=alpha,
            opt_obj_ind=obj_ind,
            var_types=var_types,
            var_ranges=var_bounds,
            precision_list=precision_list,
        )

        n_objs = len(obj_names)
        # uses conf to get predictions
        objs = (
            np.ones(
                [
                    n_objs,
                ]
            )
            * np.inf
        )
        for j in range(n_objs):
            objs[j] = self.obj_funcs[j](vars, wl_id) * moo_ut._get_direction(
                opt_type=self.opt_type, obj_index=j
            )

        # If the current objective type is Integer,
        # further find the optimal value for other objectives with float type
        if anchor_option == "2_step":
            if self.obj_types[obj_ind] == VarTypes.INTEGER:
                utopia_init = np.zeros(
                    [
                        n_objs,
                    ]
                )
                utopia_init[obj_ind] = objs[obj_ind]
                utopia_tmp, nadir_tmp = Points(objs=utopia_init), Points(objs=objs)
                # select the first objective with float type
                float_obj_ind = [
                    i
                    for i, obj_type in enumerate(self.obj_types)
                    if obj_type == VarTypes.FLOAT
                ][0]
                obj_bounds_dict_so = self._form_obj_bounds_dict(
                    utopia_tmp, nadir_tmp, obj_names, float_obj_ind
                )
                if verbose:
                    print("obj_bounds are:")
                    print(obj_bounds_dict_so)

                objs_update, vars_update = self.mogd.constraint_so_opt(
                    wl_id,
                    obj=obj_names[float_obj_ind],
                    accurate=accurate,
                    alpha=alpha,
                    opt_obj_ind=float_obj_ind,
                    var_types=var_types,
                    var_range=var_bounds,
                    obj_bounds_dict=obj_bounds_dict_so,
                    precision_list=precision_list,
                )
                return np.array(objs_update), vars_update
            else:
                return objs, vars
        elif anchor_option == "1_step":
            return objs, vars
        else:
            raise Exception(f"anchor_option {anchor_option} is not valid!")

    def get_utopia_and_nadir(
        self, plans: list[Points], n_objs: int
    ) -> Tuple[Optional[Points], Optional[Points]]:
        """
        get the utopia and nadir points
        :param plans: list, each element is a Point (defined class).
        :param n_objs: int, the number of objectives
        :return:
                utopia: Points, utopia point
                nadir: Points, nadir point
        """
        ## assume minimization
        best_objs, worst_objs = [], []
        for i in range(n_objs):
            best_obj, worst_obj = np.inf, -np.inf
            for point in plans:
                obj = point.objs[i]
                if obj is None:
                    return None, None

                if obj < best_obj:
                    best_obj = obj

                if obj > worst_obj:
                    worst_obj = obj

            best_objs.append(best_obj)
            worst_objs.append(worst_obj)

        utopia = Points(np.array(best_objs))
        nadir = Points(np.array(worst_objs))

        return utopia, nadir

    def generate_sub_rectangles(
        self, utopia: Points, nadir: Points, middle: Points, flag: str = "good"
    ) -> List[Rectangles]:
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
            if all((middle.objs - point.objs) > 0):
                ## the utopia point
                sub_rect = Rectangles(point, middle)
            elif all((middle.objs - point.objs) < 0):
                ## the nadir point
                ## the rectangle with nadir point will
                # always be dominated by the middle point
                if flag == "good":
                    continue
                else:
                    sub_rect = Rectangles(middle, point)
            else:
                sub_rect_u, sub_rect_n = self.get_utopia_and_nadir(
                    [point, middle], n_objs
                )
                if sub_rect_u is None or sub_rect_n is None:
                    raise Exception("Cannot find utopia/nadir points")
                sub_rect = Rectangles(sub_rect_u, sub_rect_n)

            rectangles.append(sub_rect)

        return rectangles

    def _get_corner_points(self, utopia: Points, nadir: Points) -> List[Points]:
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
        corner_points = [Points(objs=obj_values) for obj_values in objs_corner_points]

        return corner_points

    def _create_grid_cells(
        self, utopia: Points, nadir: Points, n_grids: int, n_objs: int
    ) -> List[Rectangles]:
        """
        create cells used in Progressive Frontier(PF)-Approximation
        Parallel (AP) algorithm
        :param utopia: Points (defined by class), the utopia point
        :param nadir: Points (defined by class), the nadir point
        :param n_grids: int, the number of grids per objective
        :param n_objs: int, the number of objectives
        :return:
                grid_cell_list, each element is a cell (Rectangle class)
        """
        grids_per_var = np.linspace(
            utopia.objs, nadir.objs, num=n_grids + 1, endpoint=True
        )
        objs_list = [grids_per_var[:, i] for i in range(n_objs)]

        ## generate cartesian product of indices for grids
        grids_inds_per_var = np.linspace(0, n_grids - 1, num=n_grids, endpoint=True)
        x = np.tile(grids_inds_per_var, (n_objs, 1))
        grids_inds = np.array([list(i) for i in itertools.product(*x)]).astype(int)

        grid_cell_list = []
        for index, grid_ind in enumerate(grids_inds):
            sub_u_objs = np.array([objs_list[i][id] for i, id in enumerate(grid_ind)])
            sub_u_point = Points(sub_u_objs)
            sub_nadir_objs = np.array(
                [objs_list[i][id + 1] for i, id in enumerate(grid_ind)]
            )
            sub_nadir_point = Points(sub_nadir_objs)
            assert all((sub_nadir_objs - sub_u_objs) >= 0)
            cell = Rectangles(sub_u_point, sub_nadir_point)
            grid_cell_list.append(cell)
        assert len(grid_cell_list) == (n_grids**n_objs)

        return grid_cell_list

    def _form_obj_bounds_dict(
        self, utopia: Points, nadir: Points, obj_names: list, opt_obj_ind: int
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
        n_objs = utopia.objs.shape[0]

        for i in range(n_objs):
            # lower and upper var_ranges per objective
            tmp = [th.tensor(0)] * 2
            obj_key = obj_names[i]

            if self.pf_option == "pf_as":
                if i == opt_obj_ind:
                    tmp[0] = solver_ut.get_tensor(int(utopia.objs[i]))
                    tmp[1] = solver_ut.get_tensor(int(nadir.objs[i]))
                else:
                    # fixme: to be the same as in Java
                    tmp[0] = solver_ut.get_tensor(int(utopia.objs[i]))
                    tmp[1] = solver_ut.get_tensor(
                        int((utopia.objs[i] + nadir.objs[i]) / 2)
                    )
            elif self.pf_option == "pf_ap":
                tmp[0] = solver_ut.get_tensor(int(utopia.objs[i]))
                tmp[1] = solver_ut.get_tensor(int(nadir.objs[i]))
            else:
                raise Exception(f"{self.pf_option} is not supported!")
            obj_bounds_dict[obj_key] = tmp

        return obj_bounds_dict

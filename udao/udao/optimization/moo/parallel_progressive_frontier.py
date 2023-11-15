import itertools
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..concepts import Constraint, Objective, Variable
from ..solver.mogd import MOGD
from ..utils import moo_utils as moo_ut
from ..utils import solver_utils as solver_ut
from ..utils.moo_utils import Point, Rectangle
from ..utils.parameters import VarTypes
from .base_moo import BaseMOO


class ParallelProgressiveFrontier(BaseMOO):
    def __init__(
        self,
        solver_params: dict,
        variables: List[Variable],
        objectives: List[Objective],
        constraints: List[Constraint],
        opt_obj_ind: int,
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

        self.opt_obj_ind = opt_obj_ind

    def solve(
        self,
        wl_id: str,
        n_grids: Optional[int] = None,
        max_iters: Optional[int] = None,
        anchor_option: str = "2_step",
        verbose: bool = False,
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
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

        if n_grids is None or max_iters is None:
            raise Exception("n_grids and max_iters should be provided")
        # create initial rectangle
        # get initial plans/form a intial hyperrectangle
        plans: List[Point] = []
        n_objs = len(self.objectives)

        all_objs_list: List[List] = []
        all_vars_list: List[List] = []
        for i in range(n_objs):
            objs, vars = self.get_anchor_points(
                wl_id=wl_id,
                obj_ind=i,
                anchor_option=anchor_option,
            )
            if objs is None:
                raise Exception("Cannot find anchor points.")
            plans.append(Point(objs, vars))
            all_objs_list.append(objs.tolist())
            all_vars_list.append(vars.tolist() if vars is not None else [])

        if verbose:
            # fixme: to be the same as Java PF
            if n_objs == 2:
                pass
            elif n_objs == 3:
                pass
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
                Point(objs=np.array(all_objs_list)[input_ind]),
                Point(objs=np.array(all_objs_list)[input_ind + 1]),
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
                    cell.utopia, cell.nadir, self.opt_obj_ind
                )
                obj_bound_cells.append(obj_bound_dict)

            if verbose:
                print("the cells are:")
                print(obj_bound_cells)
            ret_list = self.mogd.optimize_constrained_so_parallel(
                wl_id=wl_id,
                objective_name=self.objectives[self.opt_obj_ind].name,
                cell_list=obj_bound_cells,
                batch_size=1,
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
        obj_ind: int,
        anchor_option: str = "2_step",
        verbose: bool = False,
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
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
        obj, vars = self.mogd.optimize_constrained_so(
            wl_id=wl_id,
            objective_name=self.objectives[obj_ind].name,
            obj_bounds_dict=None,
            batch_size=16,
        )

        n_objs = len(self.objectives)
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
            objs[j] = (
                self.objectives[j].function(vars, wl_id) * self.objectives[j].direction
            )

        # If the current objective type is Integer,
        # further find the optimal value for other objectives with float type
        if anchor_option == "2_step":
            if self.objectives[obj_ind].type == VarTypes.INTEGER:
                utopia_init = np.zeros(
                    [
                        n_objs,
                    ]
                )
                utopia_init[obj_ind] = objs[obj_ind]
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
                if verbose:
                    print("obj_bounds are:")
                    print(obj_bounds_dict_so)

                objs_update, vars_update = self.mogd.optimize_constrained_so(
                    wl_id,
                    objective_name=self.objectives[float_obj_ind].name,
                    obj_bounds_dict=obj_bounds_dict_so,
                )
                return np.array(objs_update), vars_update
            else:
                return objs, vars
        elif anchor_option == "1_step":
            return objs, vars
        else:
            raise Exception(f"anchor_option {anchor_option} is not valid!")

    def get_utopia_and_nadir(
        self, plans: list[Point], n_objs: int
    ) -> Tuple[Point | None, Point | None]:
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

        utopia = Point(np.array(best_objs))
        nadir = Point(np.array(worst_objs))

        return utopia, nadir

    def _create_grid_cells(
        self, utopia: Point, nadir: Point, n_grids: int, n_objs: int
    ) -> List[Rectangle]:
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
            sub_u_point = Point(sub_u_objs)
            sub_nadir_objs = np.array(
                [objs_list[i][id + 1] for i, id in enumerate(grid_ind)]
            )
            sub_nadir_point = Point(sub_nadir_objs)
            assert all((sub_nadir_objs - sub_u_objs) >= 0)
            cell = Rectangle(sub_u_point, sub_nadir_point)
            grid_cell_list.append(cell)
        assert len(grid_cell_list) == (n_grids**n_objs)

        return grid_cell_list

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
            obj_bounds_dict[objective.name] = [
                solver_ut.get_tensor(int(utopia.objs[i])),
                solver_ut.get_tensor(int(nadir.objs[i])),
            ]

        return obj_bounds_dict

import itertools
from typing import List, Optional, Tuple

import numpy as np

from ...utils.logging import logger
from ..utils import moo_utils as moo_ut
from ..utils import solver_utils as solver_ut
from ..utils.moo_utils import Point, Rectangle
from .progressive_frontier import AbstractProgressiveFrontier


class ParallelProgressiveFrontier(AbstractProgressiveFrontier):
    def solve(
        self,
        wl_id: str,
        n_grids: int,
        max_iters: int,
        anchor_option: str = "2_step",
        verbose: bool = False,
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

        # create initial rectangle
        # get initial plans/form a intial hyperrectangle
        plans: List[Point] = []
        n_objs = len(self.objectives)

        all_objs_list: List[List] = []
        all_vars_list: List[List] = []
        for i in range(n_objs):
            anchor_point = self.get_anchor_point(
                wl_id=wl_id,
                obj_ind=i,
                anchor_option=anchor_option,
            )
            if anchor_point.vars is None:
                raise Exception("This should not happen.")
            plans.append(anchor_point)
            all_objs_list.append(anchor_point.objs.tolist())
            all_vars_list.append(anchor_point.vars.tolist())

        if n_objs == 2:
            pass
        elif n_objs == 3:
            pass
        else:
            raise Exception(f"{n_objs} objectives are not supported for now!" f"")
        iter = 0
        while iter < max_iters:
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
            utopia, nadir = self.get_utopia_and_nadir(plan)
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

            logger.debug(f"the cells are: {obj_bound_cells}")
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

            logger.debug(f"the po_objs_list is: {po_objs_list}")
            logger.debug(f"the po_vars_list is: {po_vars_list}")
            all_objs_list.extend(po_objs_list)
            all_vars_list.extend(po_vars_list)
            all_objs, all_vars = moo_ut.summarize_ret(all_objs_list, all_vars_list)
            all_objs_list = all_objs.tolist() if all_objs is not None else []
            all_vars_list = all_vars.tolist() if all_vars is not None else []
            iter = iter + 1
        return np.array(all_objs_list), np.array(all_vars_list)

    @staticmethod
    def _create_grid_cells(
        utopia: Point, nadir: Point, n_grids: int, n_objs: int
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
        for grid_ind in grids_inds:
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
        obj_bounds_dict = {}

        for i, objective in enumerate(self.objectives):
            obj_bounds_dict[objective.name] = [
                solver_ut.get_tensor(int(utopia.objs[i])),
                solver_ut.get_tensor(int(nadir.objs[i])),
            ]

        return obj_bounds_dict

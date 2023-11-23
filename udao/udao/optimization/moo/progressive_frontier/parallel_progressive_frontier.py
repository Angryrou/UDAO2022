import itertools
from typing import List, Tuple

import numpy as np

from ....utils.logging import logger
from ...utils import moo_utils as moo_ut
from ...utils.exceptions import NoSolutionError
from ...utils.moo_utils import Point, Rectangle
from .base_progressive_frontier import BaseProgressiveFrontier


class ParallelProgressiveFrontier(BaseProgressiveFrontier):
    def solve(
        self,
        wl_id: str,
        n_grids: int,
        max_iters: int,
        anchor_option: str = "2_step",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        solve MOO by PF-AP (Progressive Frontier - Approximation Parallel)
        Parameters:
        ----------
        wl_id: str
            workload id, e.g. '1-7'
        n_grids: int
            the number of cells set in pf-ap
        max_iters: int
            the number of iterations in pf-ap
        anchor_option: str
            choice for anchor points calculation
        Returns:
        --------
        po_objs: ndarray
            Pareto optimal objective values, of shape
            (n_solutions, n_objs)
        po_vars: ndarray
            corresponding variables of Pareto solutions, of shape
            (n_solutions, n_vars)
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

        if n_objs < 2 or n_objs > 3:
            raise Exception(f"{n_objs} objectives are not supported for now!")

        for i in range(max_iters):
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
                raise NoSolutionError("Cannot find utopia/nadir points")
            # create uniform n_grids ^ (n_objs) grid cells based on the rectangle
            grid_cells_list = self._create_grid_cells(utopia, nadir, n_grids, n_objs)

            obj_bound_cells = []
            for cell in grid_cells_list:
                obj_bound_dict = self._form_obj_bounds_dict(cell.utopia, cell.nadir)
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
                    logger.debug("This is an empty area!")
                    continue
                else:
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

        return np.array(all_objs_list), np.array(all_vars_list)

    @staticmethod
    def _create_grid_cells(
        utopia: Point, nadir: Point, n_grids: int, n_objs: int
    ) -> List[Rectangle]:
        """
        Create cells used in Progressive Frontier(PF)-Approximation
        Parallel (AP) algorithm

        Parameters:
        ----------
        utopia: Point
            the utopia point
        nadir: Point
            the nadir point
        n_grids: int
            the number of grids per objective
        n_objs: int
            the number of objectives

        Returns:
        -------
            List[Rectangle]
            The rectangles in which to perform optimization.
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
        if len(grid_cell_list) != (n_grids**n_objs):
            raise Exception(
                f"Unexpected: the number of grid cells is"
                f"not equal to {n_grids**n_objs}"
            )

        return grid_cell_list

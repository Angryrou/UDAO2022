# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Progressive Frontier Algorithm
#
# Created at 15/09/2022
import heapq
import itertools
from abc import ABCMeta, abstractmethod

from optimization.solver.mogd import MOGD


import numpy as np
import optimization.moo.moo_utils as moo_ut
from optimization.moo.base_moo import BaseMOO


class Node():
    def __init__(self, utopia, nadir, volume):
        self.neg_vol = -volume
        self.utopia = utopia
        self.nadir = nadir

    # Override the `__lt__()` function to make `Node` class work with min-heap
    def __lt__(self, other):
        return self.neg_vol < other.neg_vol

class Rectangles():
    def __init__(self, lower_bounds, upper_bounds):
        ## array: (n_objs * 1)
        self.upper_bounds = upper_bounds.objs
        self.lower_bounds = lower_bounds.objs
        self.n_objs = upper_bounds.objs.shape[0]
        self.volume = self.cal_volume(upper_bounds.objs, lower_bounds.objs)
        self.neg_vol = -self.volume

    def cal_volume(self, upper_bounds, lower_bounds):
        volume = abs(np.prod(upper_bounds - lower_bounds))
        return volume

    # Override the `__lt__()` function to make `Node` class work with min-heap
    def __lt__(self, other):
        return self.neg_vol < other.neg_vol

class Points():
    def __init__(self, objs, params):
        self.objs = objs
        self.params = params
        self.n_objs = objs.shape[0]

class ProgressiveFrontier(BaseMOO):
    def __init__(self, other_params: dict, bounds, debug: bool):
        super().__init__()
        self.option = other_params["pf_option"]
        self.inner_solver = other_params["inner_solver"]
        if self.inner_solver == "mogd":
            self.mogd = MOGD(other_params["mogd_params"], bounds, debug)

        # by default, take the first objective to be optimized, and the others are taken as constraints
        self.opt_obj_ind = 0

    def solve(self, vars, var_types, n_probes, n_objs, n_vars):
        if self.option == "pf_as":
            po_objs, po_vars = self.solve_pf_as(vars, self.opt_obj_ind, var_types, n_probes, n_objs, n_vars)
        elif self.option == "pf_ap":
            # todo
            pass
        else:
            raise ValueError(self.option)

        return po_objs, po_vars

    def solve_pf_as(self, vars, opt_obj_ind, var_types, n_probes, n_objs, n_vars):
        # PF-S do not support multi-threading as the solutions are searched within the hyperrectangle with maximum volume,
        # which is generated sequentially

        ## setup the priority queue sorted by hyperrectangle volume
        ## the initial is empty
        pq = []

        ## get initial plans/form a intial hyperrectangle
        plans = []

        for i in range(n_objs):
            # objs: list, params: array(bs*n_vars)
            obj, params = self.mogd.single_objective_opt(obj=f"obj_{i + 1}", opt_obj_ind=i, var_types=var_types,
                                                         n_vars=n_vars, n_objs=n_objs)
            plans.append(Points(np.array(obj), params))

        ## compute initial utopia and nadir point
        utopia, nadir = self.compute_bounds(plans, n_objs)
        if utopia == None:
            print("Cannot find utopia/nadir points")
            return None, None
        seg = Rectangles(utopia, nadir)
        heapq.heappush(pq, seg)
        count = n_objs

        while count <= n_probes:
            seg = heapq.heappop(pq)
            utopia, nadir = Points(seg.lower_bounds, params=None), Points(seg.upper_bounds, params=None)
            middle = Points((utopia.objs + nadir.objs) / 2, params=None)

            ## by default, take obj_1 as to optimize, and the others (e.g. obj_2, obj_3, ...) as constraints
            obj, params = self.mogd.constraint_so_opt(obj=f"obj_{opt_obj_ind + 1}", opt_obj_ind=opt_obj_ind, var_types=var_types,
                                                      n_vars=n_vars, n_objs=n_objs, lower=utopia.objs, upper=middle.objs,
                                                      bs=16, verbose=False)
            middle = Points(np.array(obj), params)
            plans.append(middle)

            count += 1
            rectangles = self.generate_sub_rectangles(utopia, nadir, middle)
            for sub_rect in rectangles:
                heapq.heappush(pq, sub_rect)

        ## filter dominated points
        po_objs_list = [point.objs for point in plans]
        po_vars_list = [point.params for point in plans]
        po_objs, po_vars = moo_ut._summarize_ret(po_objs_list, po_vars_list)

        return po_objs, po_vars

    def solve_pf_ap(self, n_probes, n_objs, n_grids):
        #todo

        # create initial rectangle
        # get initial plans/form a intial hyperrectangle
        plans = []
        objs = []
        for i in range(n_objs):
            obj, params = self.inner_solver.so(obj=f"obj_{i}")
            objs.append(obj)
            for j in (np.delete(range(n_objs), i)):
                obj_other = self.inner_solver.predict(params, obj=f"obj_{j}")
                objs.append(obj_other)
            plans.append(Points(np.array(objs), params))

        ## compute initial utopia and nadir point
        utopia, nadir = self.compute_bounds(plans, n_objs)

        # create uniform n_grids ^ (n_objs) grid cells based on the initial rectangle
        grid_cells_list = self._create_grid_cells(utopia, nadir, n_grids, n_objs)
        all_plans = []

        ## filter dominated points
        po_objs_cand = [point.objs for point in plans]
        po_vars_cand = [point.params for point in plans]
        po_inds = moo_ut.is_pareto_efficient(po_objs_cand)
        po_objs = po_objs_cand[po_inds]
        po_vars = po_vars_cand[po_inds]

        return po_objs, po_vars

    def compute_bounds(self, plans, n_objs):
        ## assume minimization
        best_objs, worst_objs = [], []
        for i in range(n_objs):
            best_obj, worst_obj = np.inf, -np.inf
            for point in plans:
                obj = point.objs[i]
                if obj == None:
                    return None, None

                if obj < best_obj:
                    best_obj = obj

                if obj > worst_obj:
                    worst_obj = obj

            best_objs.append(best_obj)
            worst_objs.append(worst_obj)

        utopia = Points(np.array(best_objs), params=None)
        nadir = Points(np.array(worst_objs), params=None)

        return utopia, nadir

    def generate_sub_rectangles(self, utopia, nadir, middle):
        rectangles = []

        n_objs = utopia.n_objs
        corner_points = self._get_corner_points(utopia, nadir)
        for point in corner_points:
            if all((middle.objs - point.objs) > 0):
                ## the utopia point
                sub_rect = Rectangles(point, middle)
            elif all((middle.objs - point.objs) < 0):
                ## the nadir point
                ## the rectangle with nadir point will always be dominated by the middle point
                continue
            else:
                sub_rect_u, sub_rect_n = self.compute_bounds([point, middle], n_objs)
                sub_rect = Rectangles(sub_rect_u, sub_rect_n)

            rectangles.append(sub_rect)

        return rectangles

    def _get_corner_points(self, utopia, nadir):
        ## total 2^n_objs corner points

        ## cartesian product of objective values
        ## e.g.
        n_objs = utopia.n_objs
        u_obj_values, n_obj_values = utopia.objs.reshape([n_objs, 1]), nadir.objs.reshape([n_objs, 1])
        grids_list = np.hstack([u_obj_values, n_obj_values])

        ## generate cartesian product of grids_list
        objs_corner_points = np.array([list(i) for i in itertools.product(*grids_list)])
        corner_points = [Points(objs=obj_values, params=None) for obj_values in objs_corner_points]

        return corner_points

    def _create_grid_cells(self, utopia, nadir, n_grids, n_objs):
        # todo: now it supports 2D and 3D, to be generalized to n-dim
        grids_per_var = np.linspace(utopia.objs, nadir.objs, num=n_grids + 1, endpoint=True)

        grid_cell_list = []
        # 2D & 3D
        if grids_per_var.shape[1] == 2:
            obj1_points, obj2_points = grids_per_var[:, 0], grids_per_var[:, 1]
            for i in range(obj1_points.shape[0]):
                for j in range((i + 1), obj1_points.shape[0], 1):
                    sub_u_point = Points(np.array([obj1_points[j], obj2_points[i]]), params=None)
                    sub_nadir_point = Points(np.array([obj1_points[j], obj2_points[j]]), params=None)
                    cell = Rectangles(sub_u_point, sub_nadir_point)
                    grid_cell_list.append(cell)
            assert len(grid_cell_list) == (n_grids ** n_objs)

        elif grids_per_var.shape[1] == 3:
            pass
        else:
            raise NotImplementedError

        return grid_cell_list














# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Weighted Sum
#
# Created at 15/09/2022

from optimization.solver.grid_search import GridSearch
from optimization.solver.random_sampler import RandomSampler
import utils.optimization.moo_utils as moo_ut
from optimization.moo.base_moo import BaseMOO

import numpy as np

class WeightedSum(BaseMOO):

    def __init__(self, ws_pairs, inner_solver, solver_params, n_objs: int, obj_funcs, opt_type, const_funcs):
        super().__init__()
        self.inner_sovler = inner_solver
        self.ws_pairs = ws_pairs
        self.n_objs = n_objs
        self.obj_funcs = obj_funcs
        self.opt_type = opt_type
        self.const_funcs = const_funcs
        if self.inner_sovler == "grid_search":
            self.gs = GridSearch(solver_params)
        elif self.inner_sovler == "random_sampler":
            self.rs = RandomSampler(solver_params)
        else:
            raise ValueError(self.inner_sovler)

    def solve(self, bounds, var_types):
        n_objs = self.n_objs
        if self.inner_sovler == "grid_search":
            vars = self.gs._get_input(bounds, var_types)
        elif self.inner_sovler == "random_sampler":
            vars = self.rs._get_input(bounds, var_types)
        else:
            raise ValueError(self.inner_sovler)

        const_violation = self._get_const_violation(vars)

        # remove vars who lead to constraint violation
        if (const_violation.size != 0) & (const_violation.max() > 0):
            ## find the var index which violate the constraint
            n_const = const_violation.shape[1]
            available_indices = range(const_violation.shape[0])
            for i in range(n_const):
                available_indice = np.where(const_violation[:, i] < 0)
                available_indices = np.intersect1d(available_indice, available_indices)
            vars_after_const_check = vars[available_indices]
        else:
            vars_after_const_check = vars

        if vars_after_const_check.size == 0:
            print("NO feasible solutions found!")
            return None, None
        else:
            po_obj_list, po_var_list = [], []

            # get n_dim objective values
            objs, po_soo_inds = [], []
            for i, obj_func in enumerate(self.obj_funcs):
                obj = obj_func(vars_after_const_check) * self._get_direction(i)
                objs.append(obj)
                # find the index of optimal value for the current obj
                po_soo_ind = np.argmin(obj)
                po_soo_inds.append(po_soo_ind)

            # transform objs to array: (n_samples/grids * n_objs)
            objs = np.array(objs).T

            for i in range(n_objs):
                po_obj_list.append(objs[po_soo_inds[i]])
                po_var_list.append(vars_after_const_check[po_soo_inds[i]])

            # normalization
            objs_min, objs_max = objs.min(0), objs.max(0)
            if all((objs_min - objs_max) < 0):
                objs_norm = (objs - objs_min) / (objs_max - objs_min)
                for ws in self.ws_pairs:
                    po_ind = self.get_soo_index(objs_norm, ws)
                    po_obj_list.append(objs[po_ind])
                    po_var_list.append(vars_after_const_check[po_ind])
                return moo_ut._summarize_ret(po_obj_list, po_var_list)

    def get_soo_index(self, objs, ws_pairs):
        obj = np.sum(objs * ws_pairs, axis=1)
        return np.argmin(obj)

    def _get_const_violation(self, vars):

        g_list = [const_func(vars) for const_func in self.const_funcs]

        # shape (n_samples/grids, n_const)
        return np.array(g_list).T

    def _get_direction(self, obj_index):
        if self.opt_type[obj_index] == "MIN":
            return 1
        else:
            return -1


# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Weighted Sum
#
# Created at 15/09/2022

from optimization.solver.grid_search import GridSearch
from optimization.solver.random_sampler import RandomSampler
import optimization.moo.moo_utils as moo_ut
from optimization.moo.base_moo import BaseMOO

import numpy as np
from abc import ABCMeta, abstractmethod

class WeightedSum(BaseMOO):

    def __init__(self, other_params: dict, n_objs: int, debug: bool):
        super().__init__()
        self.inner_sovler = other_params["inner_solver"]
        self.num_ws_pairs = other_params["num_ws_pairs"]
        self.n_objs = n_objs
        self.ws_pairs = self._get_ws_pairs(n_objs)

        if self.inner_sovler == "grid_search":
            self.gs = GridSearch(other_params["gs_params"], debug)
        elif self.inner_sovler == "random_sample":
            self.rs = RandomSampler(other_params["rs_params"], debug)
        else:
            raise ValueError(self.inner_sovler)

    def solve(self, bounds, var_types, n_objs):
        if self.inner_sovler == "grid_search":
            vars = self.gs._get_input(bounds, var_types)
        elif self.inner_sovler == "random_sample":
            vars = self.rs._get_input(bounds, var_types)
        else:
            raise ValueError(self.inner_sovler)

        const_violation = self._const_function(vars)
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
            for i in range(n_objs):
                obj = self._obj_function(vars_after_const_check, obj=f"obj_{i + 1}")
                objs.append(obj)
                # find the index of optimal value for the current obj
                po_soo_ind = np.argmin(obj)
                po_soo_inds.append(po_soo_ind)

            # transform objs to array: (n_samples * n_objs)
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

    def _get_ws_pairs(self, n_objs):
        ## fixme: not even, to further generate even weights for n-dim
        ## now the weights is generate by:
        # e.g. for 4d
        # w1 = [0, 0.1, ..., 1]
        # w2 = w3 = w4 = [(1-0) / 3, (1-0.1)/3, (1-0.2)/3, ..., (1-1)/3)

        w1 = np.hstack([np.arange(0, 1, 1 / self.num_ws_pairs), 1])
        w = (1 - w1) / (n_objs - 1)
        ws_pairs = [[w1, w2] for w1, w2 in zip(w1, w)]
        return ws_pairs

    @abstractmethod
    def _obj_function(self, vars, obj):
        pass

    @abstractmethod
    def _const_function(self, vars):
        pass


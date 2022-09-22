# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: moo entry point
#
# Created at 21/09/2022

from optimization.moo.weighted_sum import WeightedSum
from optimization.moo.progressive_frontier import ProgressiveFrontier
import utils.optimization.moo_utils as moo_ut

class GenericMOO:

    def __init__(self, moo_algo, solver, obj_names, obj_funcs, opt_type, const_funcs, var_types, var_bounds, add_confs):
        assert moo_algo in ["weighted_sum", "progressive_frontier", "evo", "mobo", "nnc"]
        self.moo_algo = moo_algo
        self.solver = solver
        self.obj_names = obj_names
        self.obj_funcs = obj_funcs
        self.opt_type = opt_type
        self.const_funcs = const_funcs
        self.var_types = var_types
        self.var_bounds = var_bounds
        self.add_confs = add_confs

    def solve(self):
        if self.moo_algo == "weighted_sum":
            ws_steps = self.add_confs[0]
            solver_params = self.add_confs[1]
            n_objs = len(self.opt_type)
            ws_pairs = moo_ut.even_weights(ws_steps, n_objs)
            ws = WeightedSum(ws_pairs, self.solver, solver_params, n_objs, self.obj_funcs, self.opt_type,
                             self.const_funcs)
            po_objs, po_vars = ws.solve(self.var_bounds, self.var_types)
        elif self.moo_algo == 'progressive_frontier':
            pf = ProgressiveFrontier()
            po_objs, po_vars = pf.solve()
        elif self.moo_algo == 'evo':
            pass
        elif self.moo_algo == "mobo":
            pass
        elif self.moo_algo == "nnc":
            pass
        else:
            raise NotImplementedError

        return po_objs, po_vars
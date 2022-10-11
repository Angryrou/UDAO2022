# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: moo entry point
#
# Created at 21/09/2022

from optimization.moo.weighted_sum import WeightedSum
# from optimization.moo.progressive_frontier import ProgressiveFrontier
# from optimization.moo.evolutionary import EVO
import utils.optimization.moo_utils as moo_ut

class GenericMOO:

    def __init__(self):
        pass

    def problem_setup(self, obj_names: list, obj_funcs: list, opt_types: list, const_funcs: list, const_types: list,
                      var_types: list, var_ranges: list):
        '''
        setup common input paramters for MOO problems
        :param obj_names: list, objective names
        :param obj_funcs: list, objective functions
        :param opt_types: list, objectives to minimize or maximize
        :param const_funcs: list, constraint functions
        :param const_types: list, constraint types ("<=" "==" or ">=", e.g. g1(x1, x2, ...) - c <= 0)
        :param var_types: list, variable types (float, integer, binary, enum)
        :param var_ranges: ndarray(n_vars, ), lower and upper var_ranges of variables(non-ENUM), and values of ENUM variables
        :return:
        '''
        self.obj_names = obj_names
        self.obj_funcs = obj_funcs
        self.opt_types = opt_types
        self.const_funcs = const_funcs
        self.const_types = const_types
        self.var_types = var_types
        self.var_ranges = var_ranges

    def solve(self, moo_algo: str, solver: str, add_params: list):
        '''
        solve MOO problems internally by different MOO algorithms
        :param moo_algo: str, the name of moo algorithm
        :param solver: str, the name of solver
        :param add_params: list, the parameters required by the specified MOO algorithm and solver
        :return: po_objs: ndarray(n_solutions, n_objs), Pareto solutions
                 po_vars: ndarray(n_solutions, n_vars), corresponding variables of Pareto solutions
        '''
        if moo_algo == "weighted_sum":
            ws_steps = add_params[0]
            solver_params = add_params[1]
            n_objs = len(self.opt_types)
            ws_pairs = moo_ut.even_weights(ws_steps, n_objs)
            ws = WeightedSum(ws_pairs, solver, solver_params, n_objs, self.obj_funcs, self.opt_types,
                             self.const_funcs, self.const_types)
            po_objs, po_vars = ws.solve(self.var_ranges, self.var_types)
        elif moo_algo == 'progressive_frontier':
            raise NotImplementedError
        elif moo_algo == 'evolutionary':
            raise NotImplementedError
        elif moo_algo == "mobo":
            raise NotImplementedError
        elif moo_algo == "normalized_normal_constraint":
            raise NotImplementedError
        else:
            raise NotImplementedError

        return po_objs, po_vars
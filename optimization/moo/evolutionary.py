# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Evolutionary Algorithm
#
# Created at 22/09/2022

import numpy as np
from optimization.moo.base_moo import BaseMOO
import random

from platypus import NSGAII, Problem, Real, Integer, nondominated, Archive

class EVO(BaseMOO):

    def __init__(self, inner_algo, obj_funcs, opt_type, const_funcs, const_types, pop_size, nfe, fix_randomness_flag):
        super().__init__()
        self.n_objs = len(obj_funcs)
        self.n_consts = len(const_funcs)
        self.obj_funcs = obj_funcs
        self.const_funcs = const_funcs
        self.const_types = const_types
        self.opt_type = opt_type
        self.inner_algo = inner_algo

        self.pop_size = pop_size
        self.nfe = nfe
        self.fix_randomness_flag = fix_randomness_flag
        if inner_algo == "NSGA-II":
            self.moo = NSGAII

    def solve(self, var_bounds, var_types):

        class LoggingArchive(Archive):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, *kwargs)
                self.log = []

            def add(self, solution):
                super().add(solution)
                self.log.append(solution)

        log_archive = LoggingArchive()

        n_vars = var_bounds.shape[1]
        n_objs = self.n_objs
        n_consts = self.n_consts

        problem = Problem(n_vars, n_objs, n_consts)  # n_vars, n_objs, n_constraints

        # find list of indices for different variable types
        float_inds = [i for i, x in enumerate(var_types) if x == "FLOAT"]
        int_inds = [i for i, x in enumerate(var_types) if x == "INTEGER"]
        binary_inds = [i for i, x in enumerate(var_types) if x == "BINARY"]

        if len(float_inds) > 0:
            for i in float_inds:
                problem.types[i] = Real(var_bounds[i, 0], var_bounds[i, 1])

        if len(int_inds) > 0:
            for i in int_inds:
                problem.types[i] = Integer(var_bounds[i, 0], var_bounds[i, 1])

        if len(binary_inds) > 0:
            for i in binary_inds:
                problem.types[i] = Integer(0, 1)

        if len(float_inds) + len(int_inds) + len(binary_inds) == 0:
            print("ERROR: No feasilbe variables provided, please check the variable types setting!")

        assert len(float_inds) + len(int_inds) + len(binary_inds) == n_vars

        # indices of constraint types
        le_inds = [i for i, x in enumerate(self.const_types) if x == "<="]
        l_inds = [i for i, x in enumerate(self.const_types) if x == "<"]

        if len(le_inds) > 0:
            for i in le_inds:
                problem.constraints[i] = "<=0"

        if len(l_inds) > 0:
            for i in l_inds:
                problem.constraints[i] = "<0"

        if len(le_inds) + len(l_inds) == 0:
            print("ERROR: No feasilbe constraints provided, please check the constraint types setting!")

        problem.function = self.toy_example

        # fix randomness
        if self.fix_randomness_flag:
            random.seed(0)

        algorithm = self.moo(problem, population_size=self.pop_size, archive=log_archive)
        algorithm.run(self.nfe)

        feasible_solutions = [s for s in algorithm.result if s.feasible]
        if len(feasible_solutions) == 0:
            feasible_solutions = [s for s in log_archive.log if s.feasible]

        if feasible_solutions == []:
            print(f'Evo({self.inner_algo}) cannot find feasible solutions!')
            po_objs_list, po_vars_list = None, None
            return po_objs_list, po_vars_list
        else:
            non_dominated = nondominated(feasible_solutions)
            non_dominated_objs = [solution.objectives._data for solution in non_dominated]
            uniq_non_dominated_objs, uniq_non_dominated_index = np.unique(np.array(non_dominated_objs), axis=0,
                                                                          return_index=True)
            uniq_non_dominated = np.array(non_dominated)[uniq_non_dominated_index].tolist()

            print(f'the number of non-dominated solutions is {len(uniq_non_dominated_objs)}')

            po_objs_list, po_vars_list = [], []
            for solution in uniq_non_dominated:

                po_vars= [x.decode(y) for [x, y] in zip(problem.types, solution.variables)]
                po_objs_list.append(solution.objectives._data)
                po_vars_list.append(po_vars)

            return np.array(po_objs_list), np.array(po_vars_list)

    def toy_example(self, vars):
        # defined_functions need the input variables to be array with shape([n, n_vars]), where n can be any number
        vars = np.array(vars)
        vars = vars.reshape([1, vars.shape[0]])

        # the formats of f_list(and g_list) is required as list[value1, value2, ...]
        f_list = [obj_func(vars).tolist()[0] for obj_func in self.obj_funcs]
        g_list = [const_func(vars).tolist()[0] for const_func in self.const_funcs]

        return f_list, g_list


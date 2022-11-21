# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#            Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: pre-define functions of Heuristic Closed Form (HCF).
#
# Created at 10/12/22
import numpy as np
import torch as th

from optimization.model.base_model import BaseModel

# class HCF:
#     """
#     An example
#     https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization_problems
#     Binh and Korn function:
#     # minimize:
#     #          f1(x1, x2) = 4 * x_1 * x_1 + 4 * x_2 * x_2
#     #          f2(x1, x2) = (x_1 - 5) * (x_1 - 5) + (x_2 - 5) * (x_2 - 5)
#     # subject to:
#     #          g1(x_1, x_2) = (x_1 - 5) * (x_1 - 5) + x_2 * x_2 <= 25
#     #          g2(x_1, x_2) = (x_1 - 8) * (x_1 - 8) + (x_2 + 3) * (x_2 + 3) >= 7.7
#     #          x_1 in [0, 5], x_2 in [0, 3]
#     """
#
#     @staticmethod
#     def obj_func1(vars):
#         '''
#         :param vars: array
#         :return:
#         '''
#         value = 4 * vars[:, 0] * vars[:, 0] + 4 * vars[:, 1] * vars[:, 1]
#         return value
#
#     @staticmethod
#     def obj_func2(vars):
#         value = (vars[:, 0] - 5) * (vars[:, 0] - 5) + (vars[:, 1] - 5) * (vars[:, 1] - 5)
#         return value
#
#     @staticmethod
#     def const_func1(vars):
#         value = (vars[:, 0] - 5) * (vars[:, 0] - 5) + vars[:, 1] * vars[:, 1] - 25
#         return value
#
#     @staticmethod
#     def const_func2(vars):
#         value = (vars[:, 0] - 8) * (vars[:, 0] - 8) + (vars[:, 1] + 3) * (vars[:, 1] + 3) - 7.7
#         return value

class HCF_3D:
    """
        An example
        https://pymoo.org/problems/many/dtlz.html#DTLZ1
        DTLZ1 function with 3 objectives and 3 variables:
        # minimize:
        #          f1(x_1, x_2, x_3) = 1/2 * x_1 * x_2 * (1 + g(x_3))
        #          f2(x_1, x_2, x_3) = 1/2 * x_1 * (1 - x_2) * (1 + g(x_3))
        #          f2(x_1, x_2, x_3) = 1/2 * (1 - x_1) * (1 + g(x_3))
        # where g(x_3) = 100 * (1 + (x_3 - 0.5)^2 - cos(20 * pi * (x_3 - 0.5)))
        # subject to:
        #          x_i in [0, 1], i = 1, 2, 3
        """

    @staticmethod
    def obj_func1(vars):
        '''
        :param vars: array
        :return:
        '''
        g = 100 * (1 + (vars[:, 2] - 0.5) * (vars[:, 2] - 0.5) - np.cos(20 * np.pi * (vars[:, 2] - 0.5)))
        value = 0.5 * vars[:, 0] * vars[:, 1] * (1 + g)
        return value

    @staticmethod
    def obj_func2(vars):
        g = 100 * (1 + (vars[:, 2] - 0.5) * (vars[:, 2] - 0.5) - np.cos(20 * np.pi * (vars[:, 2] - 0.5)))
        value = 0.5 * vars[:, 0] * ( 1 - vars[:, 1]) * (1 + g)
        return value

    @staticmethod
    def obj_func3(vars):
        g = 100 * (1 + (vars[:, 2] - 0.5) * (vars[:, 2] - 0.5) - np.cos(20 * np.pi * (vars[:, 2] - 0.5)))
        value = 0.5 * (1 - vars[:, 0]) * (1 + g)
        return value

class HCF(BaseModel):
    def __init__(self, obj_names: list, var_ranges: list):
        '''
        :param objs: list, name of objectives
        :param training_vars: ndarray(n_training_samples, n_vars), input to train GPR models
        '''
        super().__init__(obj_names)
        self.initialize(var_ranges)

    def initialize(self, var_ranges):
        '''
        initialize parameters and fit GPR models for objectives separately
        :param objs: list, name of objectives
        :param training_vars: ndarray(n_training_samples, n_vars), input to train GPR models
        :return:
        '''
        self.var_ranges = var_ranges

    def fit(self):
        pass

    def normalize_config(self, config):
        """
        :param *args:
        :param real_conf: numpy.array[int]
        :return: normalized to 0-1
        """
        var_min, var_max = self.get_conf_range_for_wl()
        normalized_conf = (config - var_min) / (var_max - var_min)
        return normalized_conf

    def get_conf_range_for_wl(self, wl_id=None):
        conf_max = self.var_ranges[:, 1]
        conf_min = self.var_ranges[:, 0]
        return conf_max, conf_min

    def internal_prediction(self, name, config_norm, *args):
        # transfrom x in the original space as (
        x1_min, x2_min = self.var_ranges[0, 0], self.var_ranges[1, 0]
        x1_range, x2_range = (self.var_ranges[0, 1] - x1_min), (self.var_ranges[1, 1] - x2_min)
        x_1 = config_norm[:, 0] * x1_range + x1_min
        x_2 = config_norm[:, 1] * x2_range + x2_min
        if name == "obj_1":
            value = 4 * x_1 * x_1 + 4 * x_2 * x_2
        elif name == "obj_2":
            value = (x_1 - 5) * (x_1 - 5) + (x_2 - 5) * (x_2 - 5)
        elif name == "g1":
            value = (x_1 - 5) * (x_1 - 5) + x_2 * x_2 - 25
        elif name == "g2":
            value = (x_1 - 8) * (x_1 - 8) + (x_2 + 3) * (x_2 + 3) - 7.7
        else:
            raise Exception(f"Objective/constraint {name} is not valid for prediction!")

        return value

class HCF_functions:

    def __init__(self, objs: list, var_ranges):
        '''
        initialization
        :param objs: list, name of objectives
        :param training_vars: ndarray(n_samples, n_vars), input used to train GPR model.
        '''
        self.hcf = HCF(objs, var_ranges)

    def get_vars_range_for_wl(self, wl_id=None):
        return self.hcf.get_conf_range_for_wl(wl_id)

    def predict_obj1(self, vars, wl_id=None):
        obj = "obj_1"
        # value = self.hcf.predict(obj, vars).reshape(-1,1)
        if not th.is_tensor(vars):
            value = self.hcf.predict(obj, vars)
        else:
            value = self.hcf.internal_prediction(obj, vars).reshape(-1,1)
        return value

    def predict_obj2(self, vars, wl_id=None):
        obj = "obj_2"
        # value = self.hcf.predict(obj, vars).reshape(-1,1)
        if not th.is_tensor(vars):
            value = self.hcf.predict(obj, vars)
        else:
            value = self.hcf.internal_prediction(obj, vars).reshape(-1,1)
        return value

    # only used for 2D example
    def const_func1(self, vars, wl_id=None):
        const = "g1"
        # value = self.hcf.predict(const, vars).reshape(-1,1)
        if not th.is_tensor(vars):
            value = self.hcf.predict(const, vars)
        else:
            value = self.hcf.internal_prediction(const, vars).reshape(-1,1)
        return value

    def const_func2(self, vars, wl_id=None):
        const = "g2"
        # value = self.hcf.predict(const, vars).reshape(-1,1)
        if not th.is_tensor(vars):
            value = self.hcf.predict(const, vars)
        else:
            value = self.hcf.internal_prediction(const, vars).reshape(-1,1)
        return value

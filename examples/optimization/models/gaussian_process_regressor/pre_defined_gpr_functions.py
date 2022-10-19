# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: pre-define functions based on Gaussian Process Regressor (GPR) model.
#
# Created at 17/10/2022

from examples.optimization.models.gaussian_process_regressor.gpr import GPR

class GPRPredictiveModels:

    def __init__(self, objs: list, training_vars):
        '''
        initialization
        :param objs: list, name of objectives
        :param training_vars: ndarray(n_samples, n_vars), input used to train GPR model.
        '''
        self.training_vars = training_vars
        self.gpr = GPR(objs, training_vars)

    def predict_obj1(self, vars):
        obj = "obj_1"
        value = self.gpr.predict(obj, vars)
        return value

    def predict_obj2(self, vars):
        obj = "obj_2"
        value = self.gpr.predict(obj, vars)
        return value

    @staticmethod
    def const_func1(vars):
        value = (vars[:, 0] - 5) * (vars[:, 0] - 5) + vars[:, 1] * vars[:, 1] - 25
        return value

    @staticmethod
    def const_func2(vars):
        value = (vars[:, 0] - 8) * (vars[:, 0] - 8) + (vars[:, 1] + 3) * (vars[:, 1] + 3) - 7.7
        return value

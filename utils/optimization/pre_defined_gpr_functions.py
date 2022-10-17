# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: pre-define functions based on Gaussian Process Regressor (GPR) model.
#
# Created at 17/10/2022

from examples.optimization.ws.predictive_model.gpr.gpr import GPR

class GPRPredictiveModels:
    # initialize model
    gpr = GPR()

    @classmethod
    def predict_obj1(cls, vars):
        obj = "obj_1"
        value = cls.gpr.predict(obj, vars)
        return value

    @classmethod
    def predict_obj2(cls, vars):
        obj = "obj_2"
        value = cls.gpr.predict(obj, vars)
        return value

    @staticmethod
    def const_func1(vars):
        value = (vars[:, 0] - 5) * (vars[:, 0] - 5) + vars[:, 1] * vars[:, 1] - 25
        return value

    @staticmethod
    def const_func2(vars):
        value = (vars[:, 0] - 8) * (vars[:, 0] - 8) + (vars[:, 1] + 3) * (vars[:, 1] + 3) - 7.7
        return value

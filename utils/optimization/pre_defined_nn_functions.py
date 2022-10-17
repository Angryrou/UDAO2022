# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: pre-define functions based on Neural Network (NN) model
#
# Created at 17/10/2022

from examples.optimization.ws.predictive_model.nn.nn import NN

class NNPredictiveModels:
    # initialize model
    nn = NN()

    @classmethod
    def predict_obj1(cls, vars):
        obj = "obj_1"
        value = cls.nn.predict(obj, vars)
        return value

    @classmethod
    def predict_obj2(cls, vars):
        obj = "obj_2"
        value = (vars[:, 0] - 5) * (vars[:, 0] - 5) + (vars[:, 1] - 5) * (vars[:, 1] - 5)
        return value

    # constraints support left hand side (LHS) e.g. g(x1, x2, ...) - C <= 0
    @classmethod
    def const_func1(cls, vars):
        C = 10
        const = "const_1"
        value = cls.nn.predict(const, vars) - C
        return value

    @staticmethod
    def const_func2(vars):
        value = (vars[:, 0] - 8) * (vars[:, 0] - 8) + (vars[:, 1] + 3) * (vars[:, 1] + 3) - 7.7
        return value
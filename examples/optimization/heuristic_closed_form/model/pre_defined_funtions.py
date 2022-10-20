# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#            Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: pre-define functions of Heuristic Closed Form (HCF).
#
# Created at 10/12/22

class HCF:
    """
    An example
    https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization_problems
    Binh and Korn function:
    # minimize:
    #          f1(x1, x2) = 4 * x_1 * x_1 + 4 * x_2 * x_2
    #          f2(x1, x2) = (x_1 - 5) * (x_1 - 5) + (x_2 - 5) * (x_2 - 5)
    # subject to:
    #          g1(x_1, x_2) = (x_1 - 5) * (x_1 - 5) + x_2 * x_2 <= 25
    #          g2(x_1, x_2) = (x_1 - 8) * (x_1 - 8) + (x_2 + 3) * (x_2 + 3) >= 7.7
    #          x_1 in [0, 5], x_2 in [0, 3]
    """

    @staticmethod
    def obj_func1(vars):
        '''
        :param vars: array
        :return:
        '''
        value = 4 * vars[:, 0] * vars[:, 0] + 4 * vars[:, 1] * vars[:, 1]
        return value

    @staticmethod
    def obj_func2(vars):
        value = (vars[:, 0] - 5) * (vars[:, 0] - 5) + (vars[:, 1] - 5) * (vars[:, 1] - 5)
        return value

    @staticmethod
    def const_func1(vars):
        value = (vars[:, 0] - 5) * (vars[:, 0] - 5) + vars[:, 1] * vars[:, 1] - 25
        return value

    @staticmethod
    def const_func2(vars):
        value = (vars[:, 0] - 8) * (vars[:, 0] - 8) + (vars[:, 1] + 3) * (vars[:, 1] + 3) - 7.7
        return value

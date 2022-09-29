# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Place to define functions for both obejctives and constraints
#
# Created at 21/09/2022

# An example
# https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization_problems
# Chankong and Haimes function:
## minimize:
##          f1(x, y) = 2 + (x - 2) * (x - 2) + (y - 1) * (y - 1)
##          f2(x, y) = 9 * x - (y - 1) * (y - 1)
## subject to:
##          g1(x, y) = x * x + y * y <= 225
##          g2(x, y) = x - 3 * y + 10 <= 0
##          x in [-20, 20], y in [-20, 20]

def obj_func1(vars):
    '''

    :param vars: array:
    :return:
    '''
    value = 2 + (vars[:, 0] - 2) * (vars[:, 0] - 2) + (vars[:, 1] - 1) * (vars[:, 1] - 1)
    return value

def obj_func2(vars):
    value = 9 * vars[:, 0] - (vars[:, 1] - 1) * (vars[:, 1] - 1)
    return value

# assume g(x1, x2, ...) <= c
def const_func1(vars):
    value = vars[:, 0] * vars[:, 0] + vars[:, 1] * vars[:, 1] - 225
    return value

def const_func2(vars):
    value = vars[:, 0] - 3 * vars[:, 1] + 10 - 0
    return value


# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: an example on Weighted Sum with Random-Sampler solver
#
# Created at 15/09/2022

# from model.architecture.heuristic_closed_form import HeuristicClosedForm
from optimization.moo.weighted_sum import WeightedSum

import numpy as np
from matplotlib import pyplot as plt

class ExampleWSWithGridSearch(WeightedSum):
    def __init__(self, gs_params, n_objs, debug):
        super().__init__(gs_params, n_objs, debug)

    def _obj_function(self, vars, obj):
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization_problems
        # Chankong and Haimes function:
        ## minimize:
        ##          f1(x, y) = 2 + (x - 2) * (x - 2) + (y - 1) * (y - 1)
        ##          f2(x, y) = 9 * x - (y - 1) * (y - 1)
        ## subject to:
        ##          g1(x, y) = x * x + y * y <= 225
        ##          g2(x, y) = x - 3 * y + 10 <= 0
        ##          x in [-20, inf], y in [-inf, 20]
        f_list = []
        if obj == "obj_1":
            value = 2 + (vars[:, 0] - 2) * (vars[:, 0] - 2) + (vars[:, 1] - 1) * (vars[:, 1] - 1)
            f_list.extend(value)
        elif obj == "obj_2":
            value = 9 * vars[:, 0] - (vars[:, 1] - 1) * (vars[:, 1] - 1)
            f_list.extend(value)
        else:
            raise ValueError(obj)
        return np.array(f_list)

    def _const_function(self, vars):
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization_problems
        # Chankong and Haimes function:
        ## minimize:
        ##          f1(x, y) = 2 + (x - 2) * (x - 2) + (y - 1) * (y - 1)
        ##          f2(x, y) = 9 * x - (y - 1) * (y - 1)
        ## subject to:
        ##          g1(x, y) = x * x + y * y <= 225
        ##          g2(x, y) = x - 3 * y + 10 <= 0
        ##          x in [-20, inf], y in [-inf, 20]

        ## add constraints
        # each g1 value shows the constraint violation
        g1 = vars[:, 0] * vars[:, 0] + vars[:, 1] * vars[:, 1] - 225
        g2 = vars[:, 0] - 3 * vars[:, 1] + 10 - 0

        ## return array type
        return np.hstack([g1.reshape([g1.shape[0], 1]), g2.reshape([g2.shape[0], 1])])

    def plot_po(self, po):
        # po: ndarray (n_solutions * n_objs)
        ## for 2D
        po_obj1 = po[:, 0]
        po_obj2 = po[:, 1]

        fig, ax = plt.subplots()
        ax.scatter(po_obj1, po_obj2, marker='o', color="blue")

        ax.set_xlabel('Obj 1')
        ax.set_ylabel('Obj 2')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    ## Problem settings
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization_problems
    # Chankong and Haimes function:
    ## minimize:
    ##          f1(x, y) = 2 + (x - 2) * (x - 2) + (y - 1) * (y - 1)
    ##          f2(x, y) = 9 * x - (y - 1) * (y - 1)
    ## subject to:
    ##          g1(x, y) = x * x + y * y <= 225
    ##          g2(x, y) = x - 3 * y + 10 <= 0
    ##          x in [-20, inf], y in [-inf, 20]

    ## for variables
    n_vars = 2
    lower = np.array([[-20], [-np.inf]])
    upper = np.array([[np.inf], [20]])
    bounds = np.hstack([lower, upper])
    var_types = ["float", "float"]
    other_params = {}

    ## for random_sample
    other_params["inner_solver"] = "random_sample"
    other_params["num_ws_pairs"] = 10
    rs_params = {}
    rs_params["n_samples_per_param"] = 10000
    other_params["rs_params"] = rs_params
    example = ExampleWSWithGridSearch(other_params, n_objs=2, debug=False)
    po_objs, po_vars = example.solve(bounds, var_types, n_objs=2)
    print(po_objs)
    print(po_vars)
    if po_objs is not None:
        example.plot_po(po_objs)

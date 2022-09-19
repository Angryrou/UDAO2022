# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: TODO
#
# Created at 9/14/22

from optimization.moo.progressive_frontier import ProgressiveFrontier

import numpy as np
from matplotlib import pyplot as plt

class ExamplePFWithMOGD(ProgressiveFrontier):
    def __init__(self, other_params, bounds, n_objs, debug):
        super().__init__(other_params, bounds, debug)

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
    other_params["pf_option"] = "pf_as"
    other_params["inner_solver"] = "mogd"

    mogd_params = {}
    mogd_params["lr"], mogd_params["wd"], mogd_params["max_iter"] = 0.01, 0.1, 500
    mogd_params["patient"], mogd_params["multistart"], mogd_params["processes"] = 20, 1, 0
    other_params["mogd_params"] = mogd_params

    example = ExamplePFWithMOGD(other_params, bounds, n_objs=2, debug=False)
    po_objs, po_vars = example.solve(vars, var_types, n_probes=15, n_objs=2, n_vars = n_vars)
    print(po_objs)
    print(po_vars)
    if po_objs is not None:
        example.plot_po(po_objs)


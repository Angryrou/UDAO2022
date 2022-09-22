# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example on running Weighted Sum with Grid-Search solver
#
# Created at 22/09/2022

from optimization.moo.generic_moo import GenericMOO
from utils.optimization.configs_parser import ConfigsParser
import utils.optimization.functions_def as func_def

from matplotlib import pyplot as plt

HELP = """
Format: python ws_grid_hcf.py -c <config> -h
    - c : The configuration file location. Default is "examples/optimization/ws_grid/heuristic_closed_form/configs.json"
Example:
    python examples/optimization/ws_grid/heuristic_closed_form/ws_grid_hcf.py -c examples/optimization/ws_grid/heuristic_closed_form/configs.json
"""

moo_algo, solver, var_types, var_bounds, obj_names, opt_types, add_params = ConfigsParser().parse_details()

moo = GenericMOO(moo_algo, solver, obj_names, obj_funcs=[func_def.obj_func1, func_def.obj_func2], opt_type=opt_types,
                 const_funcs=[func_def.const_func1, func_def.const_func2], var_types=var_types, var_bounds=var_bounds,
                 add_confs=add_params)

po_objs, po_vars = moo.solve()
print(po_objs)
print(po_vars)

def plot_po(po):
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

plot_po(po_objs)

# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example on running Weighted Sum with Grid-Search solver
#
# Created at 22/09/2022

from optimization.moo.generic_moo import GenericMOO
from utils.optimization.configs_parser import ConfigsParser
from utils.optimization.pre_defined_funtions import HCF
import utils.optimization.moo_utils as moo_ut

HELP = """
Format: python ws_hcf.py -c <config> -h
    - c : The configuration file location. Default is "examples/optimization/ws/heuristic_closed_form/hcf_configs_grid_search.json"
Example:
    python examples/optimization/ws/heuristic_closed_form/ws_hcf.py -c examples/optimization/ws/heuristic_closed_form/hcf_configs_grid_search.json
"""

# get input parameters
moo_algo, solver, var_types, var_ranges, obj_names, opt_types, const_types, add_params = ConfigsParser().parse_details()
# problem setup
moo = GenericMOO()
moo.problem_setup(obj_names=obj_names, obj_funcs=[HCF.obj_func1, HCF.obj_func2], opt_types=opt_types,
                  const_funcs=[HCF.const_func1, HCF.const_func2], const_types=const_types,
                  var_types=var_types, var_ranges=var_ranges)
# solve MOO problem
po_objs, po_vars = moo.solve(moo_algo, solver, add_params)

print("Pareto solutions:")
print(po_objs)
print("Variables:")
print(po_vars)

if po_objs is not None:
    moo_ut.plot_po(po_objs, n_obj=po_objs.shape[1])

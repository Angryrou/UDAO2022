# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example on running Weighted Sum with Grid-Search solver
#
# Created at 22/09/2022

from optimization.moo.generic_moo import GenericMOO
from utils.optimization.configs_parser import ConfigsParser
from utils.optimization.pre_defined_nn_functions import NNPredictiveModels
import utils.optimization.moo_utils as moo_ut

HELP = """
Format: python nn_main.py -c <config> -h
    - c : The configuration file location. Default is "examples/optimization/ws/predictive_model/nn/nn_configs_grid_search.json"
Example:
    python examples/optimization/ws/predictive_model/nn/nn_main.py -c examples/optimization/ws/predictive_model/nn/nn_configs_grid_search.json
"""

# get input parameters
moo_algo, solver, var_types, var_ranges, obj_names, opt_types, const_types, add_params = ConfigsParser().parse_details()

# model set up
predictive_model = NNPredictiveModels()

# problem setup
moo = GenericMOO()
moo.problem_setup(obj_names=obj_names, obj_funcs=[predictive_model.predict_obj1, predictive_model.predict_obj2], opt_types=opt_types,
                  const_funcs=[predictive_model.const_func1, predictive_model.const_func2], const_types=const_types,
                  var_types=var_types, var_ranges=var_ranges)
# solve MOO problem
po_objs, po_vars = moo.solve(moo_algo, solver, add_params)

print("Pareto solutions:")
print(po_objs)
print("Variables:")
print(po_vars)

if po_objs is not None:
    moo_ut.plot_po(po_objs, n_obj=po_objs.shape[1])

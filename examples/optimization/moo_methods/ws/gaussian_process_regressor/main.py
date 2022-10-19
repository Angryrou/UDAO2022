# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example on running Weighted Sum with Grid-Search solver (with GPR predictive model)
#
# Created at 14/10/2022

from optimization.moo.generic_moo import GenericMOO
from utils.optimization.configs_parser import ConfigsParser
from examples.optimization.models.gaussian_process_regressor.pre_defined_gpr_functions import GPRPredictiveModels
import utils.optimization.moo_utils as moo_ut

HELP = """
Format: python main.py -c <config> -h
    - c : The configuration file location. Default is "examples/optimization/moo_methods/ws/gaussian_process_regressor/gpr_configs_grid_search.json"
Example:
    python examples/optimization/moo_methods/ws/gaussian_process_regressor/main.py -c examples/optimization/moo_methods/ws/gaussian_process_regressor/gpr_configs_grid_search.json
"""

# get input parameters
moo_algo, solver, var_types, var_ranges, obj_names, opt_types, const_types, add_params = ConfigsParser().parse_details()

# model set up
training_vars =moo_ut.get_training_input(var_types, var_ranges, n_samples=50)
predictive_model = GPRPredictiveModels(obj_names, training_vars)

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

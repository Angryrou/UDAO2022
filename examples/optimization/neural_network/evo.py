# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example on running Weighted Sum with Grid-Search solver (with GPR predictive model)
#
# Created at 14/10/2022

from optimization.moo.generic_moo import GenericMOO
from utils.optimization.configs_parser import ConfigsParser
from examples.optimization.neural_network.model import NNPredictiveModels
import utils.optimization.moo_utils as moo_ut

import numpy as np

HELP = """
Format: python ws.py -c <config> -h
    - c : The configuration file location. Default is "examples/optimization/gaussian_process_regressor/configs/ws_grid_search.json"
Example:
    python examples/optimization/gaussian_process_regressor/ws.py -c examples/optimization/gaussian_process_regressor/configs/ws_grid_search.json
"""

# get input parameters
moo_algo, solver, var_types, var_ranges, obj_names, opt_types, const_types, add_params = ConfigsParser().parse_details()

# model set up
predictive_model = NNPredictiveModels()

# problem setup
moo = GenericMOO()

if len(obj_names) == 2:
    moo.problem_setup(obj_names=obj_names, obj_funcs=[predictive_model.predict_obj1, predictive_model.predict_obj2], opt_types=opt_types,
                  const_funcs=[predictive_model.const_func1, predictive_model.const_func2], const_types=const_types, var_types=var_types, var_ranges=var_ranges)
elif len(obj_names) == 3:
    moo.problem_setup(obj_names=obj_names, obj_funcs=[predictive_model.predict_obj1, predictive_model.predict_obj2, predictive_model.predict_obj3], opt_types=opt_types,
                  const_funcs=[], const_types=const_types, var_types=var_types, var_ranges=var_ranges)
else:
    raise Exception(f"{len(obj_names)} objectives are not supported in the code repository for now!")

# solve MOO problem
po_objs_list, po_vars_list, jobIds, time_cost_list = moo.solve(moo_algo, solver, add_params)

for i, wl_id in enumerate(jobIds):
    po_objs, po_vars = po_objs_list[i], po_vars_list[i]
    time_cost = time_cost_list[i]
    print(f"Pareto solutions of wl_{wl_id}:")
    print(po_objs)
    print(f"Variables of wl_{wl_id}:")
    print(po_vars)
    print(f"Time cost of wl_{wl_id}:")
    print(time_cost)

    data_path = f"./examples/optimization/neural_network/evo/data/{po_objs.shape[1]}d/{solver}/"
    results = np.hstack([po_objs, po_vars])
    moo_ut.save_results(data_path, results, wl_id, mode="data")
    moo_ut.save_results(data_path, [time_cost], wl_id, mode="time")

    if po_objs is not None:
        moo_ut.plot_po(po_objs, n_obj=po_objs.shape[1], title="evo")

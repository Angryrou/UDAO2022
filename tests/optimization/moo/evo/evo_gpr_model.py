# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example on running Weighted Sum with Grid-Search solver (with GPR predictive model)
#
# Created at 14/10/2022

from optimization.moo.generic_moo import GenericMOO
from utils.optimization.configs_parser import ConfigsParser
import tests.optimization.solver.models_def_ws_evo as model_def
import utils.optimization.moo_utils as moo_ut

import numpy as np
import time

HELP = """
Format: python ws.py -c <config> -h
    - c : The configuration file location. Default is "examples/optimization/gaussian_process_regressor/configs/ws_grid_search.json"
Example:
    python examples/optimization/gaussian_process_regressor/ws.py -c examples/optimization/gaussian_process_regressor/configs/ws_grid_search.json
"""

# get input parameters
moo_algo, solver, var_types, var_ranges, obj_names, opt_types, const_types, add_params = ConfigsParser().parse_details()

# the model is already set up in model_def

# problem setup
moo = GenericMOO()

if len(obj_names) == 2:
    moo.problem_setup(obj_names=obj_names, obj_funcs=[model_def.obj_func1, model_def.obj_func2], opt_types=opt_types,
                  const_funcs=[], const_types=const_types, var_types=var_types, var_ranges=var_ranges,
                  wl_list=model_def.wl_list_, wl_ranges=model_def.scaler_map, vars_constraints=model_def.conf_constraints, accurate=model_def.accurate, std_func=model_def._get_tensor_obj_std)
elif len(obj_names) == 3:
    moo.problem_setup(obj_names=obj_names, obj_funcs=[model_def.obj_func1, model_def.obj_func2, model_def.obj_func4], opt_types=opt_types,
                  const_funcs=[], const_types=const_types, var_types=var_types, var_ranges=var_ranges,
                  wl_list=model_def.wl_list_, wl_ranges=model_def.scaler_map, vars_constraints=model_def.conf_constraints, accurate=model_def.accurate, std_func=model_def._get_tensor_obj_std)
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

    data_path = f"./tests/optimization/moo/evo/test/{po_objs.shape[1]}d/"
    results = np.hstack([po_objs, po_vars])
    moo_ut.save_results(data_path, results, wl_id, mode="data")
    moo_ut.save_results(data_path, [time_cost], wl_id, mode="time")

    # if po_objs is not None:
    #     moo_ut.plot_po(po_objs, n_obj=po_objs.shape[1])

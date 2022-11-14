# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example on running Weighted Sum with Grid-Search solver
#
# Created at 22/09/2022

from optimization.moo.generic_moo import GenericMOO
from utils.optimization.configs_parser import ConfigsParser
import tests.optimization.solver.models_def as model_def
import utils.optimization.moo_utils as moo_ut

import numpy as np
import time

HELP = """
Format: python pf_mogd_model.py -c <config> -h
    - c : The configuration file location. Default is "examples/optimization/pf_mogd/predictive_model/model_configs_mogd.json"
Example:
    python examples/optimization/pf_mogd/predictive_model/pf_mogd_model.py -c examples/optimization/pf_mogd/predictive_model/model_configs_mogd.json
"""

moo_algo, solver, var_types, var_ranges, obj_names, opt_types, const_types, add_params = ConfigsParser().parse_details()

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

# assume one or more jobs
po_objs_list, po_vars_list, job_ids, time_cost_list = moo.solve(moo_algo, solver, add_params)

for i, wl_id in enumerate(job_ids):
    po_objs, po_vars = po_objs_list[i], po_vars_list[i]
    time_cost = time_cost_list[i]
    print(f"Pareto solutions of wl_{wl_id}:")
    print(po_objs)
    print(f"Variables of wl_{wl_id}:")
    print(po_vars)
    print(f"Time cost of wl_{wl_id}:")
    print(time_cost)

    pf_option = add_params[1]
    path = f"./tests/optimization/moo/pf/test/{po_objs.shape[1]}d/{pf_option}/"
    results = np.hstack([po_objs, po_vars])
    moo_ut.save_results(path, results, wl_id, mode="data")
    moo_ut.save_results(path, [time_cost], wl_id, mode="time")

    # if po_objs is not None:
    #     moo_ut.plot_po(po_objs, n_obj=po_objs.shape[1])

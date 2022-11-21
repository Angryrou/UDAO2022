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
import time

HELP = """
Format: python ws.py -c <config> -h
    - c : The configuration file location. Default is "examples/optimization/gaussian_process_regressor/configs/ws_grid_search.json"
Example:
    python examples/optimization/gaussian_process_regressor/ws.py -c examples/optimization/gaussian_process_regressor/configs/ws_grid_search.json
"""

# get input parameters
moo_algo, solver, var_types, var_ranges, obj_names, opt_types, obj_types, const_types, const_names, add_params = ConfigsParser().parse_details()

# model set up
# training_vars =moo_ut.get_training_input(var_types, var_ranges, n_samples=50)
data_file = "examples/optimization/heuristic_closed_form/ws/data/2d/random_sampler/jobId_None/data.txt"
training_vars = np.loadtxt(data_file, dtype='float32')[:,len(obj_names):]
predictive_model = NNPredictiveModels(obj_names + const_names, training_vars, var_ranges)

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

    # if solver == "grid_search":
    #     assert np.all(po_vars) == np.all(
    #         [[0, 0],
    #          [0.90909091, 0],
    #          [1.76767677, 0],
    #          [2.52525253, 0.51515152],
    #          [3.23232323, 1.21212121],
    #          [3.83838384, 1.84848485],
    #          [4.44444444, 2.45454545],
    #          [5, 3]]
    #     )
    # elif solver == "random_sampler":
    #     assert np.all(po_vars) == np.all(
    #         [[1.82984755e-01, 6.53120612e-03],
    #          [8.01352144e-01, 4.99324021e-04],
    #          [1.70848140e+00, 1.63546919e-02],
    #          [2.50618156e+00, 4.62353067e-01],
    #          [3.18137096e+00, 1.19111097e+00],
    #          [3.85358386e+00, 1.86044176e+00],
    #          [4.44723954e+00, 2.45183338e+00],
    #          [4.95843170e+00, 2.97148312e+00]]
    #     )
    # else:
    #     raise Exception(f"Solver {solver} is not available!")

    # save data
    # data_path = f"./examples/optimization/gaussian_process_regressor/ws/data/{po_objs.shape[1]}d/{solver}/"
    # results = np.hstack([po_objs, po_vars])
    # moo_ut.save_results(data_path, results, wl_id, mode="data")
    # moo_ut.save_results(data_path, [time_cost], wl_id, mode="time")
    #
    # if po_objs is not None:
    #     moo_ut.plot_po(po_objs, n_obj=po_objs.shape[1], title="ws_gpr")

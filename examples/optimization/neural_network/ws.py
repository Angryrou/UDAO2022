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
data_file = "examples/optimization/neural_network/training_data/jobId_None/data.txt"
training_vars = np.loadtxt(data_file, dtype='float32')[:,len(obj_names):]
predictive_model = NNPredictiveModels(obj_names, const_names, training_vars, var_ranges)

# problem setup
moo = GenericMOO()

if len(obj_names) == 2:
    moo.problem_setup(obj_names=obj_names, obj_funcs=[predictive_model.predict_obj1, predictive_model.predict_obj2], opt_types=opt_types,
                  const_funcs=[predictive_model.const_func1, predictive_model.const_func2], const_types=const_types, var_types=var_types, var_ranges=var_ranges)
elif len(obj_names) == 3:
    pass
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

    if len(obj_names) == 2:
        if solver == "grid_search":
            assert (np.round(po_vars, 5) == np.round(np.array(
                [[0., 0.],
                 [0.70707071, 0.],
                 [1.41414141, 0.],
                 [2.02020202, 0.03030303],
                 [2.57575758, 0.60606061],
                 [3.13131313, 1.15151515],
                 [3.63636364, 1.63636364],
                 [4.14141414, 2.12121212],
                 [4.5959596, 2.57575758],
                 [5., 3.]]
            ), 5)).all()
        elif solver == "random_sampler":
            assert (np.round(po_vars, 5) == np.round(np.array(
                [[1.82984755e-01, 6.53120612e-03],
                 [7.40020282e-01, 1.27071018e-03],
                 [1.38321782e+00, 3.87861901e-03],
                 [2.00456366e+00, 1.51113463e-02],
                 [2.59216155e+00, 5.88534056e-01],
                 [3.12915707e+00, 1.11015369e+00],
                 [3.65228034e+00, 1.63179066e+00],
                 [4.08528492e+00, 2.13814558e+00],
                 [4.58025454e+00, 2.56593583e+00],
                 [4.95843170e+00, 2.97148312e+00]]
            ), 5)).all()
        else:
            raise Exception(f"Solver {solver} is not available!")
        print("Test successfully!")

        # save data
        data_path = f"./examples/optimization/neural_network/data/{moo_algo}/{po_objs.shape[1]}d/{solver}/"
        results = np.hstack([po_objs, po_vars])
        moo_ut.save_results(data_path, results, wl_id, mode="data")
        moo_ut.save_results(data_path, [time_cost], wl_id, mode="time")

        # if po_objs is not None:
        #     moo_ut.plot_po(po_objs, n_obj=po_objs.shape[1], title="ws_nn")

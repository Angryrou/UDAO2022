# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example on running Weighted Sum with Grid-Search solver (with GPR predictive model)
#
# Created at 14/10/2022

from optimization.moo.generic_moo import GenericMOO
from utils.optimization.configs_parser import ConfigsParser
from examples.optimization.gaussian_process_regressor.model import GPRPredictiveModels
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
data_file = "examples/optimization/gaussian_process_regressor/training_data/jobId_None/data.txt"
training_vars = np.loadtxt(data_file, dtype='float32')[:,len(obj_names):]
model = GPRPredictiveModels(obj_names, const_names, training_vars, var_ranges)

# problem setup
moo = GenericMOO()

if len(obj_names) == 2:
    moo.problem_setup(obj_names=obj_names, obj_funcs=[model.predict_obj1, model.predict_obj2], opt_types=opt_types, obj_types=obj_types,
                  const_funcs=[model.const_func1, model.const_func2], const_types=const_types, var_types=var_types, var_ranges=var_ranges,
                      wl_ranges=model.get_vars_range_for_wl)
elif len(obj_names) == 3:
    moo.problem_setup(obj_names=obj_names, obj_funcs=[model.predict_obj1, model.predict_obj2, model.predict_obj3], opt_types=opt_types, obj_types=obj_types,
                  const_funcs=[], const_types=const_types, var_types=var_types, var_ranges=var_ranges, wl_ranges=model.get_vars_range_for_wl)
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
        # pf-ap
        assert (po_vars == np.array(
            [[0.  ,0.  ],
             [0.  ,0.24],
             [0.  ,0.38],
             [0.06,0.86],
             [0.  ,1.11],
             [0.  ,1.82],
             [1.98,2.  ],
             [1.65,2.95],
             [2.74,3.  ],
             [3.18,3.  ],
             [3.26,3.  ],
             [4.3 ,3.  ],
             [4.69,3.  ],
             [5.  ,3.  ]]
        )).all()

        # # pf-as
        # assert (np.round(po_vars, 5) == np.round(np.array(
        #     [[0.  ,0.  ],
        #      [0.  ,0.97],
        #      [0.  ,1.1 ],
        #      [0.  ,1.37],
        #      [0.  ,1.67],
        #      [0.04,2.23],
        #      [0.  ,2.94],
        #      [4.69,3.  ],
        #      [5.  ,3.  ]]
        # ), 5)).all()
        print("Test successfully!")

        # save data
        data_path = f"./examples/optimization/gaussian_process_regressor/data/{moo_algo}/{po_objs.shape[1]}d/{solver}/"
        results = np.hstack([po_objs, po_vars])
        moo_ut.save_results(data_path, results, wl_id, mode="data")
        moo_ut.save_results(data_path, [time_cost], wl_id, mode="time")

        # if po_objs is not None:
        #     moo_ut.plot_po(po_objs, n_obj=po_objs.shape[1], title="pf_gpr_PF-AP")
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#            Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: parse configurations
# Refer to the configs_parser.py in UDAO repository
#
# Created at 21/09/2022

import argparse, json
import numpy as np

class ConfigsParser():
    def __init__(self):
        parser = argparse.ArgumentParser(description="ConfigsParser")
        parser.add_argument("-c", "--config", required=True,
                            help="the configuration file location, try -c examples/optimization/ws_random/heuristic_closed_form/configs.json")
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

    def parse_details(self):
        args = self.parse()
        try:
            with open(args.config) as f:
                configs = json.load(f)
        except:
            raise Exception(f"{args.config} does not exist")

        # Load the configuration information correctly
        try:
            moo_algo = configs['moo_algo']
            solver = configs['solver']
            var_types, var_bounds = self.get_vars_conf(configs['variables'])
            obj_names, opt_types = self.get_objs_conf(configs['objectives'])
            add_params = []
            if moo_algo == "weighted_sum":
                ws_steps = configs['additional_params']['ws_steps']
                add_params.append(ws_steps)
                solver_params = configs['additional_params']['solver_params'] # the number of grids/samples per variables
                add_params.append(solver_params)

        except:
            raise Exception(f"configurations are not well specified.")

        return [moo_algo, solver, var_types, var_bounds, obj_names, opt_types, add_params]

    def get_vars_conf(self, var_params):
        var_types = [var["type"] for var in var_params]
        var_bounds = [[var["min"], var["max"]] for var in var_params]

        return var_types, np.array(var_bounds)

    def get_objs_conf(self, obj_params):
        obj_names = [obj["name"] for obj in obj_params]
        opt_types = [obj["optimize_trend"] for obj in obj_params]

        return obj_names, opt_types


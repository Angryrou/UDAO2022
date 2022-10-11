# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#            Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: parse configurations
# Refer to the configs_parser.py in UDAO repository
#
# Created at 21/09/2022

import argparse, json
import numpy as np

from utils.parameters import VarTypes

class ConfigsParser():
    def __init__(self):
        parser = argparse.ArgumentParser(description="ConfigsParser")
        parser.add_argument("-c", "--config", required=True,
                            help="the configuration file location, try -c examples/optimization/ws_random/heuristic_closed_form/configs.json")
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

    def parse_details(self, option=None):
        args = self.parse()
        try:
            with open(args.config) as f:
                configs = json.load(f)
        except:
            raise Exception(f"{args.config} does not exist")

        if option == None:
            # Load the configuration information correctly
            try:
                moo_algo = configs['moo_algo']
                solver = configs['solver']
                var_types, var_ranges = self.get_vars_conf(configs['variables'])
                obj_names, opt_types = self.get_objs_conf(configs['objectives'])
                const_types = self.get_const_types(configs['constraints'])
                add_params = []
                if moo_algo == "weighted_sum":
                    ws_steps = configs['additional_params']['ws_steps']
                    add_params.append(ws_steps)
                    solver_params = configs['additional_params'][
                        'solver_params']  # the number of grids/samples per variables
                    add_params.append(solver_params)
                elif moo_algo == "progressive_frontier":
                    pf_option = configs['additional_params']["pf_option"]
                    add_params.append(pf_option)
                    n_probes = configs['additional_params']["n_probes"]
                    add_params.append(n_probes)
                    n_grids = configs['additional_params']["n_grids"]
                    add_params.append(n_grids)

                    mogd_params = configs['additional_params']["mogd_params"]
                    add_params.append(mogd_params)

                elif moo_algo == "evolutionary":
                    inner_algo = configs['additional_params']['inner_algo']
                    add_params.append(inner_algo)
                    pop_size = configs['additional_params']['pop_size']
                    add_params.append(pop_size)
                    nfe = configs['additional_params']['nfe']
                    add_params.append(nfe)
                    flag = configs['additional_params']['fix_randomness_flag']
                    add_params.append(flag)
                    seed = configs['additional_params']['seed']
                    add_params.append(seed)
                else:
                    raise Exception(f"Algorithm {moo_algo} is not configured")

            except:
                raise Exception(f"configurations are not well specified.")

            return [moo_algo, solver, var_types, var_ranges, obj_names, opt_types, const_types, add_params]
        else:
            model_params = configs["model"]
            return model_params

    def get_vars_conf(self, var_params):
        var_types, var_bounds = [], []

        for var in var_params:
            if var["type"] == "FLOAT":
                var_types.append(VarTypes.FLOAT)
                var_bounds.append([var["min"], var["max"]])
            elif (var["type"] == "INTEGER"):
                var_types.append(VarTypes.INTEGER)
                var_bounds.append([var["min"], var["max"]])
            elif (var["type"] == "BINARY"):
                var_types.append(VarTypes.BOOL)
                var_bounds.append([var["min"], var["max"]])
            elif var["type"] == "ENUM":
                var_types.append(VarTypes.ENUM)
                enum_values = var["values"]
                var_bounds.append(enum_values)
            else:
                error_var_type = var["type"]
                raise Exception(f"Variable type {error_var_type} is not supported!")

        return var_types, np.array(var_bounds)

    def get_objs_conf(self, obj_params):
        obj_names = [obj["name"] for obj in obj_params]
        opt_types = [obj["optimize_trend"] for obj in obj_params]

        return obj_names, opt_types

    def get_const_types(self, const_type_params):
        const_types = [const["type"] for const in const_type_params]

        return const_types

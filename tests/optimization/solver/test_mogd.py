# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Correctness test for MOGD
#
# Created at 29/09/2022
from optimization.solver.mogd import MOGD
from utils.optimization.configs_parser import ConfigsParser
import tests.optimization.solver.models_def as model_def
import utils.optimization.solver_utils as solver_ut

BATCH_OFF_TEST_JOBS = "1-7,9-3,13-4,20-4,24-3,30-0".split(',')
class MOGDTest():
    def __init__(self):
        moo_algo, solver, var_types, var_ranges, obj_names, opt_types, const_types, add_params = ConfigsParser().parse_details()

        self.precision_list = add_params[0]
        # the commented parameters are used for PF algorithm
        # self.pf_option = add_params[1]
        # self.n_probes = add_params[2]
        # self.n_grids = add_params[3]
        self.mogd_params = add_params[4]

        self.var_types = var_types
        self.var_ranges = var_ranges

        self.mogd = MOGD(self.mogd_params)
        self.mogd._problem(wl_list=model_def.wl_list_, wl_ranges=model_def.scaler_map, vars_constraints=model_def.conf_constraints, accurate=model_def.accurate, std_func=model_def._get_tensor_obj_std, obj_funcs=[model_def.obj_func1, model_def.obj_func2], obj_names=obj_names, opt_types=opt_types, const_funcs=[],
                           const_types=const_types)

    def prediction_test(self):
        ####### test predictive_model prediction
        knob_list = ["k1","k2","k3","k4","k5","k6","k7","k8","s1","s2","s3","s4"]
        print('== predict function ==')
        for obj in ['"latency"', '"cores"']:
            for zmesg in [
                '{"s3":31,"s4":179, "k1":112,"k2":21,"k3":3,"k4":6,"k5":180,"k6":211,"k7":1,"k8":0.51,"Objective":' + obj + ',"JobID":"1-7","s1":59000,"s2":336}',
                '{"s3":242,"s4":36, "k1":96,"k2":15,"k3":4,"k4":5,"k5":432,"k6":17,"k7":0,"k8":0.59,"Objective":' + obj + ',"JobID":"9-3","s1":31240,"s2":441}',
                '{"s3":478,"s4":145, "k1":12,"k2":18,"k3":2,"k4":4,"k5":86,"k6":104,"k7":0,"k8":0.74,"Objective":' + obj + ',"JobID":"13-4","s1":10300,"s2":138}',
                '{"s3":189,"s4":87, "k1":113,"k2":9,"k3":3,"k4":7,"k5":73,"k6":202,"k7":1,"k8":0.62,"Objective":' + obj + ',"JobID":"20-4","s1":29380,"s2":502}',
                '{"s3":299,"s4":163,"k1":211,"k2":21,"k3":2,"k4":6,"k5":376,"k6":169,"k7":0,"k8":0.68,"Objective":' + obj + ',"JobID":"24-3","s1":98700,"s2":291}',
                '{"s3":240,"s4":36, "k1":43,"k2":33,"k3":4,"k4":5,"k5":480,"k6":7,"k7":1,"k8":0.69,"Objective":' + obj + ',"JobID":"30-0","s1":11240,"s2":172}'
                ]:
                wl_id, conf_raw_val, obj = solver_ut.json_parse_predict(zmesg, knob_list=knob_list)
                if obj == 'latency':
                    obj_ind = 0
                elif obj == 'cores':
                    obj_ind = 1
                else:
                    raise Exception(f"Objective {obj} is not supported!")

                if wl_id is None:
                    print(-1)
                else:
                    print(self.mogd.predict(wl_id, conf_raw_val, obj_ind, self.var_types, self.var_ranges))

    def so_test(self):
        ####### test mogd.single_obj_opt (opt_scenario1 in ICDE)
        print('== single objective optimization ==')
        if model_def.accurate:
            print('== accurate ==')
        else:
            print('== inaccurate ==')

        for obj in ['latency', 'cores']:
            for wl_id in BATCH_OFF_TEST_JOBS:
                if obj == 'latency':
                    obj_ind = 0
                elif obj == 'cores':
                    obj_ind = 1
                else:
                    raise Exception(f"Objective {obj} is not supported!")

                so = self.mogd.single_objective_opt(wl_id, obj, accurate=model_def.accurate, alpha=model_def.alpha, opt_obj_ind=obj_ind, var_types=self.var_types,
                                                    var_ranges=self.var_ranges,
                                                    precision_list=self.precision_list, bs=16)
                print(f"{wl_id}_so {obj} value is: {so[0]: .5f}")
                print(f"{wl_id}_so vars is: {so[1]}")
    def co_test(self):
        ###### test mogd.constrained_co_opt (opt_scenario2 in ICDE)
        print('== constrained single objective optimization ==')
        if model_def.accurate:
            print('== accurate ==')
        else:
            print('== inaccurate ==')

        for obj in ['latency', 'cores']:
            if obj == 'latency':
                obj_ind = 0
            elif obj == 'cores':
                obj_ind = 1
            else:
                raise Exception(f"Objective {obj} is not supported!")
            for wl_id in BATCH_OFF_TEST_JOBS:
                obj_bounds_dict = {"latency": [solver_ut._get_tensor(0), solver_ut._get_tensor(10000000)],
                                   "cores": [solver_ut._get_tensor(0), solver_ut._get_tensor(58)]
                                   }
                co = self.mogd.constraint_so_opt(wl_id, obj, accurate=model_def.accurate, alpha=model_def.alpha, opt_obj_ind=obj_ind, var_types=self.var_types, var_range=self.var_ranges,
                                                   obj_bounds_dict=obj_bounds_dict, precision_list=self.precision_list)
                print(f"The optimized objective is {obj}")
                print(f"{wl_id}_co_lat is: {co[0][0]: .5f}")
                print(f"{wl_id}_co_cores is: {co[0][1]: .1f}")
                print(f"{wl_id}_co vars is: {co[1][0]}")

    def co_parallel_test(self):
        print('== parallel constrained single-objective optimization ==')
        for obj in ['latency', 'cores']:
            if obj == 'latency':
                obj_ind = 0
            elif obj == 'cores':
                obj_ind = 1
            else:
                raise Exception(f"Objective {obj} is not supported!")
            for wl_id in BATCH_OFF_TEST_JOBS:
                cell_list = [{"latency":[solver_ut._get_tensor(0),solver_ut._get_tensor(10000000)], "cores":[solver_ut._get_tensor(0), solver_ut._get_tensor(29)]},
                             {"latency":[solver_ut._get_tensor(0),solver_ut._get_tensor(10000000)], "cores":[solver_ut._get_tensor(30),solver_ut._get_tensor(58)]}]
                ret = self.mogd.constraint_so_parallel(wl_id, obj, accurate=model_def.accurate, alpha=model_def.alpha,
                                                       opt_obj_ind=obj_ind, var_types=self.var_types, var_ranges=self.var_ranges, cell_list=cell_list, precision_list=self.precision_list)
                print(f"The optimized objective is {obj}")
                print(f"{wl_id}_[cell_1]_lat: {wl_id} --> {ret[0][0][0]: .5f}")
                print(f"{wl_id}_[cell_1]_cores: {wl_id} --> {ret[0][0][1]: .1f}")
                print(f"{wl_id}_[cell_2]_lat: {wl_id} --> {ret[0][1][0]: .5f}")
                print(f"{wl_id}_[cell_2]_cores: {wl_id} --> {ret[0][1][1]: .1f}")
                print(f"{wl_id}_vars --> {ret[1]}")

if __name__ == '__main__':
    ###### workload configurations for correctness test
    mogd_test = MOGDTest()
    mogd_test.prediction_test()
    mogd_test.so_test()
    mogd_test.co_test()
    mogd_test.co_parallel_test()



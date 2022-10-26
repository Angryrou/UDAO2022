# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Correctness test for MOGD
#
# Created at 29/09/2022
from optimization.solver.mogd import MOGD
from utils.optimization.configs_parser import ConfigsParser
import tests.optimization.solver.models_def as model_def
import utils.optimization.solver_utils as solver_ut

class MOGDTest():
    def __init__(self):
        moo_algo, solver, var_types, var_ranges, obj_names, opt_types, const_types, add_params = ConfigsParser().parse_details()

        self.precision_list = add_params[0]
        self.pf_option = add_params[1]
        self.n_probes = add_params[2]
        self.n_grids = add_params[3]
        self.mogd_params = add_params[4]

        self.var_types = var_types
        self.var_ranges = var_ranges

        self.mogd = MOGD(self.mogd_params)
        self.mogd._problem(obj_funcs=[model_def.obj_func1, model_def.obj_func2], opt_types=opt_types, const_funcs=[],
                           const_types=const_types)

    def prediction_test(self, conda_env):
        ####### test predictive_model prediction
        # test input
        # [211, 21, 2, 6, 376, 169, 0, 0.68, 98700, 291, 299, 163]
        # # expected results by using the original MOGD solver (UDAO repository) under the conda env UDAO2022 (udao2022)
        # latency = [3801.05469]
        # cores = [42]
        # # expected results by using the original MOGD solver (UDAO repository) under the same conda env as in ICDE2021 paper (py36-solvers)
        # latency = [3801.06030]
        # cores = [42]

        vars = solver_ut._get_tensor([[211, 21, 2, 6, 376, 169, 0, 0.68, 98700, 291, 299, 163]])
        # latency
        obj_pred_lat = self.mogd._get_tensor_obj_pred(vars, 0)
        # cores
        obj_pred_cores = self.mogd._get_tensor_obj_pred(vars, 1)

        print(f"latency prediction: {obj_pred_lat.item():.5f}")
        print(f"cores prediction: {obj_pred_cores.item():.5f}")

        if conda_env == 'UDAO2022':
            assert round(obj_pred_lat.item(), 5) == 3801.05469
            assert round(obj_pred_cores.item()) == 42
        elif conda_env == 'ICDE2021':
            assert round(obj_pred_lat.item(), 5) == 3801.06030
            assert round(obj_pred_cores.item()) == 42
        else:
            raise Exception("Please double-check/configure the conda env!")

    def so_test(self, conda_env):
        ####### test mogd.single_obj_opt (opt_scenario1 in ICDE)
        # # expected results by using the original MOGD solver (UDAO repository) under the conda env UDAO2022 (udao2022)
        # latency = [1626.45032]
        # cores = [4]
        # # expected results by using the original MOGD solver (UDAO repository) under the same conda env as in ICDE2021 paper (py36-solvers)
        # latency = [1626.45276]
        # cores = [4]

        so_lat = self.mogd.single_objective_opt("latency", opt_obj_ind=0, var_types=self.var_types, var_range=self.var_ranges,
                                              precision_list=self.precision_list, bs=16)
        so_cores = self.mogd.single_objective_opt("cores", opt_obj_ind=1, var_types=self.var_types, var_range=self.var_ranges,
                                                precision_list=self.precision_list, bs=16)

        print(f"so_lat is: {so_lat[0]: .5f}")
        print(f"so_cores is: {so_cores[0]: .5f}")

        if conda_env == 'UDAO2022':
            assert round(so_lat[0], 5) == 1626.45032
            assert round(so_cores[0]) == 4
        elif conda_env == 'ICDE2021':
            assert round(so_lat[0], 5) == 1626.45276
            assert round(so_cores[0]) == 4
        else:
            raise Exception("Please double-check/configure the conda env!")

    def co_test(self, conda_env):
        ###### test mogd.constrained_co_opt (opt_scenario2 in ICDE)
        # # expected results by using the original MOGD solver (UDAO repository) under the conda env UDAO2022 (udao2022)
        # obj=latency: [latency, cores] = [6450.26123, 50]
        # obj=cores: [latency, cores] = [9046.02930, 4]
        # # expected results by using the original MOGD solver (UDAO repository) under the same conda env as in ICDE2021 paper (py36-solvers)
        # obj=latency: [latency, cores] = [6450.26758, 50]
        # obj=cores: [latency, cores] = [9046.03906, 4]
        obj_bounds_dict = {"latency": [solver_ut._get_tensor(0), solver_ut._get_tensor(10000000)],
                           "cores": [solver_ut._get_tensor(0), solver_ut._get_tensor(58)]
                           }
        co_lat = self.mogd.constraint_so_opt("latency", opt_obj_ind=0, var_types=self.var_types, var_range=self.var_ranges,
                                           obj_bounds_dict=obj_bounds_dict, precision_list=self.precision_list)
        co_cores = self.mogd.constraint_so_opt("cores", opt_obj_ind=1, var_types=self.var_types, var_range=self.var_ranges,
                                             obj_bounds_dict=obj_bounds_dict, precision_list=self.precision_list)

        print(f"co_lat is: {co_lat}")
        print(f"co_cores is: {co_cores}")

        if conda_env == 'UDAO2022':
            assert round(co_lat[0][0], 5) == 6450.26123
            assert round(co_lat[0][1]) == 50
            assert round(co_cores[0][0], 5) == 9046.02930
            assert round(co_cores[0][1]) == 4
        elif conda_env == 'ICDE2021':
            assert round(co_lat[0][0], 5) == 6450.26758
            assert round(co_lat[0][1]) == 50
            assert round(co_cores[0][0], 5) == 9046.03906
            assert round(co_cores[0][1]) == 4
        else:
            raise Exception("Please double-check/configure the conda env!")

if __name__ == '__main__':
    ###### workload configurations for correctness test
    # for wl_id = "24-3"
    # var names = ["k1","k2","k3","k4","k5","k6","k7","k8","s1","s2","s3","s4"]
    # GPR is accurate

    ###### set the conda environment
    # Please make sure the settings here is the same as the environment you are running
    # conda_env = 'ICDE2021' ## named as 'py36-solvers'
    conda_env = 'UDAO2022' ## named as 'udao2022'

    mogd_test = MOGDTest()
    mogd_test.prediction_test(conda_env)
    mogd_test.so_test(conda_env)
    mogd_test.co_test(conda_env)



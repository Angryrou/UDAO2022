### Running MOGD Test
This file shows how to run the Multi-Objective Gradient Descent (MOGD) correctness test.

#### Document Description
-  `tests/optimization/checkpoints/data_gpr_batch.pkl`: the data used for Gaussian Process Regressor (GPR) model, 
which is the same as the one used in ICDE2021 paper.
    - NOTE: The data is not in the repository. Please download it from the `hex3@node13` on ercilla under the path `/opt/hex_users/hex3/common/test_mogd/checkpoints`.
    And then copy to the path `tests/optimization/checkpoints` in `UDAO2022` repository. 
- `solver/model_configs_modg.json`: The configuration file to feed in GPR.
- `solver/models_def.py`: The script to initialize the GPR model and to get objective predictions, which is the same as what we used in the ICDE2021 paper.
- `test_mogd`: The script to run correctness test over MOGD solver. It provides tests under two conda environments, 
 i.e. `py36-solvers` used in ICDE paper, and the `udao2022` environment for the current code release.
    - NOTE: Both `py36-solvers` and `udao2022` have already been installed in `hex3@node13` on ercilla.

#### MOGD Test

- **MOGD Overview**
  
  The MOGD provides optimization for Single-Objective (SO) and Constrained single-objective Optimziation (CO).
It also supports to run CO parallelly.
  
- **MOGD Correctness Test**

    The file `tests/optimization/solver/test_mogd.py` provides correctness test for predictions, SO and CO based on the GPR model (used in ICDE paper) for one workload . 
  The workload configurations and test functions are as follows:
    ```bash
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
    ```
  The following is the command to run this test:
    ```bash
    export PYTHONPATH=$PWD # export PYTHONPATH=~/your_path_to/UDAO2022
    python tests/optimization/solver/test_mogd.py -c tests/optimization/solver/model_configs_mogd.json
    ```
   NOTE: By default, the `test_mogd.py` takes the parameter for conda environment as `udao2022`. 
   If you want to double-check whether the current MOGD returns the same results as that in ICDE paper, 
   please modify the parameter named `conda_env` as `ICDE2021` manually in the script and change the running the conda environment accordingly.
     
### Running MOGD Test
This file shows how to run the Multi-Objective Gradient Descent (MOGD) correctness test.

#### Document Description
-  `tests/optimization/checkpoints/data_gpr_batch.pkl`: 
   The data used for Gaussian Process Regressor (GPR) model, 
which is the same as the one used in ICDE2021 paper.
    - NOTE: The data is not in the repository. Please:
      1) download it from the `hex3@node13` on ercilla under the path `/opt/hex_users/hex3/common/test_mogd/checkpoints`.
      2) create directory `tests/optimization/checkpoints`
      3) copy the `data_gpr_batch.pkl` to the path `tests/optimization/checkpoints` in `UDAO2022` repository.
- `tests/optimization/solver/model_configs_modg.json`: 
  The configuration file to feed in GPR.
- `tests/optimization/solver/models_def.py`: 
  The script to initialize the GPR model and to get objective predictions, which is the same as what we used in the ICDE2021 paper.
- `tests/optimization/solver/test_mogd.py`: 
  The script to run correctness test over MOGD solver. It provides tests for MOGD used in ICDE paper under the conda environment `py36-solvers`.

#### MOGD Test

- **MOGD Overview**
  
  The MOGD provides optimization for Single-Objective (SO) and Constrained single-objective Optimziation (CO).
It also supports to run CO parallelly.
  
- **MOGD Parameters**
    - Common parameters for MOGD:
    ```json
    "mogd_params":
      {
      "learning_rate": 0.1,
      "weight_decay": 0.1,
      "max_iters": 100,
      "patient": 20,
      "stress": 10,
      "multistart": 4,
      "processes": 1,
      "seed": 0
        }
    ```
   - Specific parameters required by the predictive models
    MOGD supports to implement accurate and inaccurate models, where users need to specify them both in the configuration file
     and the script of define predictive model functions.
     ```json
     "model": {
        "name": "gpr",
        "gpr_weights_path": "tests/optimization/checkpoints/data_gpr_batch.pkl",
        "default_ridge": 1.0,
        "accurate": false,
        "alpha": 3.0  /* used in inaccurate model, if accurate is true, alpha = 0  */
     }
     ```
     ```python
     # constraints for variables
     # the purpose is to put variable values into a region where the model performs better
     if accurate:
        conf_constraints = None
     else:
        conf_constraints = {"vars_min": np.array([64, 8, 2, 6, 24, 35, 0, 0.5, 5000, 64, 10, 36]),
                        "vars_max": np.array([144, 24, 4, 8, 192, 145, 1, 0.75, 20000, 256, 100, 144])}
     
     # std functions for loss calculation of inaccurate models in MOGD
     def _get_tensor_obj_std(wl_id, conf, obj):
        ...
     
     # get upper and lower bounds of all variable for one workload
     def _get_conf_range_for_wl(wl_id):
        ...
     
     # initialize GPR models for all workloads
     def _get_gp_models(data, proxy_jobs, wl_list, ridge):
        ...
     ```
    - Parameters are passed to MOGD by:
    ```python
    from optimization.solver.mogd import MOGD
    from utils.optimization.configs_parser import ConfigsParser
    import tests.optimization.solver.models_def as model_def  # define predictive model functions
    class MOGDTest():
        def __init__(self):
            moo_algo, solver, var_types, var_ranges, obj_names, opt_types, const_types, add_params = ConfigsParser().parse_details()
    
            self.precision_list = add_params[0]
            self.mogd_params = add_params[4]
    
            self.var_types = var_types
            self.var_ranges = var_ranges
    
            self.mogd = MOGD(self.mogd_params)
            self.mogd._problem(wl_list=model_def.wl_list_, wl_ranges=model_def.scaler_map, vars_constraints=model_def.conf_constraints, accurate=model_def.accurate, std_func=model_def._get_tensor_obj_std, obj_funcs=[model_def.obj_func1, model_def.obj_func2], obj_names=obj_names, opt_types=opt_types, const_funcs=[],
                               const_types=const_types)
    ```
    For more details, please look into the [configuration file](../../tests/optimization/solver/model_configs_mogd.json)
    , the script of [model definition](../../tests/optimization/solver/models_def.py) 
    and the [test script](../../tests/optimization/solver/test_mogd.py) as an example.


- **MOGD Correctness Test**

    The file `tests/optimization/solver/test_mogd.py` provides correctness test for predictions, SO and CO based on the GPR model (used in ICDE paper) for one workload . 
  The workload configurations and test functions are as follows:
    ```python
    ###### workload configurations for correctness test
    # BATCH_OFF_TEST_JOBS = "1-7,9-3,13-4,20-4,24-3,30-0".split(',')
    # var names = ["k1","k2","k3","k4","k5","k6","k7","k8","s1","s2","s3","s4"]
    # objectives = ['latency', 'cores']

    mogd_test = MOGDTest()
    mogd_test.prediction_test()
    mogd_test.so_test()
    mogd_test.co_test()
    mogd_test.co_parallel_test()
    ```
  The following is the command to run this test:
    ```bash
    export PYTHONPATH=$PWD # export PYTHONPATH=~/your_path_to/UDAO2022
    python tests/optimization/solver/test_mogd.py -c tests/optimization/solver/model_configs_mogd.json
    ```
  
The correctness test runs on `Ercilla node19` under conda environment `py36-solvers`. 
<details>
<summary>
The following is the expected results of inaccurate models from the MOGD solver in ICDE paper, with the parameters shown above.
It includes the correctness tests for prediction, SO, CO and parallel CO.
</summary> 

```bash
== predict function ==
# for latency
4348.74658
6777.44336
13380.95508
12344.32227
3801.06519
139027.29688
# for cores
57.00000
56.00000
36.00000
27.00000
42.00000
56.00000

== opt1 function ==
[inaccurate] 1-7 --> k1:144;k2:18;k3:4;k4:8;k5:48;k6:145;k7:1;k8:0.60;s1:9039;s2:124;s3:21;s4:144&latency:2561.86011
[inaccurate] 9-3 --> k1:144;k2:24;k3:3;k4:6;k5:49;k6:145;k7:1;k8:0.60;s1:9811;s2:128;s3:10;s4:144&latency:4253.35254
[inaccurate] 13-4 --> k1:144;k2:24;k3:3;k4:6;k5:49;k6:145;k7:1;k8:0.60;s1:10031;s2:128;s3:10;s4:144&latency:6490.23193
[inaccurate] 20-4 --> k1:144;k2:24;k3:3;k4:6;k5:51;k6:145;k7:1;k8:0.60;s1:10434;s2:131;s3:11;s4:36&latency:4966.75342
[inaccurate] 24-3 --> k1:144;k2:24;k3:3;k4:6;k5:49;k6:145;k7:1;k8:0.60;s1:9812;s2:127;s3:10;s4:144&latency:2927.69873
[inaccurate] 30-0 --> k1:144;k2:24;k3:3;k4:6;k5:49;k6:145;k7:1;k8:0.60;s1:9893;s2:128;s3:10;s4:144&latency:125491.38281
[inaccurate] 1-7 --> k1:84;k2:8;k3:2;k4:7;k5:192;k6:39;k7:0;k8:0.50;s1:5000;s2:64;s3:100;s4:144&cores:16.00000
[inaccurate] 9-3 --> k1:84;k2:8;k3:2;k4:7;k5:192;k6:39;k7:0;k8:0.50;s1:5000;s2:64;s3:100;s4:144&cores:16.00000
[inaccurate] 13-4 --> k1:84;k2:8;k3:2;k4:7;k5:192;k6:36;k7:0;k8:0.50;s1:5000;s2:64;s3:100;s4:144&cores:16.00000
[inaccurate] 20-4 --> k1:84;k2:8;k3:2;k4:7;k5:192;k6:42;k7:0;k8:0.50;s1:5000;s2:64;s3:100;s4:144&cores:16.00000
[inaccurate] 24-3 --> k1:84;k2:8;k3:2;k4:7;k5:192;k6:40;k7:0;k8:0.50;s1:5000;s2:64;s3:100;s4:144&cores:16.00000
[inaccurate] 30-0 --> k1:84;k2:8;k3:2;k4:7;k5:192;k6:39;k7:0;k8:0.50;s1:5000;s2:64;s3:100;s4:144&cores:16.00000

== opt2 function ==
[inaccurate] 1-7 --> k1:64;k2:16;k3:2;k4:6;k5:33;k6:118;k7:1;k8:0.63;s1:5000;s2:209;s3:60;s4:36&cores:32.00000;latency:4662.23242
[inaccurate] 9-3 --> k1:95;k2:20;k3:3;k4:8;k5:24;k6:53;k7:0;k8:0.65;s1:20000;s2:256;s3:100;s4:144&cores:57.00000;latency:6998.95654
[inaccurate] 13-4 --> k1:70;k2:24;k3:3;k4:6;k5:192;k6:35;k7:0;k8:0.52;s1:20000;s2:130;s3:100;s4:144&cores:57.00000;latency:11052.61816
[inaccurate] 20-4 --> k1:64;k2:16;k3:2;k4:6;k5:33;k6:120;k7:1;k8:0.63;s1:5000;s2:209;s3:59;s4:36&cores:32.00000;latency:11331.82324
[inaccurate] 24-3 --> k1:95;k2:20;k3:3;k4:8;k5:24;k6:52;k7:0;k8:0.65;s1:20000;s2:256;s3:100;s4:144&cores:57.00000;latency:4268.14453
[inaccurate] 30-0 --> k1:64;k2:19;k3:2;k4:6;k5:78;k6:139;k7:1;k8:0.65;s1:11279;s2:256;s3:100;s4:144&cores:38.00000;latency:213990.12500
[inaccurate] 1-7 --> k1:64;k2:8;k3:2;k4:6;k5:24;k6:35;k7:0;k8:0.50;s1:5000;s2:64;s3:10;s4:36&cores:16.00000;latency:5518.72266
[inaccurate] 9-3 --> k1:64;k2:8;k3:2;k4:6;k5:24;k6:35;k7:0;k8:0.50;s1:5000;s2:64;s3:10;s4:36&cores:16.00000;latency:11033.08691
[inaccurate] 13-4 --> k1:64;k2:8;k3:2;k4:6;k5:24;k6:35;k7:0;k8:0.50;s1:5000;s2:64;s3:10;s4:36&cores:16.00000;latency:16005.14844
[inaccurate] 20-4 --> k1:64;k2:8;k3:2;k4:6;k5:24;k6:35;k7:0;k8:0.50;s1:5000;s2:64;s3:10;s4:36&cores:16.00000;latency:14678.56348
[inaccurate] 24-3 --> k1:64;k2:8;k3:2;k4:6;k5:24;k6:35;k7:0;k8:0.50;s1:5000;s2:64;s3:10;s4:36&cores:16.00000;latency:7213.73877
[inaccurate] 30-0 --> k1:64;k2:8;k3:2;k4:6;k5:24;k6:35;k7:0;k8:0.50;s1:5000;s2:64;s3:10;s4:36&cores:16.00000;latency:299666.96875

== opt3 function ==
[inaccurate] 1-7 --> k1:64;k2:13;k3:2;k4:6;k5:24;k6:97;k7:1;k8:0.61;s1:5000;s2:163;s3:25;s4:36&cores:26.00000;latency:4915.92725|k1:64;k2:16;k3:3;k4:6;k5:59;k6:71;k7:1;k8:0.62;s1:5000;s2:64;s3:10;s4:36&cores:48.00000;latency:4177.25879
[inaccurate] 9-3 --> k1:64;k2:13;k3:2;k4:6;k5:24;k6:97;k7:1;k8:0.61;s1:5000;s2:163;s3:25;s4:36&cores:26.00000;latency:9280.54102|k1:64;k2:16;k3:3;k4:6;k5:58;k6:71;k7:1;k8:0.62;s1:5000;s2:64;s3:10;s4:36&cores:48.00000;latency:6328.74219
[inaccurate] 13-4 --> k1:64;k2:14;k3:2;k4:6;k5:150;k6:126;k7:1;k8:0.51;s1:11411;s2:137;s3:55;s4:36&cores:28.00000;latency:14012.83691|k1:64;k2:20;k3:2;k4:6;k5:72;k6:70;k7:1;k8:0.60;s1:5000;s2:64;s3:10;s4:36&cores:40.00000;latency:11723.70898
[inaccurate] 20-4 --> k1:64;k2:13;k3:2;k4:6;k5:24;k6:100;k7:1;k8:0.61;s1:5000;s2:163;s3:24;s4:36&cores:26.00000;latency:12337.13770|k1:64;k2:16;k3:3;k4:6;k5:58;k6:74;k7:1;k8:0.62;s1:5000;s2:64;s3:10;s4:36&cores:48.00000;latency:9761.66309
[inaccurate] 24-3 --> k1:64;k2:13;k3:2;k4:6;k5:24;k6:91;k7:1;k8:0.61;s1:5000;s2:163;s3:25;s4:36&cores:26.00000;latency:6449.56006|k1:95;k2:20;k3:3;k4:8;k5:24;k6:52;k7:0;k8:0.65;s1:20000;s2:256;s3:100;s4:144&cores:57.00000;latency:4268.14453
[inaccurate] 30-0 --> k1:65;k2:14;k3:2;k4:6;k5:149;k6:125;k7:1;k8:0.51;s1:11122;s2:136;s3:54;s4:36&cores:28.00000;latency:244384.35938|k1:64;k2:24;k3:2;k4:6;k5:24;k6:35;k7:1;k8:0.52;s1:5000;s2:64;s3:17;s4:53&cores:48.00000;latency:191522.51562
[inaccurate] 1-7 --> k1:64;k2:8;k3:2;k4:6;k5:24;k6:35;k7:0;k8:0.50;s1:5000;s2:64;s3:10;s4:36&cores:16.00000;latency:5518.72266|k1:64;k2:15;k3:2;k4:6;k5:24;k6:58;k7:0;k8:0.54;s1:5000;s2:67;s3:10;s4:36&cores:30.00000;latency:5141.49463
[inaccurate] 9-3 --> k1:64;k2:8;k3:2;k4:6;k5:24;k6:35;k7:0;k8:0.50;s1:5000;s2:64;s3:10;s4:36&cores:16.00000;latency:11033.08691|k1:64;k2:15;k3:2;k4:6;k5:24;k6:58;k7:0;k8:0.54;s1:5000;s2:67;s3:10;s4:36&cores:30.00000;latency:9939.08398
[inaccurate] 13-4 --> k1:64;k2:8;k3:2;k4:6;k5:24;k6:35;k7:0;k8:0.50;s1:5000;s2:64;s3:10;s4:36&cores:16.00000;latency:16005.14844|k1:64;k2:15;k3:2;k4:6;k5:24;k6:54;k7:0;k8:0.54;s1:5000;s2:67;s3:10;s4:36&cores:30.00000;latency:14634.89258
[inaccurate] 20-4 --> k1:64;k2:8;k3:2;k4:6;k5:24;k6:35;k7:0;k8:0.50;s1:5000;s2:64;s3:10;s4:36&cores:16.00000;latency:14678.56348|k1:64;k2:15;k3:2;k4:6;k5:24;k6:61;k7:0;k8:0.54;s1:5000;s2:67;s3:10;s4:36&cores:30.00000;latency:13312.47070
[inaccurate] 24-3 --> k1:64;k2:8;k3:2;k4:6;k5:24;k6:35;k7:0;k8:0.50;s1:5000;s2:64;s3:10;s4:36&cores:16.00000;latency:7213.73877|k1:64;k2:15;k3:2;k4:6;k5:24;k6:57;k7:0;k8:0.54;s1:5000;s2:67;s3:10;s4:36&cores:30.00000;latency:6510.43750
[inaccurate] 30-0 --> k1:64;k2:8;k3:2;k4:6;k5:24;k6:35;k7:0;k8:0.50;s1:5000;s2:64;s3:10;s4:36&cores:16.00000;latency:299666.96875|k1:64;k2:15;k3:2;k4:6;k5:24;k6:58;k7:0;k8:0.54;s1:5000;s2:67;s3:10;s4:36&cores:30.00000;latency:292968.87500
```    
</details>

## Features for generalization in PF algorithm
1. In the current MOGD implementation, the variables are normalized by default to feed in the GPR models, which is required as the input.
While a more general case would be that models (e.g. HCF) take the origial variable values rather than the normalized. So this needs to be improved further, which I marked as `fixme` in the MOGD solver;
2. Loss design for constrained functions;
3. Double-check the loss design for the objectives with negative values under minimization optimization (e.g. throughput)
     

[comment]: # (Taken from UDAO repository. Path: UDAO/snapshot/pythonZMQServer/batch_knob_format.md)

Batch Knob and Message Format

## JobID

- For batch workloads, there are 58 offline workloads and 200 online workloads. 
- The Job ID for batch workloads are in the form `a-b`, where `a` is the template id chosen from `1` to `30`, and b is the variant id chosen from `0` to `8`. 
E.g., Job ID `1-0` is the first workload generated from template 1.
- It is worth mentioning that template `10` and `11` only generate 3 workloads, while other templates generate 9 workloads 
in total. So we get all the workloads `28 * 9 + 2 * 3 = 258` workloads.
- We randomly choose 1 workload from template `10` and `11`, 2 workloads from other templates to get `1 * 2 + 2 * 28 = 58` offline workloads. 

See the JobID list for [offline](../../examples/model/spark/batch_confs/JobID_offline.txt) and [online](../../examples/model/spark/batch_confs/JobID_online.txt).


## Objective 

There are three user objectives from NNR model now

- latency, named as `latency` in the message.
- CPU utils, named as `cpu` in the message.
- network utils, named as `network` in the message.

## Configurations

### knob alias

For batch workloads, we use k1 to k8 to represent the 8 Spark general knobs, and s1 to s4 to represent 4 Spark SQL 
related knobs

```python
knob_alias = {
  "k1": "spark.defalut.parallelism",
  "k2": "spark.executor.instances",
  "k3": "spark.executor.cores",
  "k4": "spark.executor.memory",
  "k5": "spark.reducer.maxsizeInFlight",
  "k6": "spark.shuffle.sort.bypassMergeThreshold",
  "k7": "spark.shuffle.compress", # 1 for True and 0 for False
  "k8": "spark.memory.fraction",
  "s1": "spark.sql.inMemoryColumnarstorage.batchsize",
  "s2": "spark.sql.files.maxPartitionBytes",
  "s3": "spark.sql.autoBroadcastJoinThreshold",
  "s4": "spark.sql.shuffle.partitions"
}
```

### knob ranges

We have the maximum values for each knob.
```python
knob_list = ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "s1", "s2", "s3", "s4"]
knob_max_list = [216, 36, 4, 8, 480, 217, 1, 75, 100000, 512, 500, 2001]
knob_min_list = [8, 2, 2, 4, 12, 7, 0, 50, 1000, 32, 10, 8]
```

It is worth mentioning:

1. `k7`: `spark.shuffle.compress` is a `True` or `False` value. Here I used `1` for `True` and `0` for `False`
2. `k8`: `spark.memory.fraction` is a Float type. To match the Integer type needed in MOO, 
I converted it to a percentile value. E.g., `0.6` --> `60`
3. `(k2, k3, k4)` has only 17 combinations as following:
    ```python
    k2k3k4_list =  [
        (2,2,4),
        (3,2,4),
        (2,4,8),
        (4,2,4),
        (3,4,8),
        (4,3,6),
        (6,2,4),
        (4,4,8),
        (8,2,4),
        (6,4,8),
        (8,3,6),
        (12,2,4),
        (8,4,8),
        (16,2,4),
        (18,4,8),
        (24,3,6),
        (36,2,4)
    ]
    ```

## Scenarios and format 

0. Rule of format
    - The key:value pairs of JobID, Objective, and knobs are joined by `";"`
    - Objective and knobs are joined by `"&"`
    - information related to a bunch of cells are joined by `"|"`
    - all the objective values accurate to 5 decimal places
    - if there is no configurations found within constrained objectives, return "not_found"
    - See more real example from 1 to 4  
    
1. Predict an objective given a specific configuration and a workload
    ```python
    # E.g., to predict cpu utilization for 1-2 upon a knob list [48, 4, 4, 8, 48, 200, 1, 60, 10000, 128, 10, 200]    
    # input: "JobID:13-4;Objective:latency;k1:48;k2:4;k3:4;k4:8;k5:48;k6:200;k7:1;k8:60;s1:10000;s2:128;s3:10;s4:200"
    # output: string form of a scalar obj value
    X_predict_latency = "JobID:13-4;Objective:latency;k1:48;k2:4;k3:4;k4:8;k5:48;k6:200;k7:1;k8:60;s1:10000;s2:128;s3:10;s4:200"
    ret = bm.predict(X_predict_latency) 
    # ret = '16903.28906'
    ```

2. MOO scenario1: Given a target objective, return the best objective result
   and its configurations
    ```python
    # E.g., to minimize the objective latency
    # input: string form of JobId and target objective
    # output: string form of conf, and minimum objective value
    X_predict_latency = "JobID:1-2;Objective:latency"
    ret = bm.opt_scenario1(zmesg=X_predict_latency, max_iter=200, lr=0.01, verbose=False)
    # ret = 'k1:143;k2:24;k3:3;k4:6;k5:335;k6:7;k7:0;k8:50;s1:99933;s2:32;s3:495;s4:377&latency:1316.51282'
    ``` 
    
3. MOO scenario2: Given k objectives with upper and lower bounds, minimize one of the objectives. Return
the scalar value for each objective and the best configuration.
    ```python
    # E.g., with upper and lower bounds for 2 objectives latency and cpu cores, minimize the latency. Return best 
    # configuration together with cpu cores used and the latency prediction
    # input: string form of JobID, target objective and k constraints
    # output: 'not_found' or string form of conf and the best objective value achieved
    X_predict_latency1 = "JobID:13-4;Objective:latency;Constraint:cores:10:20;Constraint:latency:00000:10000"
    ret1 = bm.opt_scenario2(X_predict_latency1, max_iter=100, lr=0.01, verbose=False) 
    # ret1 = 'not_found'
    X_predict_latency2 = "JobID:13-4;Objective:latency;Constraint:cores:10:20;Constraint:latency:10000:20000"
    ret2 = bm.opt_scenario2(X_predict_latency2, max_iter=100, lr=0.01, verbose=False) 
    # ret2 = 'k1:181;k2:4;k3:4;k4:8;k5:410;k6:7;k7:0;k8:50;s1:2213;s2:512;s3:25;s4:8&cores:16.00000;latency:11511.57129'
    ```
   
4. MOO scenario3: a parallel version of scenario2. Scenario 2 requests a cell in the MOO space while scenario 3 will 
request a bunch of cells in the MOO space. And we can handle them sequentially or simultaneously.
    ```python
    # E.g., send request of zmesg1 and zmesg2 together. 
    # input: string form of a bunch of zmesg in scenario connected by "|": "cell_1|cell_2|...|cell_k"
    # output: string form of a bunch of returned resutls connected by "|"
    zmesg = "JobID:13-4;Objective:latency;Constraint:cores:10:20;Constraint:latency:00000:10000|JobID:13-4;Objective:latency;Constraint:cores:10:20;Constraint:latency:10000:20000"
    ret = bm.opt_scenario3(zmesg, max_iter=100, lr=0.01, verbose=False) 
    # ret = 'not_found|k1:181;k2:4;k3:4;k4:8;k5:410;k6:7;k7:0;k8:50;s1:2213;s2:512;s3:25;s4:8&cores:16.00000;latency:11511.57129'
    ```
   
5. MOO scenario4: a weigthed-sum approach to handle MOO. Returen the scalar value for each objective and the recommended configuration.
    ```python 
    # please pay attention to the order of the usage of objectives (could be extended later on)
    zmesg = "JobID:13-4;objectives:latency:cores;weights:0.3:0.7"
    ret = bm.opt_scenario4(zmesg, max_iter=100, lr=0.01, verbose=False)
    # ret = 'k1:181;k2:4;k3:4;k4:8;k5:410;k6:7;k7:0;k8:50;s1:2213;s2:512;s3:25;s4:8&latency:11511.57129;cores:16'
   ```   
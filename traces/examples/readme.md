Trace Collection Notes
---

### Default Spark Parameters

Most parameters have a default value according to Spark. For the resource parameters that do not have default values, here are our consideration.

1. Our Spark-cluster has 5 worker nodes, each with 30 cores and 700G memory.
2. in Yarn, `spark.executor.cores=1` by default, while the `spark.executor.instances` and `spark.executor.memory` do not have the default values.
3. We set `spark.executor.instances=14` in the default setting to support 7 Spark SQLs running in parallel. 
4. We set `spark.executor.memory=2G` to meet the 1core-2G memory ratio between cores and memory.

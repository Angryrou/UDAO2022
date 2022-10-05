Traces
======

From the `trace` package and examples, we describe and point out how we handle the trace collection in different benchmarks and system settings, including

<!--ts-->
* [Single-query environment](#single-query-environment)
  - [Spark 2.3.1](#spark-231)
    - [Spark-TPCxBB](#spark-tpcxbb)
  - [Postgres 12.4](#postgres-124)
    - [PSQL-TPCH](#psql-tpch)
    - [PSQL-TPCDS](#psql-tpcds)
* [Multi-query environment](#multi-query-environment)
  - [Spark 3.2.1](#spark-321)
    - [Spark-TPCH](#spark-tpch)
    - [Spark-TPCDS](#spark-tpcds)
  - MaxCompute (confidential)
<!--te-->


## Single-query environment

### Spark 2.3.1

#### Spark-TPCxBB

### Postgres 12.4

#### PSQL-TPCH

#### PSQL-TPCDS

## Multi-query environment

Our traces from the multi-query environment are either from the production workloads or the workloads mimicing the production.
Although the real traces from the industry world is confidential, we run TPCH and TPCDs over Spark 3.2.1 to mimicinng the real world system states.

### Spark 3.2.1

Here is the list of the selected Spark knobs (when `spark.dynamicAllocation.enabled` is disabled)

```yaml
k1: spark.executor.memory
k2: spark.executor.cores
k3: spark.executor.instances
k4: spark.defalut.parallelism
k5: spark.reducer.maxSizeInFlight
k6: spark.shuffle.sort.bypassMergeThreshold
k7: spark.shuffle.compress
k8: spark.memory.fraction
s1: spark.sql.inMemoryColumnarStorage.batchSize
s2: spark.sql.files.maxPartitionBytes
s3: spark.sql.autoBroadcastJoinThreshold
s4: spark.sql.shuffle.partitions
```

Internal Knobs over the 6-node Spark cluster (each with 30 cores and 680G memory) with the [best practice][1] at Amazon
EMR.

1. k1 takes `{4G, 8G, 16G, 32G, 64G}`
2. k2 takes `2-5` (workload)
3. k3 takes `4-10` (8-50 cores in total)
4. k4 takes `1-3x` total cores `parallelism = [1|2|3] * k2 * k3`, value in the range of [8, 150]
5. k5 takes `{12M, 24M, 48M, 96M, 192M, 384M}`
6. k6 takes `{ON, OFF}` to control how the knob adapting its value
7. k7 takes `{True, False}`
8. k8 takes `{0.5, 0.55, 0.6, 0.65, 0.7, 0.75}`
9. s1 takes `{2500, 5000, 10000, 20000, 40000}`
10. s2 takes `{32M, 64M, 128M, 256M, 512M}`
11. s3 takes `{10M, 20M, 40M, 80M, 160M, 320M}`
12. s4 takes `{ON, OFF}` to control either the knob choose `parallelism` or `2001` (highly compressed data
    when `s4>2000`)

We also fixed some knobs according to the [best practice][1]. E.g., we have the following lines in `spark-defaults.conf` 
```bash
spark.serializer=org.apache.spark.serializer.KryoSerializer
spark.kryoserializer.buffer.max=512m
spark.sql.adaptive.enabled=false
spark.sql.cbo.enabled=true
spark.sql.cbo.joinReorder.dp.star.filter=true
spark.sql.cbo.joinReorder.enabled=true
spark.sql.cbo.planStats.enabled=true
spark.sql.cbo.starSchemaDetection=true
```

[1]: https://aws.amazon.com/blogs/big-data/best-practices-for-successfully-managing-memory-for-apache-spark-applications-on-amazon-emr/

[2]: https://spoddutur.github.io/spark-notes/distribution_of_executors_cores_and_memory_for_spark_application.html

#### Spark-TPCH

Here are the key steps for the trace collection. For more details, please refer to [3.Spark-TPCH-and-TPCDS.md](./3.Spark-TPCH-and-TPCDS.md)

```bash
# generate ~100K queries
bash examples/trace/spark/1.query_generation_tpch.sh $PWD/resources/tpch-kit $PWD/resources/tpch-kit/spark-sqls 4545
# 80% collected in LHS
# generate and run LHS configurations
python examples/trace/spark/5.generate_scripts_for_lhs.py -b TPCH -q resources/tpch-kit/spark-sqls --num-processes 30 --num-templates 22 --num-queries-per-template 3637
python examples/trace/spark/6.run_all_pressure_test.py -b TPCH --num-processes 22 --num-templates 22 --num-queries-per-template-to-run 3637 
# 10% in BO-latency
# todo
# 10% in BO-cost
# todo
```

#### Spark-TPCDS

 


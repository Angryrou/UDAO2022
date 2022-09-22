Traces
======

<!--ts-->

* [Spark_TPCH](#spark_tpch)
    * [collection](#collection)
    * [parsing](#parsing)
* [Spark_TPCDS](#spark_tpch)
    * [collection](#collection)
    * [parsing](#parsing)

<!--te-->

Spark
=====

Configurations
--------------

List of the selected Spark knobs (when `spark.dynamicAllocation.enabled` is disabled)

```yaml
k1 -> spark.executor.memory
k2 -> spark.executor.cores
k3 -> spark.executor.instances
k4 -> spark.defalut.parallelism
k5 -> spark.reducer.maxSizeInFlight
k6 -> spark.shuffle.sort.bypassMergeThreshold
k7 -> spark.shuffle.compress
k8 -> spark.memory.fraction
s1 -> spark.sql.inMemoryColumnarStorage.batchSize
s2 -> spark.sql.files.maxPartitionBytes
s3 -> spark.sql.autoBroadcastJoinThreshold
s4 -> spark.sql.shuffle.partitions
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

[1]: https://aws.amazon.com/blogs/big-data/best-practices-for-successfully-managing-memory-for-apache-spark-applications-on-amazon-emr/

[2]: https://spoddutur.github.io/spark-notes/distribution_of_executors_cores_and_memory_for_spark_application.html

TPCH
----

1. Setup TPCH benchmark over a Spark cluster

```bash
git clone https://github.com/Angryrou/spark-sql-perf.git
cd spark-sql-perf
bin/run --help # testing env
sbt +package


# an example of running in our Ercilla Spark cluster, look into `my_set_benchmark.sh` for more details
bm=TPCH
sf=100
bash ~/chenghao/spark-sql-perf/src/main/scripts/benchmark_sf_testing/my_set_benchmark.sh $bm $sf
```

2. Prepare the codebase for query generation (clone, compile and validate).

```bash
# add tpch-kit under `resources`
OS=MACOS # or LINUX, for both Spark and Postgres
bash examples/trace/1.setup_tpch.sh MACOS
```

3. Generate SparkSQLs. Check the example below

```bash
# bash examples/trace/spark/1.query_generation_tpch.sh <tpch-kit path> <query-out path> <#queries per template> <SF-100 by default>
bash examples/trace/spark/1.query_generation_tpch.sh $PWD/resources/tpch-kit $PWD/resources/tpch-kit/spark-sqls 3  
```

4. Generate configurations via LHS and BO. Check the example below

```bash
export PYTHONPATH="$PWD"
python examples/trace/spark/2.knob_sampling.py
```

5. Trigger trace collection.
   - an example of running single query in our Ercilla Spark cluster.
    ```bash
    export PYTHONPATH="$PWD"
    python examples/trace/spark/3.run_one.py
    ```
   - an example in the single-query environment
   - an example in the multi-query environment

TPCDS
-----     
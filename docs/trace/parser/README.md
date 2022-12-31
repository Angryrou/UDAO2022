@@ -0,0 +1,96 @@
Traces Parser
======

From the `trace.parser` package and examples, we describe and point out how we handle the trace parsing in different system settings, including

<!--ts-->
* [Spark](#spark)
* [PSQL](#psql)
<!--te-->

## Spark

We consider features from multiple channels including

| Parsing Sources            | For offline training                                          | For online inference                                                           |
|----------------------------|---------------------------------------------------------------|--------------------------------------------------------------------------------|
| Ch1: Query plan            | Rest API                                                      | Internal API                                                                   |
| Ch2: Input stats           | Input stages: Hive Metastore<br/> Subsequent stages: Rest API | Input stages: Hive Metastore<br/>Subsequent stages: Internal API (As AQE does) |
| Ch3: Machine states        | OS command `Nmon`                                             | OS command `Nmon`                                                              |
| Ch4: (optional) Hardware   | -                                                             | -                                                                              |
| Ch5: Configuration (Knobs) | Provided                                                      | Provided                                                                       |

1. For machine traces, we extract the `nmon` traces to `csv` files and extract needed information accordingly
```bash
# tpch_100, lhs
bash examples/trace/spark-parser/1.nmon_parser.sh hex1 examples/trace/spark/6.run_all_pressure_test/nmon/nmon examples/trace/spark-parser/outs/tpch_100_lhs/1.nmon
python -u examples/trace/spark-parser/1.mach_parser.py -b TPCH --sampling lhs --src-path examples/trace/spark-parser/outs/tpch_100_lhs/1.nmon --dst-path examples/trace/spark-parser/outs/tpch_100_lhs/1.mach --timezone-ahead 2
# tpch_100, bo
bash examples/trace/spark-parser/1.nmon_parser.sh hex1 examples/trace/spark/8.run_all_pressure_bo/TPCH/nmon/nmon examples/trace/spark-parser/outs/tpch_100_bo/1.nmon
python -u examples/trace/spark-parser/1.mach_parser.py -b TPCH --sampling bo --src-path examples/trace/spark-parser/outs/tpch_100_bo/1.nmon --dst-path examples/trace/spark-parser/outs/tpch_100_bo/1.mach --timezone-ahead 1

# tpcds_100, lhs
bash examples/trace/spark-parser/1.nmon_parser.sh hex2 examples/trace/spark/6.run_all_pressure_test/nmon/nmon examples/trace/spark-parser/outs/tpcds_100_lhs/1.nmon
python -u examples/trace/spark-parser/1.mach_parser.py -b TPCDS --sampling lhs --src-path examples/trace/spark-parser/outs/tpcds_100_lhs/1.nmon --dst-path examples/trace/spark-parser/outs/tpcds_100_lhs/1.mach --timezone-ahead 2
# tpcds_100, bo
bash examples/trace/spark-parser/1.nmon_parser.sh hex2 examples/trace/spark/8.run_all_pressure_bo/TPCDS/nmon/nmon examples/trace/spark-parser/outs/tpcds_100_bo/1.nmon
python -u examples/trace/spark-parser/1.mach_parser.py -b TPCDS --sampling bo --src-path examples/trace/spark-parser/outs/tpcds_100_bo/1.nmon --dst-path examples/trace/spark-parser/outs/tpcds_100_bo/1.mach --timezone-ahead 1
```

2. generate the tabular csv files
```bash
cd ~/chenghao/UDAO2022
export PYTHONPATH="$PWD"
# lhs,query-level
python -u examples/trace/spark-parser/2.tabular_downloader_query_level.py -b TPCH --sampling lhs \
--url-header http://10.0.0.1:18088/api/v1/applications/application_1663600377480 --lamda 100 \
--dst-path-header examples/trace/spark-parser/outs --url-suffix-start 3827 --url-suffix-end 83840
python -u examples/trace/spark-parser/2.tabular_generator_query_level.py -b TPCH --sampling lhs \
--dst-path-header examples/trace/spark-parser/outs --dst-path-matches "*query_traces*.parquet"
# lhs,stage-level
python -u examples/trace/spark-parser/3.tabular_downloader_stage_level.py -b TPCH --sampling lhs \
--url-header http://10.0.0.1:18088/api/v1/applications/application_1663600377480 --lamda 100 \
--dst-path-header examples/trace/spark-parser/outs --url-suffix-start 3827 --url-suffix-end 83840
# todo stage-level generate.
# bo,query-level
python -u examples/trace/spark-parser/2.tabular_downloader_query_level.py -b TPCH --sampling bo \
--url-header http://10.0.0.1:18088/api/v1/applications/application_1666935336888 --lamda 50 \
--dst-path-header examples/trace/spark-parser/outs --url-suffix-start 24 --url-suffix-end 19999
python -u examples/trace/spark-parser/2.tabular_generator_query_level.py -b TPCH --sampling bo \
--dst-path-header examples/trace/spark-parser/outs --dst-path-matches "*query_traces*.parquet"
# bo,stage-level

# prepare the data to CSV files
mkdir -p examples/trace/spark-parser/outs/prepared/tpch_100_query_traces
cp examples/trace/spark-parser/outs/prepared/tpch_100_*/2.*/query_traces/* examples/trace/spark-parser/outs/tpch_100_query_traces

export PYTHONPATH="$PWD"
# lhs,query-level
python -u examples/trace/spark-parser/2.tabular_downloader_query_level.py -b TPCDS --sampling lhs \
--url-header http://10.0.0.7:18088/api/v1/applications/application_1663600383047 --lamda 50 \
--dst-path-header examples/trace/spark-parser/outs --url-suffix-start 73995 --url-suffix-end 154025
python -u examples/trace/spark-parser/2.tabular_generator_query_level.py -b TPCDS --sampling lhs \
--dst-path-header examples/trace/spark-parser/outs --dst-path-matches "*query_traces*.parquet"
# lhs,stage-level
python -u examples/trace/spark-parser/3.tabular_downloader_stage_level.py -b TPCDS --sampling lhs \
--url-header http://10.0.0.7:18088/api/v1/applications/application_1663600383047 --lamda 50 \
--dst-path-header examples/trace/spark-parser/outs --url-suffix-start 73995 --url-suffix-end 154025

# bo,query-level
python -u examples/trace/spark-parser/2.tabular_downloader_query_level.py -b TPCDS --sampling bo \
--url-header http://10.0.0.7:18088/api/v1/applications/application_1667574472856 --lamda 1000 \
--dst-path-header examples/trace/spark-parser/outs --url-suffix-start 1 --url-suffix-end 19364
python -u examples/trace/spark-parser/2.tabular_generator_query_level.py -b TPCDS --sampling bo \
--dst-path-header examples/trace/spark-parser/outs --dst-path-matches "*query_traces*.parquet"
# bo,stage-level
```


3. maintain traces in SparkSQL

| Database         | Tables                                                                                                |
|------------------|-------------------------------------------------------------------------------------------------------|
| tpch_100_traces  | lhs_mach_traces, lhs_query_traces, lhs_stage_traces, bo_mach_traces, bo_query_traces, bo_stage_traces |
| tpcds_100_traces | lhs_mach_traces, lhs_query_traces, lhs_stage_traces, bo_mach_traces, bo_query_traces, bo_stage_traces |



### Ch1: Query plan

### Ch2: Inst stats



### Ch5: ...

## PSQL
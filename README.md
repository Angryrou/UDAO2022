# UDAO2022

## Dependency 

The python virtual environment
```bash
conda create -n <name> python=3.9
pip install -r requirements.txt
```

The supportive trace collection projects. Put or solflink those projects under `resources`

[our customized tpch-kit][1]
```bash
OS=MACOS # or LINUX, for both Spark and Postgres
bash examples/trace/1.setup_tpch.sh MACOS
```

[our customized spark-sql-perf][2]
```bash
cd resources/
git clone git@github.com:Angryrou/spark-sql-perf.git
cd spark-sql-perf
bin/run --help # testing env
sbt +package

# this should be different among different systems.
# an example of running in our cluster, look into `my_set_benchmark.sh` for more details
bm=TPCH
sf=100
bash ~/chenghao/spark-sql-perf/src/main/scripts/benchmark_sf_testing/my_set_benchmark.sh $bm $sf 
```
   
[1]: https://github.com/Angryrou/tpch-kit
[2]: https://github.com/Angryrou/spark-sql-perf

## Requirement

Rules in implementation:
1. use the variable type in the function name.
2. to be added
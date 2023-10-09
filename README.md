# UDAO2022

## Dependency 

The python virtual environment
```bash
# cpu
conda create -n <name> python=3.9
conda install pytorch torchvision torchaudio cpuonly -c pytorch # version 1.13.1
pip install -r requirements.txt

# gpu 
conda create -n udao-gpu python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia # version 1.13.1
pip install -r requirements-gpu.txt
conda install -c dglteam dgl-cuda11.6==0.9.1post1
conda install -c anaconda cudatoolkit=11 # to be able to run dgl's gpu version


```

The supportive trace collection projects. Put or solflink those projects under `resources`

[our customized tpch-kit][1]
```bash
OS=MACOS # or LINUX, for both Spark and Postgres
bash examples/trace/1.setup_tpch.sh $OS
```

[our customized tpcds-kit][2]
```bash
OS=MACOS # or LINUX, for both Spark and Postgres
bash examples/trace/2.setup_tpcds.sh $OS
```

[our customized spark-sql-perf][3]
```bash
cd resources/
git clone https://github.com/Angryrou/spark-sql-perf.git
cd spark-sql-perf
bin/run --help # testing env
sbt +package

# this should be different among different systems.
# an example of running in our cluster, look into `my_set_benchmark.sh` for more details
bm=TPCH
sf=100
bash $PWD/resources/spark-sql-perf/src/main/scripts/benchmark_sf_testing/my_set_benchmark.sh $bm $sf 
```
   
[1]: https://github.com/Angryrou/tpch-kit
[2]: https://github.com/Angryrou/tpcds-kit
[3]: https://github.com/Angryrou/spark-sql-perf

## Requirement

Rules in implementation:
1. use the variable type in the function name.
2. to be added

## Pre-commit hooks

Install pre-commit hooks
```bash
pre-commit install
```

Then every time you commit, the following pre-commit hooks are run:
- black: code formatter
- ruff: fast linter
- mypy: strict type checking
- isort: import sorter
- standard hooks from precommit: trailing whitespace, end of file newline, etc.

## Documentation

Install sphinx in your conda environment
```bash
conda install sphinx
```
Then go to the udao/docs directory and build the docs
```bash
cd docs
make html
```

You can then open the index.html file in the _build/html directory to view the documentation.
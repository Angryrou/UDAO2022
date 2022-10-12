OS=$1 # MACOS or LINUX

cd resources
# Clone our customized `tpcds-kit` adapted to SparkSQL
git clone git@github.com:Angryrou/tpcds-kit.git
cd tpcds-kit/tools
make OS=$OS

# to validate, the outputs are the same as queries from https://github.com/databricks/spark-sql-perf/blob/master/src/main/resources/tpch/queries
cd .. # to tpcds-kit
bash validate_sparksql_gen.sh # should not be any output.

cd .. # back to the home
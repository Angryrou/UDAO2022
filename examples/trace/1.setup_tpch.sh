OS=$1 # MACOS or LINUX

cd resources
# Clone our customized `tpch-kit` adapted to SparkSQL
git clone git@github.com:Angryrou/tpch-kit.git
cd tpch-kit/dbgen
make MACHINE=$OS DATABASE=POSTGRESQL

# to validate, the outputs are the same as queries from https://github.com/databricks/spark-sql-perf/blob/master/src/main/resources/tpch/queries
cd .. # to tpch-kit
bash validate_sparksql_gen.sh # should not be any output.

cd .. # back to the home
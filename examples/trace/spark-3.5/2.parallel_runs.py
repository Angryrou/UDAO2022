# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 28/05/2023
import itertools, os
import time, argparse, shutil

from trace.collect.framework import error_handler
from utils.common import BenchmarkUtils
from multiprocessing import Pool, Manager


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--local", type=int, default=0)

    def parse(self):
        return self.parser.parse_args()


def make_scripts(num_parallel,
                 knob_sign,
                 spath="/opt/hex_users/$USER/chenghao/spark-stage-tuning",
                 jpath="/opt/hex_users/$USER/spark-3.2.1-hadoop3.3.0/jdk1.8",
                 spark_home="/opt/hex_users/$USER/spark",
                 oplan_header="resources/tpch-kit/spark-sqls"):
    for t in TEMPLATES:
        q_sign = f"q{t}-1"
        out_header = f"{PG}/{knob_sign}/{q_sign}"
        for i in range(1, num_parallel + 1):
            name = f"TPCH100_PER_BM_{q_sign}_{num_parallel}_{i}"
            spark_script = f"""\
        
# knob_sign: {knob_sign}
# q_sign = {q_sign}
        
spath={spath}
jpath={jpath}
lpath={spath}/src/main/resources/log4j2.properties
name={name}

{spark_home}/bin/spark-submit \\
--class edu.polytechnique.cedar.spark.benchmark.RunTemplateQueryWithoutExtension \\
--name $name \\
--master "local[*]" \\
--deploy-mode client \\
--conf spark.executorEnv.JAVA_HOME=${{jpath}} \\
--conf spark.yarn.appMasterEnv.JAVA_HOME=${{jpath}} \\
--conf spark.default.parallelism=40 \\
--conf spark.sql.adaptive.enabled=true \\
--conf spark.sql.parquet.compression.codec=snappy \\
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \\
--conf spark.kryoserializer.buffer.max=512m \\
--conf spark.sql.adaptive.advisoryPartitionSizeInBytes={s1} \\
--conf spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin={s2} \\
--conf spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold={s3} \\
--conf spark.sql.autoBroadcastJoinThreshold={s4} \\
--conf spark.sql.parquet.compression.codec=snappy \\
--conf spark.sql.broadcastTimeout=10000 \\
--conf spark.rpc.askTimeout=12000 \\
--conf spark.shuffle.io.retryWait=60 \\
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \\
--conf spark.kryoserializer.buffer.max=512m \\
--driver-java-options "-Dlog4j.configuration=file:$lpath" \\
--conf spark.driver.extraClassPath=file://{spath}/benchmark-res/libs/mysql-connector-j-8.0.33.jar \\
--conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:log4j.properties" \\
--files "$lpath" \\
--jars {spark_home}/examples/jars/scopt_2.12-3.7.1.jar \\
$spath/target/scala-2.12/spark-stage-tuning_2.12-1.0-SNAPSHOT.jar \\
-b TPCH -t {t} -q 1 -s 1 -l {oplan_header}
            """ if LOCAL else f"""\
        
# knob_sign: {knob_sign}
# q_sign = {q_sign}
        
spath={spath}
jpath={jpath}
lpath={spath}/src/main/resources/log4j2.properties
name={name}

{spark_home}/bin/spark-submit \\
--class edu.polytechnique.cedar.spark.benchmark.RunTemplateQueryWithoutExtension \\
--name $name \\
--master yarn \\
--deploy-mode client \\
--conf spark.executorEnv.JAVA_HOME=${{jpath}} \\
--conf spark.yarn.appMasterEnv.JAVA_HOME=${{jpath}} \\
--conf spark.executor.memory=16g \\
--conf spark.executor.cores=5 \\
--conf spark.executor.instances=4 \\
--conf spark.default.parallelism=40 \\
--conf spark.reducer.maxSizeInFlight=48m \\
--conf spark.shuffle.sort.bypassMergeThreshold=200 \\
--conf spark.shuffle.compress=true \\
--conf spark.memory.fraction=0.6 \\
--conf spark.sql.inMemoryColumnarStorage.batchSize=10000 \\
--conf spark.sql.files.maxPartitionBytes=128MB \\
--conf spark.sql.shuffle.partitions=200 \\
--conf spark.sql.adaptive.coalescePartitions.parallelismFirst=false \\
--conf spark.yarn.am.cores=5 \\
--conf spark.yarn.am.memory=16g \\
--conf spark.sql.adaptive.enabled=true \\
--conf spark.sql.adaptive.advisoryPartitionSizeInBytes={s1} \\
--conf spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin={s2} \\
--conf spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold={s3} \\
--conf spark.sql.autoBroadcastJoinThreshold={s4} \\
--conf spark.sql.parquet.compression.codec=snappy \\
--conf spark.sql.broadcastTimeout=10000 \\
--conf spark.rpc.askTimeout=12000 \\
--conf spark.shuffle.io.retryWait=60 \\
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \\
--conf spark.kryoserializer.buffer.max=512m \\
--driver-java-options "-Dlog4j.configuration=file:$lpath" \\
--conf "spark.driver.extraJavaOptions=-Xms20g" \\
--conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:log4j.properties" \\
--files "$lpath" \\
--jars {spark_home}/examples/jars/scopt_2.12-3.7.1.jar \\
$spath/target/scala-2.12/spark-stage-tuning_2.12-1.0-SNAPSHOT.jar \\
-b TPCH -t {t} -q 1 -s 100 -l {oplan_header}
            """
            file_name = f"{knob_sign}_{q_sign}_{num_parallel}_{i}.sh"
            os.makedirs(f"{out_header}", exist_ok=True)
            with open(f"{out_header}/{file_name}", "w") as f:
                f.write(spark_script)
            print(f"script prepared for running {knob_sign}_{q_sign}")


def submit(q_sign, knob_sign, num_parallel, trial):
    script_path = f"{PG}/{knob_sign}/{q_sign}"
    file_name = f"{knob_sign}_{q_sign}_{num_parallel}_{trial}.sh"
    log_file = f"{script_path}/{knob_sign}_{q_sign}_{num_parallel}_{trial}.log"
    print(f"Thread {trial}: start running {q_sign}")
    os.system(f"bash {script_path}/{file_name}.sh > {log_file} 2>&1")
    name = f"TPCH100_PER_BM_{q_sign}_{num_parallel}_{trial}"
    json_file = f"{name}.json"
    assert (os.path.exists(json_file))
    shutil.move(json_file, f"{script_path}/{json_file}")


def run_parallel_queries(q_sign, knob_sign, num_parallel):
    if num_parallel == 1:
        submit(q_sign, knob_sign, num_parallel, num_parallel)
    else:
        pool = Pool(processes=num_parallel)
        for trial in range(1, num_parallel + 1):
            pool.apply_async(func=submit,
                             args=(q_sign, knob_sign, num_parallel, trial),
                             error_callback=error_handler)
            time.sleep(1)
        pool.close()
        pool.join()


def main():
    knob_sign = f"{s1}_{s2}_{s3}_{s4}"
    for num_parallel in [1, 3, 5]:
        make_scripts(num_parallel, knob_sign, spath, jpath, spark_home, oplan_header)

    for t in TEMPLATES:
        q_sign = f"q{t}-1"
        print(f"start working on {q_sign}")
        for num_parallel in [1, 3, 5]:
            run_parallel_queries(q_sign, knob_sign, num_parallel)


if __name__ == '__main__':

    args = Args().parse()

    BM = "TPCH"
    PG = "examples/trace/spark-3.5/playground/parallel_run"
    TEMPLATES = BenchmarkUtils.get(BM)
    N_TEMPLATES = len(TEMPLATES)
    LOCAL = False if args.local == 0 else True

    s1 = "64MB"
    s2 = "0.2"
    s3 = "0"
    s4 = "10MB"

    project_path = "/opt/hex_users/hex1/chenghao/spark-stage-tuning"

    if LOCAL:
        spath = "/Users/chenghao/ResearchHub/repos/spark-stage-tuning"
        spark_home = "$SPARK_HOME"
        jpath = "$JAVA_HOME"
        oplan_header = "/Users/chenghao/ResearchHub/repos/UDAO2022/resources/tpch-kit/spark-sqls"
    else:
        spath = "/opt/hex_users/$USER/chenghao/spark-stage-tuning"
        jpath = "/opt/hex_users/$USER/spark-3.2.1-hadoop3.3.0/jdk1.8"
        spark_home = "/opt/hex_users/$USER/spark"
        oplan_header = "/opt/hex_users/$USER/chenghao/UDAO2022/resources/tpch-kit/spark-sqls"

    main()

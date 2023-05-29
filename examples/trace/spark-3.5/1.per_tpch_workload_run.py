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


args = Args().parse()

BM = "TPCH"
PG = "examples/trace/spark-3.5/playground/tpch/per_workload"
REPS = 3
TEMPLATES = BenchmarkUtils.get(BM)
N_TEMPLATES = len(TEMPLATES)
LOCAL = False if args.local == 0 else True

s1_list = ["32MB", "64MB", "128MB"]
s2_list = ["0.1", "0.2", "0.5"]
s3_list = ["0", "33554432", "67108864", "134217728"]  # ["0", "32MB", "64MB", "128MB"]
s4_list = ["10MB", "20MB", "50MB", "100MB", "200MB"]
project_path = "/opt/hex_users/hex1/chenghao/spark-stage-tuning"
n_processes = 8


def make_scripts(s1, s2, s3, s4,
                 spath="/opt/hex_users/$USER/chenghao/spark-stage-tuning",
                 jpath="/opt/hex_users/$USER/spark-3.2.1-hadoop3.3.0/jdk1.8",
                 spark_home="/opt/hex_users/$USER/spark",
                 oplan_header="resources/tpch-kit/spark-sqls"):
    knob_sign = f"{s1}_{s2}_{s3}_{s4}"
    out_header = f"{PG}/{knob_sign}"
    for t in TEMPLATES:
        q_sign = f"q{t}-1"
        name = f"TPCH100_PER_BM_{knob_sign}_{q_sign}"
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
        file_name = f"{knob_sign}_{q_sign}.sh"
        os.makedirs(f"{out_header}", exist_ok=True)
        with open(f"{out_header}/{file_name}", "w") as f:
            f.write(spark_script)
        print(f"script prepared for running {knob_sign}_{q_sign}")
    return knob_sign


def submit(q_sign, knob_sign, trial, current_cores, cores):
    script_path = f"{PG}/{knob_sign}"
    log_file = f"{PG}/{knob_sign}/{knob_sign}_{q_sign}.log.{trial + 1}"
    print(f"Thread {q_sign}: start running")
    start = time.time()
    os.system(f"bash {script_path}/{knob_sign}_{q_sign}.sh > {log_file} 2>&1")

    json_file = f"TPCH100_PER_BM_{knob_sign}_{q_sign}.json"
    assert (os.path.exists(json_file))
    shutil.move(json_file, f"{script_path}/{json_file}.{trial + 1}")
    with lock:
        current_cores.value -= cores
        print(f"Thread {q_sign}: finish running, takes {time.time() - start}s, current_cores={current_cores.value}")


def run_workload(knob_sign, trial):
    script_path = f"{PG}/{knob_sign}"

    finished = True
    for t in TEMPLATES:
        q_sign = f"q{t}-1"
        json_file = f"TPCH100_PER_BM_{knob_sign}_{q_sign}.json"
        if not os.path.exists(f"{script_path}/{json_file}.{trial + 1}"):
            finished = False
            break
    if finished:
        print(f"{script_path} existed.")
        return

    cores = 25  # (4 + 1) * 5
    cluster_cores = 150
    submit_index = 0
    current_cores = m.Value("i", 0)
    pool = Pool(processes=n_processes)
    while submit_index < N_TEMPLATES:
        with lock:
            if cores + current_cores.value <= cluster_cores:
                q_sign = f"q{submit_index + 1}-1"
                current_cores.value += cores
                if_submit = True
                print(f"Main Process: submit {q_sign}, current_cores = {current_cores.value}")
            else:
                if_submit = False
        if if_submit:
            pool.apply_async(func=submit,
                             args=(q_sign, knob_sign, trial, current_cores, cores),
                             error_callback=error_handler)
            submit_index += 1
        time.sleep(1)
    pool.close()
    pool.join()


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

m = Manager()
lock = m.RLock()
for s1, s2, s3, s4 in itertools.product(s1_list, s2_list, s3_list, s4_list):
    knob_sign = make_scripts(s1, s2, s3, s4, spath, jpath, spark_home, oplan_header)

for trial in range(REPS):
    for s1, s2, s3, s4 in itertools.product(s1_list, s2_list, s3_list, s4_list):
        knob_sign = f"{s1}_{s2}_{s3}_{s4}"
        print(f"--- start {knob_sign}, trial {trial + 1}")
        start_time = time.time()
        run_workload(knob_sign, trial)
        print(f"--- finish {knob_sign}, trial {trial + 1}, cost {time.time() - start_time}")

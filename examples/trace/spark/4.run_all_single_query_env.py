# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: an example of running queries in single query env at Ercilla Spark cluster:
#              we run 22 TPCH queries in sequential with the default configuration,
#              and we trigger the system monitor in each worker nodes.
#
# Created at 9/23/22
import os
import time

from trace.collect.framework import SparkCollect
from utils.common import JsonUtils
from utils.data.configurations import SparkKnobs, KnobUtils
from utils.data.feature import NmonUtils

SEED = 42
workers = ["node2", "node3", "node4", "node5", "node6"]
OUT_HEADER = "examples/trace/spark/4.run_all_single_query_env"
qid = "1"
REMOTE_HEADER = "~/chenghao"

spark_knobs = SparkKnobs(meta_file="resources/knob-meta/spark.json")
knobs = spark_knobs.knobs
conf_dict = {k.name: k.default for k in knobs}
JsonUtils.print_dict(conf_dict)

spark_collect = SparkCollect(
    benchmark="TPCH",
    scale_factor=100,
    spark_knobs=spark_knobs,
    query_header="resources/tpch-kit/spark-sqls",
    seed=SEED
)
knob_dict = spark_knobs.conf2knobs(conf_dict)
knob_sign = KnobUtils.knobs2sign([knob_dict[k.id] for k in knobs], knobs)

# prepare scripts for running
file_names = [
    spark_collect.save_one_script(
        tid=str(tid),
        qid="1",
        conf_dict=conf_dict,
        out_header=OUT_HEADER
    )
    for tid in range(1, 23)
]

nmon_reset = NmonUtils.nmon_remote_reset(workers, remote_header=REMOTE_HEADER)
nmon_start = NmonUtils.nmon_remote_start(workers, remote_header=REMOTE_HEADER, name_suffix="", duration=3600, freq=1)
nmon_stop = NmonUtils.nmon_remote_stop(workers)
nmon_agg = NmonUtils.nmon_remote_agg(workers, remote_header=REMOTE_HEADER, local_header=OUT_HEADER, name_suffix="")

try:
    os.system(nmon_reset)
    os.system(nmon_start)
    time.sleep(10)
    for file_name in file_names:
        start = time.time()
        os.system(f"bash {OUT_HEADER}/{file_name} > {OUT_HEADER}/{file_name}.log 2>&1")
        print(f"finished running for {file_name}, takes {time.time() - start}s")
    os.system(nmon_stop)
    os.system(nmon_agg)
except Exception as e:
    print(f"failed to run due to {e}")

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

SEED = 42
workers = ["node2", "node3", "node4", "node5", "node6"]
out_path = "examples/trace/spark/4.run_all_single_query_env"
qid = "1"

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

# prepare scripts to start and stop nmon over all the worker nodes.
nmon_reset, nmon_start, nmon_stop, nmon_agg = spark_collect.make_commands_worker_system_states(
    workers=workers,
    duration=10,
    tmp_path="~/chenghao",
    out_path=out_path,
    name_suffix=""
)
print("nmon commands prepared.")

# prepare scripts for running
for tid in range(1, 23):
    file_name = f"q{tid}-{qid}.sh"
    spark_script = spark_collect.make_script(
        tid=tid,
        qid=qid,
        knob_sign=knob_sign,
        conf_dict=conf_dict,
        out_path=out_path
    )
    print(spark_script)
    with open(f"{out_path}/{file_name}", "w") as f:
        f.write(spark_script)
    print(f"script {tid}-{qid} prepared for running")

# try:
#     start = time.time()
#     os.system(f"bash {out_path}/{file_name}")
#     print(f"finished running, takes {time.time() - start}s")
# except Exception as e:
#     print(f"failed to run due to {e}")

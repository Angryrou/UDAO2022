# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: an example of how to run one query with a configuration
#
# Created at 9/20/22
import os
import time

from trace.collect.framework import SparkCollect
from utils.common import JsonUtils
from utils.data.configurations import SparkKnobs, KnobUtils

SEED = 42
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
out_header = "examples/trace/spark/3.run_one_outs"

file_name = spark_collect.save_one_script(
    tid="1",
    qid="1",
    conf_dict=conf_dict,
    out_header=out_header,
    if_aqe=False
)

try:
    start = time.time()
    os.system(f"bash {out_header}/{file_name}")
    print(f"finished running, takes {time.time() - start}s")
except Exception as e:
    print(f"failed to run due to {e}")

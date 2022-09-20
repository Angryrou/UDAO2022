# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: an example of how to run one query with a configuration
#
# Created at 9/20/22
import os
from trace.collect.framework import SparkCollect
from utils.data.configurations import SparkKnobs, KnobUtils
from utils.common import JsonUtils

SEED=42
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
out_path = "examples/trace/spark/3.run_one_outs"
tid, qid = "1", "1"
file_name = f"q{tid}-{qid}.sh"
spark_script = spark_collect.make_script(
    tid="1",
    qid="1",
    knob_sign=knob_sign,
    conf_dict=conf_dict,
    out_path=out_path
)

with open(f"{out_path}/{file_name}", "w") as f:
    f.write(spark_script)

try:
    os.system(f"bash {out_path}/{file_name}")
except Exception as e:
    print(f"failed to run due to {e}")
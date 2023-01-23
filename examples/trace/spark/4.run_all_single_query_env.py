# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: an example of running queries in single query env at Ercilla Spark cluster:
#              we run 22 TPCH queries in sequential with the default configuration,
#              and we trigger the system monitor in each worker nodes.
#
# Created at 9/23/22
import argparse
import os
import time

from trace.collect.framework import SparkCollect
from utils.common import JsonUtils, BenchmarkUtils
from utils.data.configurations import SparkKnobs, KnobUtils
from utils.data.feature import NmonUtils

class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("-k", "--knob-meta-file", type=str, default="resources/knob-meta/spark.json")
        self.parser.add_argument("-s", "--seed", type=int, default=42)
        self.parser.add_argument("-q", "--query-header", type=str, default="resources/tpch-kit/spark-sqls")
        self.parser.add_argument("--num-templates", type=int, default=22)
        self.parser.add_argument("--if-aqe", type=int, default=0)
        self.parser.add_argument("--worker", type=str, default=None)

    def parse(self):
        return self.parser.parse_args()

args = Args().parse()
seed = args.seed
benchmark = args.benchmark
query_header = args.query_header
if_aqe = False if args.if_aqe == 0 else True
qid = "1"
OUT_HEADER = "examples/trace/spark/4.run_all_single_query_env"
REMOTE_HEADER = "~/chenghao"
if args.worker is None:
    workers = BenchmarkUtils.get_workers(benchmark)
else:
    workers = BenchmarkUtils.get_workers(args.worker)

spark_knobs = SparkKnobs(meta_file=args.knob_meta_file)
knobs = spark_knobs.knobs
conf_dict = {k.name: k.default for k in knobs}
JsonUtils.print_dict(conf_dict)

spark_collect = SparkCollect(
    benchmark=benchmark,
    scale_factor=100,
    spark_knobs=spark_knobs,
    query_header=query_header,
    seed=seed
)
knob_dict = spark_knobs.conf2knobs(conf_dict)
knob_sign = KnobUtils.knobs2sign([knob_dict[k.id] for k in knobs], knobs)

# prepare scripts for running
templates = BenchmarkUtils.get(benchmark)
assert len(templates) == args.num_templates
file_names = [
    spark_collect.save_one_script(
        tid=tid,
        qid="1",
        conf_dict=conf_dict,
        out_header=OUT_HEADER,
        if_aqe=if_aqe
    )
    for tid in templates
]

nmon_reset = NmonUtils.nmon_remote_reset(workers, remote_header=REMOTE_HEADER)
nmon_start = NmonUtils.nmon_remote_start(workers, remote_header=REMOTE_HEADER, name_suffix="", counts=3600, freq=1)
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

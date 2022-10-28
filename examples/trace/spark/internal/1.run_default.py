# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: run default queries for a benchmark with and without AQE.
#
# python examples/trace/spark/internal/1.run_default.py --debug 1 --if-aqe 0
# python examples/trace/spark/internal/1.run_default.py --debug 1 --if-aqe 1
# python examples/trace/spark/internal/1.run_default.py --debug 0 --if-aqe 0 --num-trials 3
# python examples/trace/spark/internal/1.run_default.py --debug 0 --if-aqe 1 --num-trials 3
#
# Created at 10/28/22

import argparse
import os
import time

import numpy as np

from trace.collect.framework import SparkCollect
from utils.common import JsonUtils, BenchmarkUtils, PickleUtils, TimeUtils
from utils.data.configurations import SparkKnobs, KnobUtils


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("-k", "--knob-meta-file", type=str, default="resources/knob-meta/spark.json")
        self.parser.add_argument("-s", "--seed", type=int, default=42)
        self.parser.add_argument("-q", "--query-header", type=str, default="resources/tpch-kit/spark-sqls")
        self.parser.add_argument("--script-header", type=str, default="resources/scripts/tpch-lhs")
        self.parser.add_argument("--out-header", type=str, default="examples/trace/spark/internal/1.run_default")
        self.parser.add_argument("--num-templates", type=int, default=22)
        self.parser.add_argument("--num-trials", type=int, default=3)
        self.parser.add_argument("--debug", type=int, default=0)
        self.parser.add_argument("--if-aqe", type=int, default=0)

    def parse(self):
        return self.parser.parse_args()


args = Args().parse()
seed = args.seed
benchmark = args.benchmark
assert benchmark.lower() == "tpch", f"unsupported benchmark {benchmark}"
query_header = args.query_header
if_aqe = False if args.if_aqe == 0 else True
out_header = f"{args.out_header}/{benchmark}_AQE_{'enabled' if if_aqe else 'disabled'}"
num_templates = args.num_templates
num_trials = args.num_trials
workers = BenchmarkUtils.get_workers("debug")
debug = False if args.debug == 0 else True


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
assert len(templates) == num_templates
if debug:
    num_templates = 2
    templates = templates[:2]
    out_header += "_debug"

file_names = [
    spark_collect.save_one_script(
        tid=tid,
        qid="1",
        conf_dict=conf_dict,
        out_header=out_header,
        if_aqe=if_aqe
    )
    for tid in templates
]


def flush_all():
    os.system("sync")
    for worker in workers:
        os.system(f"ssh {worker} sync")


dts = np.zeros((num_templates, num_trials))
for i, file_name in enumerate(file_names):
    for j in range(args.num_trials):
        flush_all()
        time.sleep(2)
        start = time.time()
        os.system(f"bash {out_header}/{file_name} > {out_header}/{file_name}_trial_{j + 1}.log 2>&1")
        dt = time.time() - start
        dts[i, j] = dt
        print(f"{file_name}, trial {j + 1}: {dt:.3f}s")

try:
    stats_name = f"durations_{num_templates}x{num_trials}_{TimeUtils.get_current_iso()}.pkl"
    PickleUtils.save(dts, out_header, stats_name)
    print(f"dts are saved at {out_header}/{stats_name}")
except Exception as e:
    print(e)
    raise

dts_mu = dts.mean(1)
dts_std = dts.std(1)

print(f"qid \t duration (s) with AQE {'enabled' if if_aqe else 'disabled'}")
for i in range(num_templates):
    qid = i + 1
    print(f"{qid}-1\t{dts_mu[i]:.3f} +- {dts_std[i]:.3f} s")

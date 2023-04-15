# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 15/04/2023
import argparse

from trace.collect.framework import SparkCollect
from utils.common import BenchmarkUtils

from utils.data.collect import sql_exec
from utils.data.configurations import SparkKnobs

class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("-k", "--knob-meta-file", type=str, default="resources/knob-meta/spark.json")
        self.parser.add_argument("-q", "--query-header", type=str, default="resources/tpch-kit/spark-sqls")
        self.parser.add_argument("--seed", type=int, default=42)
        self.parser.add_argument("--if-aqe", type=int, default=0)
        self.parser.add_argument("--n-trials", type=int, default=8)
        self.parser.add_argument("--q-sign", type=str, default="q3-1")

    def parse(self):
        return self.parser.parse_args()


args = Args().parse()
bm, sf = args.benchmark.lower(), int(args.scale_factor)
query_header = args.query_header
seed = args.seed
debug = False
if_aqe = False if args.if_aqe == 0 else True
aqe_sign = "aqe_on" if if_aqe else "aqe_off"
q_sign = args.q_sign
n_trials = args.n_trials

workers = BenchmarkUtils.get_workers("hex1")
spark_knobs = SparkKnobs(meta_file=args.knob_meta_file)
knobs = spark_knobs.knobs
conf_dict = {k.name: k.default for k in knobs}

script_header = f"examples/trace/spark/internal/2.knob_hp_tuning/{bm}_{aqe_sign}/{q_sign}"
spark_collect = SparkCollect(
    benchmark=bm,
    scale_factor=sf,
    spark_knobs=spark_knobs,
    query_header=query_header,
    seed=seed
)

dts = sql_exec(spark_collect, conf_dict, n_trials, workers,
               out_header=script_header, debug=debug, q_sign=q_sign, if_aqe=if_aqe)

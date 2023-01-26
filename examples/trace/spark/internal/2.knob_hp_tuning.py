# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: tune SQL knobs
#
# Created at 10/28/22

import argparse
import re

from trace.collect.sampler import LHSSampler
from utils.common import BenchmarkUtils
from utils.data.configurations import SparkKnobs
from utils.parameters import VarTypes
from utils.data.collect import run_q_confs

import numpy as np


class InnerKnobs(SparkKnobs):
    def __init__(self, meta_file, knob_type, seed):
        super().__init__(meta_file)
        if knob_type == "res":
            self.inner_knobs = [k for k in self.knobs if k.id in ["k1", "k2", "k3"]]
        elif knob_type == "sql":
            self.inner_knobs = [k for k in self.knobs if k.id in ["k4", "s1", "s2", "s3", "s4"]]
        else:
            raise Exception(f"unsupported knob_type {knob_type}")
        self.knob_type = knob_type
        self.default_conf_dict = {k.name: k.default for k in self.knobs}
        self.seed = seed

    def inner_knob2conf(self, inner_knob_df):
        df = inner_knob_df.copy()
        for k in self.inner_knobs:
            if k.type == VarTypes.INTEGER:
                if k.scale == "linear":
                    df[k.id] = k.factor * df[k.id]
                elif k.scale == "log":
                    df[k.id] = k.factor * (k.base ** df[k.id])
                else:
                    raise Exception(f"unsupported scale attribute {k.scale}")

        if self.knob_type == "res":
            # default k4=2
            df["k4"] = df["k2"] * df["k3"] * 2
            assert list(df.columns) == ["k1", "k2", "k3", "k4"]
        elif self.knob_type == "sql":
            # default k2 = 5, k3 = 4
            pc = 5 * 4 * df["k4"]  # get the partition count
            df["k4"] = 5 * 4 * 2
            df["s4"] = [pc_ if s4_ else 2048 for s4_, pc_ in zip(df["s4"], pc)]
            assert list(df.columns) == ["k4", "s1", "s2", "s3", "s4"]
        else:
            raise Exception(f"unsupported knob_type {knob_type}")

        df_cols = list(df.columns)
        for k in self.knobs:
            if k.id in ["k1", "k2", "k3", "k4", "k5", "s1", "s2", "s3"] and k.id in df_cols:
                df[k.id] = df[k.id].astype(str) + (k.unit if k.unit is not None else "")
        conf_df = df.rename(columns={k.id: k.name for k in self.knobs if k.id in df_cols})

        for k, v in self.default_conf_dict.items():
            if k not in conf_df.columns:
                conf_df[k] = v

        conf_df = conf_df[[k.name for k in self.knobs]]
        return conf_df

    def get_lhs_samples(self, n_lhs):
        lhs_sampler = LHSSampler(self.inner_knobs, seed=self.seed)
        samples, inner_knob_df = lhs_sampler.get_samples(n_lhs, debug=True)
        conf_df = self.inner_knob2conf(inner_knob_df)
        return samples, inner_knob_df, conf_df


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("-k", "--knob-meta-file", type=str, default="resources/knob-meta/spark.json")
        self.parser.add_argument("-s", "--seed", type=int, default=42)
        self.parser.add_argument("-q", "--query-header", type=str, default="resources/tpch-kit/spark-sqls")
        self.parser.add_argument("--out-header", type=str, default="examples/trace/spark/internal/2.knob_hp_tuning")
        self.parser.add_argument("--q-sign", type=str, default="1")
        self.parser.add_argument("--knob-type", type=str, default="res", help="res|sql")
        self.parser.add_argument("--num-conf-lhs", type=int, default=20)
        self.parser.add_argument("--num-conf-bo", type=int, default=20)
        self.parser.add_argument("--num-trials", type=int, default=3)
        self.parser.add_argument("--debug", type=int, default=0)
        self.parser.add_argument("--worker", type=str, default="debug")
        self.parser.add_argument("--if-aqe", type=int, default=1)

    def parse(self):
        return self.parser.parse_args()


if __name__ == '__main__':
    args = Args().parse()
    seed = args.seed
    np.random.seed(seed)
    benchmark = args.benchmark.lower()
    assert benchmark.lower() == "tpch", f"unsupported benchmark {benchmark}"
    query_header = args.query_header
    q_sign = BenchmarkUtils.extract_sampled_q_sign(benchmark, args.q_sign)

    if_aqe = False if args.if_aqe == 0 else True
    out_header = f"{args.out_header}/{benchmark.lower()}_aqe_{'on' if if_aqe else 'off'}/{q_sign}"
    knob_type = args.knob_type
    assert knob_type in ["res", "sql"], f"unsupported knob_type {knob_type}"
    n_lhs = args.num_conf_lhs
    n_bo = args.num_conf_bo
    n_trials = args.num_trials
    debug = False if args.debug == 0 else True
    is_aqe = True
    workers = BenchmarkUtils.get_workers(args.worker)

    print(f"1. get {n_lhs} configurations via LHS")
    spark_knobs = InnerKnobs(meta_file=args.knob_meta_file, knob_type=knob_type, seed=seed)
    samples, inner_knob_df, conf_df = spark_knobs.get_lhs_samples(n_lhs)
    conf_df.to_csv(f"{out_header}/lhs_{knob_type}.csv", index=None, header=True)
    
    print(inner_knob_df.to_string())
    print(conf_df.to_string())
    print()

    print(f"2. run {n_lhs} objective values corresponding to the configurations")
    objs = run_q_confs(benchmark, 100, spark_knobs, query_header, out_header, seed, workers, n_trials, debug, q_sign,
                       conf_df, if_aqe=if_aqe)
    print(objs)
    print()
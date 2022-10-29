# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: tune SQL knobs
#
# Created at 10/28/22

import argparse
import os
import time

from trace.collect.framework import SparkCollect
from trace.collect.sampler import LHSSampler
from utils.common import BenchmarkUtils
from utils.data.configurations import SparkKnobs
from utils.parameters import VarTypes

import numpy as np

def flush_all(workers):
    os.system("sync")
    for worker in workers:
        os.system(f"ssh {worker} sync")

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
        self.parser.add_argument("--target-query", type=str, default="1")
        self.parser.add_argument("--knob-type", type=str, default="res", help="res|sql")
        self.parser.add_argument("--num-conf-lhs", type=int, default=20)
        self.parser.add_argument("--num-conf-bo", type=int, default=20)
        self.parser.add_argument("--num-trials", type=int, default=3)
        self.parser.add_argument("--debug", type=int, default=0)

    def parse(self):
        return self.parser.parse_args()


if __name__ == '__main__':

    args = Args().parse()
    seed = args.seed
    benchmark = args.benchmark
    assert benchmark.lower() == "tpch", f"unsupported benchmark {benchmark}"
    query_header = args.query_header
    out_header = f"{args.out_header}/{benchmark}_AQE_enabled"
    tid = args.target_query
    knob_type = args.knob_type
    assert knob_type in ["res", "sql"], f"unsupported knob_type {knob_type}"
    n_lhs = args.num_conf_lhs
    n_bo = args.num_conf_bo
    n_trials = args.num_trials
    debug = False if args.debug == 0 else True
    is_aqe = True
    workers = BenchmarkUtils.get_workers("debug")

    print(f"1. get {n_lhs} configurations via LHS")
    spark_knobs = InnerKnobs(meta_file=args.knob_meta_file, knob_type=knob_type, seed=seed)
    samples, inner_knob_df, conf_df = spark_knobs.get_lhs_samples(n_lhs)

    print(inner_knob_df.to_string())
    print(conf_df.to_string())
    print()

    spark_collect = SparkCollect(
        benchmark=benchmark,
        scale_factor=100,
        spark_knobs=spark_knobs,
        query_header=query_header,
        seed=seed
    )

    def sql_exec(spark_collect, tid, conf_dict, n_trials, workers, out_header):
        """return a list of dts in `n_trials`"""

        # prepare the scripts for running.
        out = f"{out_header}/{tid}-1"
        file_name = spark_collect.save_one_script(tid, "1", conf_dict, out_header=out, if_aqe=True)

        # check if the results has been run already
        res_file = f"{out}/{file_name}.dts"
        if os.path.exists(res_file):
            try:
                with open(res_file) as f:
                    dts = [float(dt_str) for dt_str in f.readlines()[0].split(",")]
                assert len(dts) == n_trials
                print(f"{res_file} has been found!")
                return dts
            except:
                print(f"{res_file} is not properly generated")
        dts = []
        for i in range(n_trials):
            flush_all(workers)
            time.sleep(1)
            start = time.time()
            os.system(f"bash {out}/{file_name} > {out}/{file_name}_trial_{i + 1}.log 2>&1")
            dts.append(time.time() - start)
            print(f"{file_name}, trial {i + 1}, {dts[i]:.3f}s")
        with open(f"{out}/{file_name}.dts", "w") as f:
            f.write(",".join([f"{dt:.3f}" for dt in dts]))
        return dts

    print(f"2. run {n_lhs} objective values corresponding to the configurations")

    objs = []
    for conf_dict in conf_df.to_dict("records"):
        dts = sql_exec(spark_collect, tid, conf_dict, n_trials, workers, out_header)
        objs.append(sum(dts) / n_trials)
    objs = np.array(objs)
    print(objs)
    print()

    # print(f"3. get {n_bo} configurations via BO...")
    # print(f"3.1 parse and normalize all parameters to 0-1")
    # knob_df2 = spark_knobs.df_conf2knob(conf_df)
    # samples2 = KnobUtils.knob_normalize(knob_df2, knobs)
    # assert (knob_df2 == knob_df).all().all()
    # assert (knob_df2 == KnobUtils.knob_denormalize(samples2, knobs)).all().all()
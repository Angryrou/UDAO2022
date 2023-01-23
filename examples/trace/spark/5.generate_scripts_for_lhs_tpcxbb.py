# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: genearte all the scripts for running Spark SQLs with the LHS configurations
# be sure that the spark-sqls are prepared in advance. E.g.,
# `bash examples/trace/spark/1.query_generation_tpch.sh $PWD/resources/tpch-kit $PWD/resources/tpch-kit/spark-sqls 4545`
#
# Created at 9/23/22
import argparse
import os
import time
import random

import numpy as np
from multiprocessing import Pool

import pandas as pd

from trace.collect.framework import SparkCollect
from utils.common import PickleUtils, BenchmarkUtils
from utils.data.configurations import SparkKnobs


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCxBB")
        self.parser.add_argument("-k", "--knob-meta-file", type=str, default="resources/knob-meta/spark.json")
        self.parser.add_argument("-s", "--seed", type=int, default=42)
        self.parser.add_argument("--query-header", type=str, default=".")
        self.parser.add_argument("--script-header", type=str, default="resources/scripts/tpcxbb-lhs")
        self.parser.add_argument("--cache-header", type=str, default="examples/trace/spark/cache")
        self.parser.add_argument("--num-templates", type=int, default=30)
        self.parser.add_argument("--num-processes", type=int, default=6)
        self.parser.add_argument("--if-aqe", type=int, default=0)

    def parse(self):
        return self.parser.parse_args()


if __name__ == '__main__':

    args = Args().parse()

    seed = args.seed
    benchmark = args.benchmark
    query_header = args.query_header
    script_header = args.script_header
    cache_header = os.path.join(args.cache_header, benchmark.lower())
    n_templates = args.num_templates
    n_processes = args.num_processes
    if_aqe = False if args.if_aqe == 0 else True
    templates = BenchmarkUtils.get(benchmark)
    assert n_templates == len(templates)

    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(script_header, exist_ok=True)
    os.makedirs(cache_header, exist_ok=True)
    spark_knobs = SparkKnobs(meta_file=args.knob_meta_file)
    knobs = spark_knobs.knobs
    spark_collect = SparkCollect(
        benchmark=benchmark,
        scale_factor=100,
        spark_knobs=spark_knobs,
        query_header=query_header,
        seed=seed
    )

    # TPCxBB
    # 30 templates, 100 queries per template, 5% offline & 95% online
    # each online query: 5 shared confs + 5 indp confs
    # each offline query: 48 shared confs (include the 5 shared with online) + 333 indp confs
    qpt = 100
    try:
        conf_df_dict = PickleUtils.load(cache_header, f"lhs_{n_templates}x{qpt}.pkl")
        print(f"found conf_df_dict...")
    except:
        print(f"not found conf_df_dict, start generating...")
        qpt_off = 5
        qpt_on = 95
        query_dict = {tid: list(range(1, 101)) for tid in templates}
        query_dict_off = {}
        query_dict_on = {}
        qids = list(range(1, 101))
        for tid in templates:
            random.shuffle(qids)
            query_dict_off[tid] = qids[:5]
            query_dict_on[tid] = qids[5:]

        knob_df_dict = {}
        conf_df_dict = {}
        start1 = time.time()
        print("1. generate 5 shared configurations among ALL queries (including online and offline)")
        N_on_shared = 5

        knob_df = spark_collect.lhs_sampler.get_samples(N_on_shared, random_state=0)
        for tid, qids in query_dict.items():
            for qid in qids:
                knob_df_dict[(tid, qid)] = knob_df

        print("2. generate 48-5=43 shared configurations among ALL offline queries)")
        N_off_shared = 48
        knob_df = spark_collect.lhs_sampler.get_samples(N_off_shared - N_on_shared, random_state=0)
        for tid, qids in query_dict_off.items():
            for qid in qids:
                knob_df_dict[(tid, qid)] = pd.concat([knob_df_dict[(tid, qid)], knob_df])

        print("3. generate 333 indpt configurations of ALL offline queries")
        N_off_indpt = 333
        for i, (tid, qids) in enumerate(query_dict_off.items()):
            for qid in qids:
                knob_df = spark_collect.lhs_sampler.get_samples(N_off_indpt, random_state=(i+1) * qid + 1)
                knob_df_dict[(tid, qid)] = pd.concat([knob_df_dict[(tid, qid)], knob_df])

        print("4. generate 5 indpt configurations of ALL online queries")
        N_on_indpt = 5
        for i, (tid, qids) in enumerate(query_dict_on.items()):
            for qid in qids:
                # conf_df = spark_knobs.df_knob2conf(
                #     spark_collect.lhs_sampler.get_samples(N_on_indpt, random_state=(i+1) * qid + 1))
                knob_df = spark_collect.lhs_sampler.get_samples(N_on_indpt, random_state=(i+1) * qid + 1)
                knob_df_dict[(tid, qid)] = pd.concat([knob_df_dict[(tid, qid)], knob_df])

        print(f"convert knob_df to conf_df")
        conf_df_dict = {k: spark_knobs.df_knob2conf(v) for k, v in knob_df_dict.items()}
        print(f"finished lhs configuration generation, cost {time.time() - start1}s")
        PickleUtils.save(conf_df_dict, cache_header, f"lhs_{n_templates}x{qpt}.pkl")

    arg_list = [(str(tid), str(qid), conf_dict, f"{script_header}/{tid}", if_aqe)
                for (tid, qid), conf_df in conf_df_dict.items()
                for conf_dict in conf_df.to_dict("records")]

    start2 = time.time()
    with Pool(processes=n_processes) as pool:
        res = pool.starmap_async(spark_collect.save_one_script, arg_list)
        res.get()
    print(f"prepared scripts for all queries, cost {time.time() - start2}s")
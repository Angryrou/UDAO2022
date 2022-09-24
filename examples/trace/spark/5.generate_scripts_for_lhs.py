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

import numpy as np
from multiprocessing import Pool

from trace.collect.framework import SparkCollect
from utils.data.configurations import SparkKnobs


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("-k", "--knob-meta-file", type=str, default="resources/knob-meta/spark.json")
        self.parser.add_argument("-s", "--seed", type=int, default=42)
        self.parser.add_argument("-q", "--query-header", type=str, default="resources/tpch-kit/spark-sqls")
        self.parser.add_argument("-o", "--out-header", type=str, default="resources/scripts/tpch-lhs")
        self.parser.add_argument("--num-templates", type=int, default=22)
        self.parser.add_argument("--num-queries-per-template", type=int, default=3637)
        self.parser.add_argument("--num-processes", type=int, default=6)

    def parse(self):
        return self.parser.parse_args()


if __name__ ==  '__main__':

    args = Args().parse()

    seed = args.seed
    query_header = args.query_header
    out_header = args.out_header
    n_templates = args.num_templates
    n_processes = args.num_processes
    # the number of queries per template for LHS, 100K / 22 * 0.8 = 3637
    # Each template we generate 4545 queries and configurations.
    # Each template we reserve 3637(80%), 454(10%), 454(10%) configurations respectively for LHS, BO(latency), BO(cost)
    qpt = args.num_queries_per_template

    np.random.seed(seed)
    os.makedirs(out_header, exist_ok=True)
    spark_knobs = SparkKnobs(meta_file=args.knob_meta_file)
    knobs = spark_knobs.knobs

    spark_collect = SparkCollect(
        benchmark="TPCH",
        scale_factor=100,
        spark_knobs=spark_knobs,
        query_header=query_header,
        seed=seed
    )

    start1 = time.time()

    for tid in range(1, n_templates + 1):
        start2 = time.time()
        print(f"start working on template {tid}")
        conf_dict_list = spark_knobs.df_knob2conf(
            spark_collect.lhs_sampler.get_samples(qpt, random_state=tid)).to_dict("records")
        print(f"configurations generated, cost {time.time() - start2}s")
        arg_list = [(str(tid), str(qid), conf_dict_list[qid-1], f"{out_header}/{tid}", ) for qid in range(1, qpt + 1)]
        with Pool(processes=n_processes) as pool:
            res = pool.starmap_async(spark_collect.save_one_script, arg_list)
            res.get()
        print(f"prepared for template {tid}, cost {time.time() - start2}s")

    print(f"finished prepared {n_templates * qpt} scripts, cost {time.time() - start1}s")


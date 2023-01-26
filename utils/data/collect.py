# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 09/01/2023

import os
import time
import numpy as np

from trace.collect.framework import SparkCollect


def flush_all(workers):
    os.system("sync")
    for worker in workers:
        os.system(f"ssh {worker} sync")


def sql_exec(spark_collect, conf_dict, n_trials, workers, out_header, debug, q_sign, if_aqe):
    """
    return a list of dts in `n_trials`, only used in the profiling purpose
    when we run sqls in the concurrent-running env, we do NOT use `sync`
    """

    # prepare the scripts for running.
    a, b = q_sign.split("-")
    tid, qid = a[1:], b
    file_name = spark_collect.save_one_script(tid, qid, conf_dict, out_header=out_header, if_aqe=if_aqe)

    # check if the results has been run already
    res_file = f"{out_header}/{file_name}.dts"
    if os.path.exists(res_file):
        try:
            with open(res_file) as f:
                dts = [float(dt_str) for dt_str in f.readlines()[0].split(",")]
            assert len(dts) == n_trials
            print(f"{res_file} has been found!")
            return dts
        except:
            print(f"{res_file} is not properly generated")
    print(f"not found {res_file}")
    dts = []
    for i in range(n_trials):
        if debug:
            dts.append(np.random.rand() * 100)
        else:
            flush_all(workers)
            time.sleep(1)
            start = time.time()
            os.system(f"bash {out_header}/{file_name} > {out_header}/{file_name}_trial_{i + 1}.log 2>&1")
            dts.append(time.time() - start)
        print(f"{file_name}, trial {i + 1}, {dts[i]:.3f}s")
    with open(f"{out_header}/{file_name}.dts", "w") as f:
        f.write(",".join([f"{dt:.3f}" for dt in dts]))
    return dts


def run_q_confs(bm, sf, spark_knobs, query_header, out_header, seed, workers, n_trials, debug, q_sign, conf_df,
                if_aqe):
    spark_collect = SparkCollect(
        benchmark=bm,
        scale_factor=sf,
        spark_knobs=spark_knobs,
        query_header=query_header,
        seed=seed
    )

    objs = []
    for conf_dict in conf_df.to_dict("records"):
        dts = sql_exec(spark_collect, conf_dict, n_trials, workers, out_header, debug, q_sign, if_aqe)
        objs.append(sum(dts) / n_trials)
    objs = np.array(objs)

    return objs

# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 09/01/2023

import os
import time
import numpy as np


def flush_all(workers):
    os.system("sync")
    for worker in workers:
        os.system(f"ssh {worker} sync")


def sql_exec(spark_collect, conf_dict, n_trials, workers, out_header, debug, tid, qid=1):
    """return a list of dts in `n_trials`"""

    # prepare the scripts for running.
    out = f"{out_header}/{tid}-{qid}"
    file_name = spark_collect.save_one_script(tid, str(qid), conf_dict, out_header=out, if_aqe=True)

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
        if debug:
            dts.append(np.random.rand() * 100)
        else:
            flush_all(workers)
            time.sleep(1)
            start = time.time()
            os.system(f"bash {out}/{file_name} > {out}/{file_name}_trial_{i + 1}.log 2>&1")
            dts.append(time.time() - start)
        print(f"{file_name}, trial {i + 1}, {dts[i]:.3f}s")
    with open(f"{out}/{file_name}.dts", "w") as f:
        f.write(",".join([f"{dt:.3f}" for dt in dts]))
    return dts
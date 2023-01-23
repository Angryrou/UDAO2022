# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: Test the system states when the resources are used as much as the Spark cluster can provide.
#
# Created at 9/23/22
import argparse, os, time
import random

import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager

from trace.collect.framework import error_handler
from utils.common import PickleUtils, BenchmarkUtils
from utils.data.feature import NmonUtils


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCxBB")
        self.parser.add_argument("-k", "--knob-meta-file", type=str, default="resources/knob-meta/spark.json")
        self.parser.add_argument("-s", "--seed", type=int, default=42)
        self.parser.add_argument("--script-header", type=str, default="resources/scripts/tpcxbb-lhs")
        self.parser.add_argument("--out-header", type=str, default="examples/trace/spark/6.run_all_pressure_test_tpcxbb")
        self.parser.add_argument("--cache-header", type=str, default="examples/trace/spark/cache")
        self.parser.add_argument("--remote-header", type=str, default="~/chenghao")
        self.parser.add_argument("--num-templates", type=int, default=30)
        self.parser.add_argument("--id-template-start", type=int, default=0)
        self.parser.add_argument("--id-template-end", type=int, default=1428)
        self.parser.add_argument("--num-processes", type=int, default=6)
        self.parser.add_argument("--cluster-cores", type=int, default=150)
        self.parser.add_argument("--counts", type=int, default=864000)
        self.parser.add_argument("--freq", type=int, default=5)
        self.parser.add_argument("--debug", type=int, default=0)
        self.parser.add_argument("--worker", type=str, default=None)

    def parse(self):
        return self.parser.parse_args()


def extract(submit_index, templates, n_templates, conf_df_dict):
    tid = templates[submit_index % n_templates]
    conf_df_i = conf_df_dict[tid].iloc[submit_index // n_templates]
    qid, knob_sign = conf_df_i.name
    cores = int(conf_df_i["spark.executor.cores"]) * (int(conf_df_i["spark.executor.instances"]) + 1)
    return tid, str(qid), knob_sign, cores


def submit(lock, current_cores, cores: int, tid: str, qid: str, knob_sign: str, debug: bool, script_header: str, log_header: str):
    script_file = f"{script_header}/{tid}/q{tid}-{qid}_{knob_sign}.sh"
    assert os.path.exists(script_file), FileNotFoundError(script_file)
    log_file = f"{log_header}/q{tid}-{qid}_{knob_sign}.log"

    print(f"Thread {tid}-{qid}: start running")
    start = time.time()
    if debug:
        time.sleep(random.randint(1, 5))
    else:
        os.system(f"bash {script_file} > {log_file} 2>&1")
    with lock:
        current_cores.value -= cores
        print(f"Thread {tid}-{qid}: finish running, takes {time.time() - start}s, current_cores={current_cores.value}")


if __name__ == '__main__':

    args = Args().parse()

    benchmark = args.benchmark
    seed = args.seed
    script_header = args.script_header
    out_header = args.out_header
    cache_header = os.path.join(args.cache_header, benchmark.lower())
    remote_header = args.remote_header
    n_templates = args.num_templates
    t_st = args.id_template_start
    t_en = args.id_template_end
    n_processes = args.num_processes
    cluster_cores = args.cluster_cores
    counts = args.counts
    freq = args.freq
    if args.worker is None:
        workers = BenchmarkUtils.get_workers(benchmark)
    else:
        workers = BenchmarkUtils.get_workers(args.worker)
    debug = False if args.debug == 0 else True
    templates = BenchmarkUtils.get(benchmark)
    log_header = f"{out_header}/log"
    nmon_header = f"{out_header}/nmon"
    os.makedirs(log_header, exist_ok=True)
    os.makedirs(nmon_header, exist_ok=True)

    qpt = 100
    assert benchmark == "TPCxBB" and n_templates == 30

    # prepare nmon commands
    nmon_reset = NmonUtils.nmon_remote_reset(workers, remote_header=remote_header)
    nmon_start = NmonUtils.nmon_remote_start(workers, remote_header=remote_header, name_suffix="",
                                             counts=counts, freq=freq)
    nmon_stop = NmonUtils.nmon_remote_stop(workers)
    nmon_agg = NmonUtils.nmon_remote_agg(workers, remote_header=remote_header, local_header=nmon_header, name_suffix="")

    # get conf_df_dict
    conf_df = pd.concat(PickleUtils.load(cache_header, f"lhs_{n_templates}x{qpt}.pkl"))
    conf_df_dict = {
        tid: conf_df.loc[tid].sample(frac=1, random_state=i)[t_st: t_en]
        for i, tid in enumerate(templates)
    }

    total_queries = sum(v.shape[0] for k, v in conf_df_dict.items())
    m = Manager()
    current_cores = m.Value("i", 0)
    lock = m.RLock()

    submit_index = 0
    pool = Pool(processes=n_processes)

    np.random.seed(seed)
    random.seed(seed)
    random.shuffle(templates)

    if not debug:
        os.system(nmon_reset)
        os.system(nmon_start)

    while submit_index < total_queries:
        tid, qid, knob_sign, cores = extract(submit_index, templates, n_templates, conf_df_dict)
        with lock:
            if cores + current_cores.value < cluster_cores:
                current_cores.value += cores
                if_submit = True
                print(f"Main Process: submit {tid}-{qid}, current_cores = {current_cores.value}")
            else:
                if_submit = False
        if if_submit:
            pool.apply_async(func=submit,
                             args=(lock, current_cores, cores, tid, qid, knob_sign, debug, script_header, log_header),
                             error_callback=error_handler)
            submit_index += 1
            if submit_index % n_templates == 0:
                random.shuffle(templates)

        time.sleep(1)

    pool.close()
    pool.join()

    if not debug:
        os.system(nmon_stop)
        os.system(nmon_agg)

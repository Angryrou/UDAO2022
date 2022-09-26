# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: Test the system states when the resources are used as much as the Spark cluster can provide.
#
# Created at 9/23/22
import argparse, os, time
import random

from multiprocessing import Pool, Manager
from trace.collect.framework import QueryQueue
from utils.common import PickleUtils
from utils.data.feature import NmonUtils


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("-k", "--knob-meta-file", type=str, default="resources/knob-meta/spark.json")
        self.parser.add_argument("-s", "--seed", type=int, default=42)
        self.parser.add_argument("--script-header", type=str, default="resources/scripts/tpch-lhs")
        self.parser.add_argument("--out-header", type=str, default="examples/trace/spark/6.run_all_pressure_test")
        self.parser.add_argument("--cache-header", type=str, default="examples/trace/spark/cache")
        self.parser.add_argument("--num-templates", type=int, default=22)
        self.parser.add_argument("--num-queries-per-template-to-run", type=int, default=400)
        self.parser.add_argument("--num-processes", type=int, default=6)
        self.parser.add_argument("--cluster-cores", type=int, default=150)
        self.parser.add_argument("--debug", type=int, default=0)

    def parse(self):
        return self.parser.parse_args()


def extract(qq, conf_df_dict, i):
    tid, qid = qq.index_to_tid_and_qid(i)
    conf_df = conf_df_dict[tid].iloc[qid - 1]
    knob_sign = conf_df.name
    cores = int(conf_df["spark.executor.cores"]) * (int(conf_df["spark.executor.instances"]) + 1)
    return tid, qid, knob_sign, cores


def submit(lock, current_cores, cores, tid, qid, knob_sign, debug, script_header, log_header):
    script_file = f"{script_header}/{tid}/q{tid}-{qid}_{knob_sign}.sh"
    assert os.path.exists(script_file), FileNotFoundError(script_file)
    log_file = f"{log_header}/q{tid}-{qid}.log"

    print(f"Thread {tid}-{qid}: start running")
    start = time.time()
    if debug:
        time.sleep(random.randint(1, 5))
    else:
        os.system(f"bash {script_file} > {log_file} 2>&1")
    with lock:
       current_cores.value -= cores
       print(f"Thread {tid}-{qid}: finish running, takes {time.time() - start}s, current_cores={current_cores.value}")


def error_handler(e):
    print('error')
    print(dir(e), "\n")
    print("-->{}<--".format(e.__cause__))

if __name__ == '__main__':

    REMOTE_HEADER = "~/chenghao"

    args = Args().parse()

    benchmark = args.benchmark
    seed = args.seed
    script_header = args.script_header
    out_header = args.out_header
    cache_header = args.cache_header
    n_templates = args.num_templates
    n_processes = args.num_processes
    qpt = args.num_queries_per_template_to_run
    cluster_cores = args.cluster_cores
    workers = ["node2", "node3", "node4", "node5", "node6"]
    debug = False if args.debug == 0 else True

    log_header = f"{out_header}/log"
    nmon_header = f"{out_header}/nmon"
    os.makedirs(log_header, exist_ok=True)
    os.makedirs(nmon_header, exist_ok=True)

    if benchmark == "TPCH":
        assert n_templates == 22
        qpt_total = 4545
    elif benchmark == "TPCDS":
        assert n_templates == 105
        qpt_total = 952
    else:
        raise ValueError(benchmark)
    if debug:
        qpt_total = qpt

    # prepare nmon commands
    nmon_reset = NmonUtils.nmon_remote_reset(workers, remote_header=REMOTE_HEADER)
    nmon_start = NmonUtils.nmon_remote_start(workers, remote_header=REMOTE_HEADER, name_suffix="",
                                             counts=86400, freq=5)
    nmon_stop = NmonUtils.nmon_remote_stop(workers)
    nmon_agg = NmonUtils.nmon_remote_agg(workers, remote_header=REMOTE_HEADER, local_header=nmon_header, name_suffix="")

    # prepare the query list
    qq = QueryQueue(n_templates=n_templates, qpt=qpt_total, seed=seed)
    # get conf_df_dict
    conf_df_dict = PickleUtils.load(cache_header, f"lhs_{n_templates}x{qpt_total}.pkl")

    if not debug:
        os.system(nmon_reset)
        os.system(nmon_start)

    total_queries = n_templates * qpt
    m = Manager()
    current_cores = m.Value("i", 0)
    lock = m.RLock()

    submit_index = 0
    pool = Pool(processes=n_processes)
    while submit_index < total_queries:
        tid, qid, knob_sign, cores = extract(qq, conf_df_dict, submit_index)
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

        time.sleep(1)

    pool.close()
    pool.join()

    if not debug:
        os.system(nmon_stop)
        os.system(nmon_agg)


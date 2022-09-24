# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: Test the system states when the resources are used as much as the Spark cluster can provide.
#
# Created at 9/23/22
import argparse, os, time

from multiprocessing import Pool
from glob import glob

from trace.collect.framework import QueryQueue
from utils.data.feature import NmonUtils


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("-k", "--knob-meta-file", type=str, default="resources/knob-meta/spark.json")
        self.parser.add_argument("-s", "--seed", type=int, default=42)
        self.parser.add_argument("--script-header", type=str, default="resources/scripts/tpch-lhs")
        self.parser.add_argument("--out-header", type=str, default="examples/trace/spark/6.run_all_pressure_test")
        self.parser.add_argument("--num-templates", type=int, default=22)
        self.parser.add_argument("--num-queries-per-template-to-run", type=int, default=400)
        self.parser.add_argument("--num-processes", type=int, default=6)

    def parse(self):
        return self.parser.parse_args()


def run_one_batch(tid, qid):
    files = glob(f"{script_header}/{tid}/q{tid}-{qid}_*")
    assert len(files) == 1
    file = files[0]
    file_name = file.split("/")[-1].split(".")[0]
    print(f"start running {file_name}")
    start = time.time()
    os.system(f"bash {file} >> {log_header}/q{tid}-{qid}.log")
    print(f"finished running for {file_name}, takes {time.time() - start}s")


if __name__ == '__main__':
    REMOTE_HEADER = "~/chenghao"

    args = Args().parse()

    benchmark = args.benchmark
    seed = args.seed
    script_header = args.script_header
    out_header = args.out_header
    n_templates = args.num_templates
    n_processes = args.num_processes
    qpt = args.num_queries_per_template_to_run
    total_cores = args.cluster_cores
    workers = ["node2", "node3", "node4", "node5", "node6"]

    log_header = f"{out_header}/log"
    nmon_header = f"{out_header}/nmon"

    if benchmark == "tpch":
        assert n_templates == 22
        qpt_total = 4545
    elif benchmark == "tpcds":
        assert n_templates == 105
        qpt_total = 952
    else:
        raise ValueError(benchmark)

    # prepare nmon commands
    nmon_reset = NmonUtils.nmon_remote_reset(workers, remote_header=REMOTE_HEADER)
    nmon_start = NmonUtils.nmon_remote_start(workers, remote_header=REMOTE_HEADER, name_suffix="",
                                             counts=86400, freq=5)
    nmon_stop = NmonUtils.nmon_remote_stop(workers)
    nmon_agg = NmonUtils.nmon_remote_agg(workers, remote_header=REMOTE_HEADER, local_header=nmon_header, name_suffix="")

    # prepare the query list
    qq = QueryQueue(n_templates=n_templates, qpt=qpt_total, seed=seed)
    total_queries = n_templates * qpt_total
    arg_list = [qq.index_to_tid_and_qid(i) for i in range(total_queries)]

    try:
        os.system(nmon_reset)
        os.system(nmon_start)

        # submit all at one time
        # (1) total cores: 150
        # (2) cores for a spark sql: 8-50
        # (3) at most spark sqls running: 150 / 8 = 18, n_processes >= 18
        with Pool(processes=n_processes) as pool:
            res = pool.starmap(run_one_batch, arg_list)

        os.system(nmon_stop)
        os.system(nmon_agg)
    except Exception as e:
        print(f"failed to run due to {e}")
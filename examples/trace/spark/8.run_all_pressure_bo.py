# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 10/30/22

import argparse, os, time
import random
import threading

from multiprocessing import Pool, Manager
from multiprocessing.managers import ValueProxy

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from trace.collect.framework import QueryQueue, error_handler, SparkCollect
from trace.collect.sampler import BOSampler
from trace.parser.spark import get_cloud_cost
from utils.common import BenchmarkUtils, PickleUtils
from utils.data.configurations import SparkKnobs, KnobUtils
from utils.data.feature import NmonUtils

N_CANDS_PER_TRIAL = 3

class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("-k", "--knob-meta-file", type=str, default="resources/knob-meta/spark.json")
        self.parser.add_argument("-s", "--seed", type=int, default=42)
        self.parser.add_argument("-q", "--query-header", type=str, default="resources/tpch-kit/spark-sqls")
        self.parser.add_argument("--out-header", type=str, default="examples/trace/spark/8.run_all_pressure_bo")
        self.parser.add_argument("--cache-header", type=str, default="examples/trace/spark/cache")
        self.parser.add_argument("--remote-header", type=str, default="~/chenghao")
        self.parser.add_argument("--num-templates", type=int, default=22)
        self.parser.add_argument("--num-queries-per-template-to-run-lhs", type=int, default=3637)
        self.parser.add_argument("--num-queries-per-template-to-run-bo", type=int, default=454)
        self.parser.add_argument("--num-processes", type=int, default=6)
        self.parser.add_argument("--cluster-cores", type=int, default=150)
        self.parser.add_argument("--counts", type=int, default=864000)
        self.parser.add_argument("--freq", type=int, default=5)
        self.parser.add_argument("--debug", type=int, default=0)
        self.parser.add_argument("--sgd-lr", type=float, default=1)
        self.parser.add_argument("--sgd-epochs", type=int, default=80)
        self.parser.add_argument("--if-aqe", type=int, default=0)

    def parse(self):
        return self.parser.parse_args()


def find_next_sample(tid, observed, knob_signs, bo_trials, target_obj_id, lr, epochs):
    X, Y = observed["X"], observed["Y"]
    # add the recommended knob values

    while True:
        bo_trials += 1
        reco_samples = bo_sampler.get_samples(N_CANDS_PER_TRIAL, X, Y[:, target_obj_id], "SGD", lr, epochs)
        # get the recommended knob values based on reco_samples by denormalizing and rounding
        reco_knob_df = KnobUtils.knob_denormalize(reco_samples, knobs)
        assert reco_knob_df.shape == (N_CANDS_PER_TRIAL, len(knobs))
        # get the signature of the recommended knob values
        reco_knob_signs = reco_knob_df.index.values

        for i in range(N_CANDS_PER_TRIAL):
            reco_knob_sign, reco_sample = reco_knob_signs[i], reco_samples[i]
            # to check whether the recommended knob values have already appeared in previous observation due to rounding.
            if reco_knob_sign in knob_signs:
                # map the new_ojb to the existed one if the reco have already been observed
                reco_i = knob_signs.index(reco_knob_sign)
                new_Y = Y[reco_i: reco_i + 1]
                # update observed results
                X = np.vstack([X, reco_sample])
                Y = np.vstack([Y, new_Y])
                print(f"{tid}: trial {bo_trials}-cand{i} recommended an observed configuration, skip running.")
            else:
                print(f"{tid}: trial {bo_trials}-cand{i} recommended a new configuration!")
                return reco_sample, {"X": X, "Y": Y}, bo_trials
        print(f"{tid}: trial {bo_trials} recommended {N_CANDS_PER_TRIAL} observed configuration, skip.")


def extract(qq, templates, i, next_sample_dict):
    tiid, qid = qq.index_to_tid_and_qid(i)
    tid = templates[tiid]
    next_sample = next_sample_dict[tid]
    return tid, qid, next_sample


def submit(
        lock: threading.RLock,
        current_cores: ValueProxy,
        cores: int,
        tid: str,
        qid: str,
        conf_dict: dict,
        debug: bool,
        out_header: str,
        new_sample: np.ndarray,
        target_obj_id: int,
        if_aqe: bool
):
    header = f"{out_header}/{tid}"
    os.makedirs(header, exist_ok=True)
    os.makedirs(f"{header}/log", exist_ok=True)

    file_name = spark_collect.save_one_script(tid, qid, conf_dict, header, if_aqe)
    log_file = f"{header}/log/q{tid}-{qid}.log"

    print(f"Thread {tid}-{qid}: start running")
    start = time.time()
    if debug:
        time.sleep(random.randint(1, 5))
    else:
        os.system(f"bash {header}/{file_name} > {log_file} 2>&1")
    lat = time.time() - start
    with lock:
        current_cores.value -= cores
        print(f"Thread {tid}-{qid}: finish running, takes {lat}s, current_cores={current_cores.value}")

    cost = get_cloud_cost(
        lat=lat,
        mem=int(conf_dict["spark.executor.memory"][:-1]),
        cores=int(conf_dict["spark.executor.cores"]),
        nexec=int(conf_dict["spark.executor.instances"])
    )
    X, Y = observed_dict[tid]["X"], observed_dict[tid]["Y"]
    new_objs = np.array([lat, cost])
    new_Y = objs_scaler_dict[tid].transform(new_objs)

    observed_dict[tid] = {
        "X": np.vstack([X, new_sample]),
        "Y": np.vstack([Y, new_Y])
    }

    next_conf, new_observed, bo_trials = find_next_sample(
        tid, observed, knob_signs, bo_trials=0, target_obj_id=target_obj_id,
        lr=sgd_lr, epochs=sgd_epochs
    )
    observed_dict[tid] = new_observed
    bo_trials_dict[tid] = bo_trials
    next_sample_dict[tid] = next_conf


if __name__ == '__main__':

    args = Args().parse()

    benchmark = args.benchmark
    seed = args.seed
    out_header = args.out_header
    cache_header = args.cache_header
    remote_header = args.remote_header
    n_templates = args.num_templates
    n_processes = args.num_processes
    qpt_lhs = args.num_queries_per_template_to_run_lhs
    qpt_bo = args.num_queries_per_template_to_run_bo
    cluster_cores = args.cluster_cores
    counts = args.counts
    freq = args.freq
    query_header = args.query_header
    workers = BenchmarkUtils.get_workers(benchmark)
    debug = False if args.debug == 0 else True
    sgd_lr, sgd_epochs = args.sgd_lr, args.sgd_epochs
    if_aqe = False if args.if_aqe == 0 else True

    templates = BenchmarkUtils.get(benchmark)

    log_header = f"{out_header}/log"
    nmon_header = f"{out_header}/nmon"
    os.makedirs(log_header, exist_ok=True)
    os.makedirs(nmon_header, exist_ok=True)

    if benchmark == "TPCH":
        assert n_templates == 22
        if not debug:
            assert qpt_lhs == 3637
    elif benchmark == "TPCDS":
        assert n_templates == 103
        if not debug:
            assert qpt_lhs == 777
    else:
        raise ValueError(benchmark)

    qpt_total = qpt_lhs + qpt_bo
    # prepare nmon commands
    nmon_reset = NmonUtils.nmon_remote_reset(workers, remote_header=remote_header)
    nmon_start = NmonUtils.nmon_remote_start(workers, remote_header=remote_header, name_suffix="",
                                             counts=counts, freq=freq)
    nmon_stop = NmonUtils.nmon_remote_stop(workers)
    nmon_agg = NmonUtils.nmon_remote_agg(workers, remote_header=remote_header, local_header=nmon_header, name_suffix="")

    # prepare the query list
    qq = QueryQueue(n_templates=n_templates, qpt=qpt_total, seed=seed)
    # get lhs statistics
    spark_knobs = SparkKnobs(meta_file="resources/knob-meta/spark.json")
    knobs = spark_knobs.knobs
    conf_df_dict = PickleUtils.load(cache_header, f"lhs_{n_templates}x{qpt_lhs}.pkl")
    objs_df_dict = PickleUtils.load(cache_header, f"lhs_{n_templates}x{qpt_lhs}_objs.pkl")
    observed_dict = {}
    objs_scaler_dict = {}
    next_sample_dict = {}
    knob_signs_dict = {}
    bo_trials_dict = {}

    bo_sampler = BOSampler(knobs, seed=seed, debug=debug)
    for tid in templates:
        conf_df = conf_df_dict[tid]
        objs_df = objs_df_dict[tid]
        if not (conf_df.index == objs_df.index).all():
            print(f"index not match b/w conf and obj for {tid}")
            objs_df_dict[tid] = objs_df.loc[conf_df.index]
        knob_df = spark_knobs.df_conf2knob(conf_df)
        samples = KnobUtils.knob_normalize(knob_df, knobs)
        objs = objs_df.values
        scaler = MinMaxScaler()
        scaler.fit(objs)
        objs_normalized = scaler.transform(objs)
        observed = {
            "X": samples, # normalized
            "Y": objs_normalized
        }
        knob_signs = knob_df.index.to_list()
        knob_signs_dict[tid] = knob_signs
        objs_scaler_dict[tid] = scaler
        next_conf, new_observed, bo_trials = find_next_sample(
            tid, observed, knob_signs, bo_trials=0, target_obj_id=0,
            lr=sgd_lr, epochs=sgd_epochs
        )
        observed_dict[tid] = new_observed
        bo_trials_dict[tid] = bo_trials
        next_sample_dict[tid] = next_conf
        print(f"{tid}, next_conf: {next_conf} at bo_trial {bo_trials}")

    if debug:
        print(f"tid, next_conf, bo_trials")
        for tid in templates:
            print(f"{tid}, {next_sample_dict[tid]}, {bo_trials_dict[tid]}")

    if not debug:
        os.system(nmon_reset)
        os.system(nmon_start)

    total_queries = n_templates * qpt_total
    m = Manager()
    current_cores = m.Value("i", 0)
    lock = m.RLock()

    spark_collect = SparkCollect(
        benchmark=benchmark,
        scale_factor=100,
        spark_knobs=spark_knobs,
        query_header=query_header,
        seed=seed
    )

    submit_index = n_templates * qpt_lhs
    pool = Pool(processes=n_processes)
    submitted = 0
    target_obj_id = 0
    while submit_index < total_queries:
        tid, qid, next_sample = extract(qq, templates, submit_index, next_sample_dict)
        knob_sign = KnobUtils.knob_denormalize(next_sample.reshape(1, -1), knobs).index.values[0]
        knob_dict = {k.id: v for k, v in zip(knobs, KnobUtils.sign2knobs(knob_sign, knobs))}
        conf_dict = spark_knobs.knobs2conf(knob_dict)
        cores = int(conf_dict["spark.executor.cores"]) * (int(conf_dict["spark.executor.instances"]) + 1)
        mem = int(conf_dict["spark.executor.memory"][:-1]) * (int(conf_dict["spark.executor.instances"]) + 1)

        with lock:
            if cores + current_cores.value < cluster_cores:
                current_cores.value += cores
                if_submit = True
                print(f"Main Process: submit {tid}-{qid}, current_cores = {current_cores.value}")
            else:
                if_submit = False
        if if_submit:
            pool.apply_async(func=submit,
                             args=(lock, current_cores, cores, tid, qid, conf_dict, debug,
                                   out_header, next_sample, target_obj_id, if_aqe),
                             error_callback=error_handler)
            submit_index += 1

        time.sleep(1)
        submitted += 1
        if submitted >= qpt_bo * n_templates:
            target_obj_id = 1

    pool.close()
    pool.join()

    if not debug:
        os.system(nmon_stop)
        os.system(nmon_agg)
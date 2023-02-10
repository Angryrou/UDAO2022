# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 10/30/22

import argparse, os, time
import random
import subprocess
from multiprocessing import Pool, Manager

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from trace.collect.framework import error_handler, SparkCollect
from trace.collect.sampler import BOSampler
from trace.parser.spark import get_cloud_cost
from utils.common import BenchmarkUtils, PickleUtils, FileUtils, JsonUtils
from utils.data.configurations import SparkKnobs, KnobUtils
from utils.data.feature import NmonUtils

N_CANDS_PER_TRIAL = 3


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCxBB")
        self.parser.add_argument("-k", "--knob-meta-file", type=str, default="resources/knob-meta/spark.json")
        self.parser.add_argument("-s", "--seed", type=int, default=42)
        self.parser.add_argument("-q", "--query-header", type=str, default="resources/tpch-kit/spark-sqls")
        self.parser.add_argument("--out-header", type=str, default="examples/trace/spark/8.run_all_pressure_bo_tpcxbb")
        self.parser.add_argument("--cache-header", type=str, default="examples/trace/spark/cache")
        self.parser.add_argument("--remote-header", type=str, default="~/chenghao")
        self.parser.add_argument("--num-templates", type=int, default=30)
        self.parser.add_argument("--num-queries-per-query-to-run-bo", type=int, default=48)
        self.parser.add_argument("--num-processes", type=int, default=6)
        self.parser.add_argument("--cluster-cores", type=int, default=150)
        self.parser.add_argument("--counts", type=int, default=864000)
        self.parser.add_argument("--freq", type=int, default=5)
        self.parser.add_argument("--debug", type=int, default=0)
        self.parser.add_argument("--sgd-lr", type=float, default=1)
        self.parser.add_argument("--sgd-epochs", type=int, default=80)
        self.parser.add_argument("--if-aqe", type=int, default=0)
        self.parser.add_argument("--worker", type=str, default=None)

    def parse(self):
        return self.parser.parse_args()


def find_next_sample(bo_sampler, knobs, q_sign, observed, knob_signs, bo_trials, target_obj_id, lr, epochs):
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
                print(f"{q_sign}: trial {bo_trials}-cand{i} recommended an observed configuration, skip running.")
            else:
                print(f"{q_sign}: trial {bo_trials}-cand{i} recommended a new configuration!")
                return reco_sample, {"X": X, "Y": Y}, bo_trials
        print(f"{q_sign}: trial {bo_trials} recommended {N_CANDS_PER_TRIAL} observed configuration, skip.")


def submit(
        lock_misc,
        bo_misc,
        cores: int,
        q_sign: str,
        knob_sign: str,
        conf_dict: dict,
        debug: bool,
        out_header: str,
        new_sample: np.ndarray,
        target_obj_id: int,
        if_aqe: bool
):
    s1, s2 = q_sign.split("-")
    tid, qid = s1[1:], s2

    lock, current_cores, spark_collect, observed_dict, next_sample_dict, bo_trials_dict, \
    objs_scaler_dict, knob_signs_dict = lock_misc
    bo_sampler, knobs, sgd_lr, sgd_epochs = bo_misc

    header = f"{out_header}/{tid}"
    os.makedirs(header, exist_ok=True)
    os.makedirs(f"{header}/log", exist_ok=True)

    file_name = spark_collect.save_one_script(tid, qid, conf_dict, header, if_aqe)
    log_file = f"{header}/log/{q_sign}.log"

    print(f"Thread {tid}-{qid} [1]: start running")
    if debug:
        lat = random.randint(1, 5)
    else:
        start = time.time()
        cmd = f"bash {header}/{file_name} > {log_file} 2>&1"
        subprocess.check_output(cmd, shell=True, timeout=3600 * 2)
        assert os.path.exists(log_file)
        log = FileUtils.read_file(log_file).lower()
        if "error" in log:
            lat = np.inf
        else:
            try:
                rows = log.split("\n")
                appid, url_head = rows[0], rows[1]
                url_str = f"{url_head}/api/v1/applications/{appid}"
                data = JsonUtils.load_json_from_url(url_str)
                lat = data["attempts"][0]["duration"] / 1000  # seconds
            except:
                lat = time.time() - start
    cost = get_cloud_cost(
        lat=lat,
        mem=int(conf_dict["spark.executor.memory"][:-1]),
        cores=int(conf_dict["spark.executor.cores"]),
        nexec=int(conf_dict["spark.executor.instances"])
    )
    knob_signs_dict[q_sign].append(knob_sign)

    with lock:
        current_cores.value -= cores
        print(f"Thread {tid}-{qid} [2]: finish running SparkSQL, takes {lat}s, current_cores={current_cores.value}")

    if debug:
        print(f"Thread {tid}-{qid} [3]: lat={lat}, cost={cost}")

    if np.isinf(lat):
        new_objs = np.array([[lat, cost]])
        new_Y = np.array([[100., 100.]])
    else:
        new_objs = np.array([[lat, cost]])
        new_Y = objs_scaler_dict[q_sign].transform(new_objs)
    if debug:
        print(f"Thread {tid}-{qid} [4]: new_objs={new_objs}, new_Y={new_Y}")

    X, Y = observed_dict[q_sign]["X"], observed_dict[q_sign]["Y"]
    observed = {
        "X": np.vstack([X, new_sample]),
        "Y": np.vstack([Y, new_Y])
    }
    next_sample, new_observed, bo_trials = find_next_sample(
        bo_sampler, knobs, q_sign, observed, knob_signs_dict[q_sign],
        bo_trials=bo_trials_dict[q_sign], target_obj_id=target_obj_id, lr=sgd_lr, epochs=sgd_epochs
    )
    if debug:
        print(f"Thread {tid}-{qid} [5]: get next sample at bo_trials {bo_trials}")

    observed_dict[q_sign] = new_observed
    bo_trials_dict[q_sign] = bo_trials
    next_sample_dict[q_sign] = next_sample
    print(f"Thread {tid}-{qid} [6]: finish updating all, observed.X: "
          f"{X.shape} -> {observed['X'].shape} -> {observed_dict[q_sign]['X'].shape}")


if __name__ == '__main__':
    args = Args().parse()

    benchmark = args.benchmark
    seed = args.seed
    out_header = f"{args.out_header}/{benchmark}"
    cache_header = os.path.join(args.cache_header, benchmark.lower())
    remote_header = args.remote_header
    n_templates = args.num_templates
    n_processes = args.num_processes
    n_bo = args.num_queries_per_query_to_run_bo
    cluster_cores = args.cluster_cores
    counts = args.counts
    freq = args.freq
    query_header = args.query_header
    if args.worker is None:
        workers = BenchmarkUtils.get_workers(benchmark)
    else:
        workers = BenchmarkUtils.get_workers(args.worker)
    debug = False if args.debug == 0 else True
    sgd_lr, sgd_epochs = args.sgd_lr, args.sgd_epochs
    if_aqe = False if args.if_aqe == 0 else True
    templates = BenchmarkUtils.get(benchmark)
    qpt = 100
    assert benchmark.lower() == "tpcxbb" and len(templates) == 30 and n_templates == 30

    log_header = f"{out_header}/log"
    nmon_header = f"{out_header}/nmon"
    os.makedirs(log_header, exist_ok=True)
    os.makedirs(nmon_header, exist_ok=True)

    # prepare nmon commands
    nmon_reset = NmonUtils.nmon_remote_reset(workers, remote_header=remote_header)
    nmon_start = NmonUtils.nmon_remote_start(workers, remote_header=remote_header, name_suffix="",
                                             counts=counts, freq=freq)
    nmon_stop = NmonUtils.nmon_remote_stop(workers)
    nmon_agg = NmonUtils.nmon_remote_agg(workers, remote_header=remote_header, local_header=nmon_header, name_suffix="")

    spark_knobs = SparkKnobs(meta_file="resources/knob-meta/spark.json")
    knobs = spark_knobs.knobs
    obj_df_dict = PickleUtils.load(cache_header, f"lhs_{n_templates}x{qpt}_objs.pkl")
    tq_dict = {tid: sorted(list(qdict.keys())) for tid, qdict in obj_df_dict.items()}

    # we have 30 x 5 queries
    np.random.seed(seed)
    tid_tmp = np.tile(np.arange(1, n_templates + 1), [5 * n_bo * 2, 1])
    tids_random_ordered = np.apply_along_axis(np.random.permutation, axis=1, arr=tid_tmp).reshape(1, -1).squeeze().astype(str)
    qid_tmp = np.tile(np.arange(5), [n_templates * n_bo * 2, 1])
    qids_random_ordered = np.apply_along_axis(np.random.permutation, axis=1, arr=qid_tmp)\
                            .reshape(-1, n_templates, 5).transpose(0, 2, 1)\
                            .reshape(-1, n_templates).reshape(1, -1).squeeze()
    q_sign_injection = [f"q{tid}-{tq_dict[tid][qid]}" for tid, qid in zip(tids_random_ordered, qids_random_ordered)]
    q_signs = np.unique(q_sign_injection)

    if debug:
        q_signs = q_signs[:3]

    # get lhs statistics

    m = Manager()
    observed_dict = m.dict()
    next_sample_dict = m.dict()
    bo_trials_dict = m.dict()
    objs_scaler_dict = {}
    knob_signs_dict = {}
    bo_sampler = BOSampler(knobs, seed=seed, debug=False)

    for tid, qdict in obj_df_dict.items():
        for qid, q_df in qdict.items():
            q_sign = f"q{tid}-{qid}"
            if debug and (q_sign not in q_signs):
                continue
            q_df = q_df.set_index("knob_sign")
            knob_df = spark_knobs.df_conf2knob(q_df)[spark_knobs.knob_names]
            samples = KnobUtils.knob_normalize(knob_df, knobs)
            objs_df = q_df[["lat", "cost"]]
            inf_mask = np.isinf(objs_df).all(1)
            objs = objs_df[~inf_mask].values
            scaler = MinMaxScaler()
            scaler.fit(objs)
            objs_normalized_df = objs_df.copy()
            objs_normalized_df[~inf_mask] = scaler.transform(objs)
            objs_normalized_df[inf_mask] = 100
            objs_normalized = objs_normalized_df.values
            observed = {
                "X": samples,  # normalized
                "Y": objs_normalized
            }

            knob_signs = knob_df.index.to_list()
            next_sample, new_observed, bo_trials = find_next_sample(
                bo_sampler, spark_knobs.knobs, q_sign, observed, knob_signs, bo_trials=0, target_obj_id=0,
                lr=sgd_lr, epochs=sgd_epochs
            )

            observed_dict[q_sign] = new_observed
            bo_trials_dict[q_sign] = bo_trials
            next_sample_dict[q_sign] = next_sample
            objs_scaler_dict[q_sign] = scaler
            knob_signs_dict[q_sign] = knob_signs

            print(f"{q_sign}, next_sample: {next_sample} at bo_trial {bo_trials}")

    if debug:
        print(f"tid, next_sample, bo_trials")
        for q_sign in q_signs:
            print(f"{q_sign}, {next_sample_dict[q_sign]}, {bo_trials_dict[q_sign]}")

    if not debug:
        os.system(nmon_reset)
        os.system(nmon_start)

    current_cores = m.Value("i", 0)
    lock = m.RLock()

    spark_collect = SparkCollect(
        benchmark=benchmark,
        scale_factor=100,
        spark_knobs=spark_knobs,
        query_header=query_header,
        seed=seed
    )

    pool = Pool(processes=n_processes)

    submit_index = 0
    total_queries = len(q_sign_injection)
    bo_conf_dict = {q_sign: {0: [], 1: []} for q_sign in q_signs}

    while submit_index < total_queries:
        target_obj_id = 0 if submit_index < (total_queries // 2) else 1
        q_sign = q_sign_injection[submit_index]
        if debug and q_sign not in q_signs:
            submit_index += 1
            continue
        next_sample = next_sample_dict[q_sign]
        knob_sign = KnobUtils.knob_denormalize(next_sample.reshape(1, -1), knobs).index.values[0]
        knob_dict = {k.id: v for k, v in zip(knobs, KnobUtils.sign2knobs(knob_sign, knobs))}
        conf_dict = spark_knobs.knobs2conf(knob_dict)
        cores = int(conf_dict["spark.executor.cores"]) * (int(conf_dict["spark.executor.instances"]) + 1)

        with lock:
            if cores + current_cores.value < cluster_cores:
                current_cores.value += cores
                if_submit = True
                print(f"Main Process: submit {q_sign}, current_cores = {current_cores.value}")
            else:
                if_submit = False
        if if_submit:
            bo_conf_dict[q_sign][target_obj_id].append(conf_dict)
            lock_misc = lock, current_cores, spark_collect, observed_dict, next_sample_dict, bo_trials_dict, \
                        objs_scaler_dict, knob_signs_dict
            bo_misc = bo_sampler, knobs, sgd_lr, sgd_epochs

            pool.apply_async(func=submit,
                             args=(lock_misc, bo_misc, cores, q_sign, knob_sign,
                                   conf_dict, debug, out_header, next_sample, target_obj_id, if_aqe),
                             error_callback=error_handler)
            submit_index += 1

        time.sleep(1)

    pool.close()
    pool.join()

    if not debug:
        os.system(nmon_stop)
        os.system(nmon_agg)

    bo_conf_df_dict = {q_sign: {0: pd.DataFrame(v[0]), 1: pd.DataFrame(v[1])} for q_sign, v in bo_conf_dict.items()}
    PickleUtils.save(bo_conf_df_dict, header=cache_header, file_name=f"bo_{len(q_signs)}x{n_bo}x2.pkl")
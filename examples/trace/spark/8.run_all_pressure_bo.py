# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 10/30/22

import pandas as pd
import numpy as np
import argparse, os

from multiprocessing import Pool, Manager
import threading
from multiprocessing.managers import ValueProxy

from trace.collect.framework import QueryQueue
from trace.collect.sampler import BOSampler
from utils.common import BenchmarkUtils, PickleUtils
from utils.data.configurations import SparkKnobs, KnobUtils
from utils.data.feature import NmonUtils
from sklearn.preprocessing import MinMaxScaler

N_CANDS_PER_TRIAL = 5

class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("-k", "--knob-meta-file", type=str, default="resources/knob-meta/spark.json")
        self.parser.add_argument("-s", "--seed", type=int, default=42)
        self.parser.add_argument("--script-header", type=str, default="resources/scripts/tpch-lhs")
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

    def parse(self):
        return self.parser.parse_args()


def find_next_conf(tid, bo_sampler, observed, knob_signs, bo_trials, target_obj_id):
    X, Y = observed["X"], observed["Y"]
    # add the recommended knob values

    while True:
        bo_trials += 1
        reco_samples = bo_sampler.get_samples(N_CANDS_PER_TRIAL, X, Y[:, target_obj_id], "SGD", lr=1, epochs=80)
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


if __name__ == '__main__':

    args = Args().parse()

    benchmark = args.benchmark
    seed = args.seed
    script_header = args.script_header
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
    workers = BenchmarkUtils.get_workers(benchmark)
    debug = False if args.debug == 0 else True
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
    next_conf_dict = {}
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
        next_conf, new_observed, bo_trials = find_next_conf(tid, bo_sampler, observed, knob_signs,
                                                            bo_trials=0, target_obj_id=0)
        observed_dict[tid] = new_observed
        bo_trials_dict[tid] = bo_trials
        next_conf_dict[tid] = next_conf

    if not debug:
        os.system(nmon_reset)
        os.system(nmon_start)

    print("")

    # total_queries = n_templates * qpt_total
    # m = Manager()
    # current_cores = m.Value("i", 0)
    # lock = m.RLock()


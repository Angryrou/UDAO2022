# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: moo uncertain exploration
#
# Created at 08/02/2023
from trace.parser.spark import get_cloud_cost
from utils.common import BenchmarkUtils, PickleUtils
from utils.model.proxy import ModelProxy
from utils.model.utils import add_pe, prepare_data_for_opt, get_sample_spark_knobs
from utils.data.configurations import SparkKnobs, KnobUtils

import numpy as np
import pandas as pd
import torch as th

import utils.optimization.moo_utils as moo_ut
import time

def data_preparation():
    # do not change the function
    q_signs = BenchmarkUtils.get_sampled_q_signs("tpch")
    ckp_sign = "b7698e80492e5d72"
    model_name = "GTN"
    data_header = "biao/cache/tpch_100"
    tabular_file = "query_level_cache_data.pkl"
    struct_file = "struct_cache.pkl"

    tabular_data, struct_data = PickleUtils.load(data_header, tabular_file), PickleUtils.load(data_header, struct_file)
    col_dict, minmax_dict = tabular_data["col_dict"], tabular_data["minmax_dict"]
    dag_dict, n_op_types = struct_data["dgl_dict"], len(struct_data["global_ops"])
    struct2template = {v["sql_struct_id"]: v["template"] for v in struct_data["struct_dict"].values()}
    df = pd.concat(tabular_data["dfs"])
    op_groups = ["ch1_type"]
    op_feats = []
    misc = (df, dag_dict, op_groups, op_feats, struct2template, col_dict, minmax_dict)
    add_pe(model_name, dag_dict)
    mp = ModelProxy(
        model_name="GTN",
        ckp_path=f"biao/ckp/{ckp_sign}",
        obj_minmax=minmax_dict["obj"],
        device=th.device("cpu"),
        op_groups=op_groups,
        n_op_types=n_op_types,
    )
    spark_knobs = SparkKnobs(meta_file="resources/knob-meta/spark.json")
    print("data prepared")
    return q_signs, mp, spark_knobs, misc


def sample_knobs(n_samples, spark_knobs, seed=42):
    """
    get random sampled `n_samples` configurations as a DataFrame.
    :param n_samples: int, number of configuration to sample.
    :param spark_knobs: SparkKnob
    :param seed: int
    :return:
    """
    knobs = spark_knobs.knobs
    np.random.seed(seed)
    samples = np.random.rand(n_samples, len(knobs))
    # add our heuristic constraints for the knob s3 (index: -2)
    if q_sign.split("-")[0] == "q14":
        samples[:, -2] = 0
    else:
        samples[:, -2] = 1
    knob_df = KnobUtils.knob_denormalize(samples, knobs)
    knob_df = knob_df.drop_duplicates()
    knob_df.index = knob_df.apply(lambda x: KnobUtils.knobs2sign(x, knobs), axis=1)
    print(f"generated {knob_df.shape[0]}/{n_samples} unique configurations")
    return knob_df


def pred_latency_and_cost(mp, q_sign, spark_knobs, misc, knob_df):
    """
    get the mean and std of a predicted latency
    :param mp: model proxy
    :param q_sign: query signature
    :param spark_knobs: spark knobs, to validate the conf_norm is valid.
    :param misc: other necessary variables for modeling
    :param knob_df: a pd.DataFrame of knobs
    :param conf_norm: a 0-1 normalized configuration 2D array (Nx12)
    :return:
    """
    assert knob_df.ndim == 2 and knob_df.shape[1] == 12, "knob_df should has 12 columns"
    knobs = spark_knobs.knobs
    conf_norm = KnobUtils.knob_normalize(knob_df, knobs)
    df, dag_dict, op_groups, op_feats, struct2template, col_dict, minmax_dict = misc
    stage_emb, ch2_norm, ch3_norm = prepare_data_for_opt(
        df, q_sign, dag_dict, mp.hp_params["ped"], op_groups, op_feats, struct2template, mp, col_dict, minmax_dict)
    lats_mu, lats_std = mp.get_lat(ch1=stage_emb, ch2=ch2_norm, ch3=ch3_norm, ch4=conf_norm,
                                   out_fmt="numpy", dropout=True)
    conf_df = spark_knobs.df_knob2conf(knob_df)
    costs_mu = get_cloud_cost(
        lat=lats_mu,
        mem=conf_df["spark.executor.memory"].str[:-1].astype(int).values.reshape(-1, 1),
        cores=conf_df["spark.executor.cores"].astype(int).values.reshape(-1, 1),
        nexec=conf_df["spark.executor.instances"].astype(int).values.reshape(-1, 1)
    )
    costs_std = get_cloud_cost(
        lat=lats_std,
        mem=conf_df["spark.executor.memory"].str[:-1].astype(int).values.reshape(-1, 1),
        cores=conf_df["spark.executor.cores"].astype(int).values.reshape(-1, 1),
        nexec=conf_df["spark.executor.instances"].astype(int).values.reshape(-1, 1)
    )
    return lats_mu, lats_std, costs_mu, costs_std

def get_soo_index(objs, ws_pairs):
    '''
    reuse code in VLDB2022
    :param objs: ndarray(n_feasible_samples/grids, 2)
    :param ws_pairs: list, one weight setting for all objectives, e.g. [0, 1]
    :return: int, index of the minimum weighted sum
    '''
    obj = np.sum(objs * ws_pairs, axis=1)
    return np.argmin(obj)

def reco_configurations_adding_objective(mp: ModelProxy, q_sign: str, spark_knobs: SparkKnobs, misc: tuple, n_samples: int, n_probes: int, weight_std: float) -> list:
    """
    Recommend Pareto-optimal configurations
    :param mp: model proxy
    :param q_sign: query signature
    :param spark_knobs: spark knobs
    :param misc: other necessary variables for modeling
    :return:
    """
    # TODO: reco knob_df.
    n_samples = n_samples
    n_probes = n_probes
    weight_std = weight_std
    steps = n_probes - 2

    # ws_pairs = moo_ut.even_weights(1/steps, 3)

    w1 = np.linspace(0, 1-weight_std, steps, endpoint=True) # (steps,)
    w2 = 1 - weight_std - w1
    w3 = np.array([weight_std]*steps)
    ws_pairs = np.concatenate((w1.reshape(-1,1), w2.reshape(-1,1), w3.reshape(-1,1)), axis=1) # weight pairs between [0,1] (steps,3)

    knob_df = sample_knobs(n_samples=n_samples, spark_knobs=spark_knobs)
    lats_mu, lats_std, costs_mu, costs_std = pred_latency_and_cost(mp, q_sign, spark_knobs, misc, knob_df)
    # knob_df: df (n_samples,12) 7,5,8,2,1,False,False,72,2,3,6,True
    # mu,std: nparray (n_samples,1)
    
    weighted_sums_std = lats_std / lats_mu + costs_std / costs_mu # weighted sum of the standard derivation

    objs = np.concatenate((lats_mu,costs_mu,weighted_sums_std), axis = 1) # (n_samples,3)
    
    po_ind_list = []

    # normalization
    objs_min, objs_max = objs.min(0), objs.max(0)
    
    if all((objs_min - objs_max) < 0):
        objs_norm = (objs - objs_min) / (objs_max - objs_min)
        for ws in ws_pairs:
            po_ind = get_soo_index(objs_norm, ws)
            po_ind_list.append(po_ind)

        # only keep non-dominated solutions
        po_objs_cand = objs[po_ind_list]
        reco_confs_cand = knob_df.iloc[po_ind_list]
        po_inds = moo_ut.is_pareto_efficient(po_objs_cand, return_mask=False)
        reco_confs = reco_confs_cand.iloc[po_inds]
        return reco_confs
    else:
        raise Exception(f"Cannot do normalization! Lower bounds of objective values are higher than their upper bounds.")

def reco_configurations_approximated_robust(mp: ModelProxy, q_sign: str, spark_knobs: SparkKnobs, misc: tuple, n_samples: int, n_probes: int, alpha: float) -> list:
    """
    Recommend Pareto-optimal configurations
    :param mp: model proxy
    :param q_sign: query signature
    :param spark_knobs: spark knobs
    :param misc: other necessary variables for modeling
    :return:
    """
    # TODO: reco knob_df.
    n_samples = n_samples
    n_probes = n_probes
    alpha = alpha
    steps = n_probes - 2

    w1 = np.linspace(0, 1, steps, endpoint=True) # (steps,)
    w2 = 1 - w1
    ws_pairs = np.concatenate((w1.reshape(-1,1), w2.reshape(-1,1)), axis=1) # weight pairs between [0,1] (steps,2)

    knob_df = sample_knobs(n_samples=n_samples, spark_knobs=spark_knobs)
    lats_mu, lats_std, costs_mu, costs_std = pred_latency_and_cost(mp, q_sign, spark_knobs, misc, knob_df)
    # knob_df: df (n_samples,12) 7,5,8,2,1,False,False,72,2,3,6,True
    # mu,std: nparray (n_samples,1)

    obj_1s = lats_mu + alpha * lats_std # only for minimization (mu + a*std) (n_samples,1)
    obj_2s = costs_mu + alpha * costs_std

    objs = np.concatenate((obj_1s,obj_2s), axis = 1) # (n_samples,2)
    
    po_ind_list = []

    # normalization
    objs_min, objs_max = objs.min(0), objs.max(0)
    
    if all((objs_min - objs_max) < 0):
        objs_norm = (objs - objs_min) / (objs_max - objs_min)
        for ws in ws_pairs:
            po_ind = get_soo_index(objs_norm, ws)
            po_ind_list.append(po_ind)

        # only keep non-dominated solutions
        po_objs_cand = objs[po_ind_list]
        reco_confs_cand = knob_df.iloc[po_ind_list]
        po_inds = moo_ut.is_pareto_efficient(po_objs_cand, return_mask=False)
        reco_confs = reco_confs_cand.iloc[po_inds]
        return reco_confs
    else:
        raise Exception(f"Cannot do normalization! Lower bounds of objective values are higher than their upper bounds.")

def reco_configurations(mp: ModelProxy, q_sign: str, spark_knobs: SparkKnobs, misc: tuple, n_samples: int, n_probes: int) -> list:
    """
    Recommend Pareto-optimal configurations
    :param mp: model proxy
    :param q_sign: query signature
    :param spark_knobs: spark knobs
    :param misc: other necessary variables for modeling
    :return:
    """
    # TODO: reco knob_df.
    n_samples = n_samples
    n_probes = n_probes
    steps = n_probes - 2

    w1 = np.linspace(0, 1, steps, endpoint=True) # (steps,)
    w2 = 1 - w1
    ws_pairs = np.concatenate((w1.reshape(-1,1), w2.reshape(-1,1)), axis=1) # weight pairs between [0,1] (steps,2)

    knob_df = sample_knobs(n_samples=n_samples, spark_knobs=spark_knobs)
    lats_mu, lats_std, costs_mu, costs_std = pred_latency_and_cost(mp, q_sign, spark_knobs, misc, knob_df)
    # knob_df: df (n_samples,12) 7,5,8,2,1,False,False,72,2,3,6,True
    # mu,std: nparray (n_samples,1)

    objs = np.concatenate((lats_mu, costs_mu), axis = 1) # (n_samples,2)
    
    po_ind_list = []

    # normalization
    objs_min, objs_max = objs.min(0), objs.max(0)
    
    if all((objs_min - objs_max) < 0):
        objs_norm = (objs - objs_min) / (objs_max - objs_min)
        for ws in ws_pairs:
            po_ind = get_soo_index(objs_norm, ws)
            po_ind_list.append(po_ind)

        # only keep non-dominated solutions
        po_objs_cand = objs[po_ind_list]
        reco_confs_cand = knob_df.iloc[po_ind_list]
        po_inds = moo_ut.is_pareto_efficient(po_objs_cand, return_mask=False)
        reco_confs = reco_confs_cand.iloc[po_inds]
        return reco_confs
    else:
        raise Exception(f"Cannot do normalization! Lower bounds of objective values are higher than their upper bounds.")

if __name__ == "__main__":
    q_signs, mp, spark_knobs, misc = data_preparation()

#=====================Original========================================================================
    n_samples = 10000 # sample number for weighted sum
    n_probes = 22 # probe number for weighted sum (find au maximum n_probes - 2 Pareto points)

    start_time = time.time()

    res = {}
    for q_sign in q_signs:
        print(f"start working on {q_sign}")
        reco_confs = reco_configurations(mp, q_sign, spark_knobs, misc, n_samples, n_probes)
        assert isinstance(reco_confs, pd.DataFrame) and (reco_confs.columns == spark_knobs.knob_names).all(), \
            "invalid reco_confs format"
        res[q_sign] = reco_confs

    out_header = "biao/outs"
    out_file_name = f"reco_samples{n_samples}_probes{n_probes}_original.pkl"  # todo: please keep track of the version of your MOO methods
    PickleUtils.save(res, out_header, out_file_name, overwrite=False)

    print(f"runtime: {time.time() - start_time}")

#=================Adding objective==============================================================
    n_samples = 10000 # sample number for weighted sum
    n_probes = 22 # probe number for weighted sum (find au maximum n_probes - 2 Pareto points)
    # weight_std = 0.3 # weight for the third objective of variance information
    weight_std_list = [0.7]

    for weight_std in weight_std_list:
        start_time = time.time()

        res = {}
        for q_sign in q_signs:
            print(f"start working on {q_sign}")
            reco_confs = reco_configurations_adding_objective(mp, q_sign, spark_knobs, misc, n_samples, n_probes, weight_std)
            assert isinstance(reco_confs, pd.DataFrame) and (reco_confs.columns == spark_knobs.knob_names).all(), \
                "invalid reco_confs format"
            res[q_sign] = reco_confs

        out_header = "biao/outs"
        out_file_name = f"reco_samples{n_samples}_probes{n_probes}_w{weight_std}.pkl"  # todo: please keep track of the version of your MOO methods
        PickleUtils.save(res, out_header, out_file_name, overwrite=False)

        print(f"runtime: {time.time() - start_time}")

#==================Approximated robust=============================================================
    n_samples = 10000 # sample number for weighted sum
    n_probes = 22 # probe number for weighted sum (find au maximum n_probes - 2 Pareto points)
    # alpha = 3 # alpha for the coefficient of the variance information
    alpha_list = [3,5,20]

    for alpha in alpha_list:
        start_time = time.time()

        res = {}
        for q_sign in q_signs:
            print(f"start working on {q_sign}")
            reco_confs = reco_configurations_approximated_robust(mp, q_sign, spark_knobs, misc, n_samples, n_probes, alpha)
            assert isinstance(reco_confs, pd.DataFrame) and (reco_confs.columns == spark_knobs.knob_names).all(), \
                "invalid reco_confs format"
            res[q_sign] = reco_confs

        out_header = "biao/outs"
        out_file_name = f"reco_samples{n_samples}_probes{n_probes}_alpha{alpha}.pkl"  # todo: please keep track of the version of your MOO methods
        PickleUtils.save(res, out_header, out_file_name, overwrite=False)

        print(f"runtime: {time.time() - start_time}")
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
    print(f"generated {knob_df.shape[1]}/{n_samples} unique configurations")
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


def reco_configurations(mp: ModelProxy, q_sign: str, spark_knobs: SparkKnobs, misc: tuple) -> list:
    """
    Recommend Pareto-optimal configurations
    :param mp: model proxy
    :param q_sign: query signature
    :param spark_knobs: spark knobs
    :param misc: other necessary variables for modeling
    :return:
    """
    # TODO: reco knob_df.
    # an example to get mu and std of the latency and cost for 100 random sampled configurations.
    knob_df = sample_knobs(n_samples=100, spark_knobs=spark_knobs)
    lats_mu, lats_std, cost_mu, cost_std = pred_latency_and_cost(mp, q_sign, spark_knobs, misc, knob_df)

    reco_confs = knob_df
    return reco_confs


if __name__ == "__main__":
    q_signs, mp, spark_knobs, misc = data_preparation()
    res = {}
    for q_sign in q_signs:
        reco_confs = reco_configurations(mp, q_sign, spark_knobs, misc)
        assert isinstance(reco_confs, pd.DataFrame) and (reco_confs.columns == spark_knobs.knob_names).all(), \
            "invalid reco_confs format"
        res[q_sign] = reco_confs

    out_header = "biao/outs"
    out_file_name = "reco.pkl"  # todo: please keep track of the version of your MOO methods
    PickleUtils.save(res, out_header, out_file_name, overwrite=False)

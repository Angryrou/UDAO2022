# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: sample code to use the glb model for optimization
#
# Created at 16/06/2023
import itertools
import os
import random

import dgl
import numpy as np

from model.architecture.avg_mlp_glb import AVGMLP_GLB
from utils.common import BenchmarkUtils, PickleUtils
import pandas as pd
import torch as th

from utils.data.configurations import SparkKnobs
from utils.model.utils import get_tensor, denorm_obj, norm_in_feat_inst


class ModelProxy:

    def __init__(self, model_name, ckp_path, obj_minmax, device, op_groups, n_op_types):
        assert model_name == "AVGMLP_GLB"
        assert os.path.exists(ckp_path)
        results_pth_sign = f"{ckp_path}/results.pth"
        weights_pth_sign = f"{ckp_path}/best_weight.pth"
        results = th.load(results_pth_sign, map_location=device)
        hp_params = results["hp_params"]
        hp_params = {**hp_params, **{
            "op_groups": op_groups,
            "n_op_types": n_op_types
        }}
        model = AVGMLP_GLB(hp_params).to(device=device)
        self.model_states = th.load(weights_pth_sign, map_location=device)["model"]
        model.load_state_dict(self.model_states)
        print(f"model loaded.")

        self.results = results
        self.hp_params = hp_params
        self.model = model
        self.device = device
        self.obj_minmax = {"min": get_tensor(obj_minmax["min"].values, device=device),
                           "max": get_tensor(obj_minmax["max"].values, device=device)}

    def get_lat(self,
                g_stage: dgl.DGLGraph,
                g_op: dgl.DGLGraph,
                normalized_ch2: th.FloatTensor,
                normalized_ch3: th.FloatTensor,
                normalized_theta_q: th.FloatTensor,
                normalized_theta_s: th.FloatTensor):
        normalized_inst_feat = th.concat([normalized_ch2, normalized_ch3, normalized_theta_q], dim=1)
        self.model.eval()
        g_stage = g_stage.to(self.device)
        g_op = g_op.to(self.device)
        with th.no_grad():
            lat_hat = self.model.forward(g_stage, g_op, normalized_theta_s, normalized_inst_feat)
        lat_hat = denorm_obj(lat_hat, self.obj_minmax)
        return lat_hat

def reset_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)


def data_preparation():
    # do not change the function
    q_signs = BenchmarkUtils.get_sampled_q_signs("tpch")
    ckp_header = "examples/optimization/glb_model/ckp/tpch_100/AVGMLP_GLB/latency/on_off_off_on_on_on"
    ckp_sign = "a0614433e796e0c3"
    model_name = "AVGMLP_GLB"
    data_header = "examples/optimization/glb_model/cache/tpch_100"
    tabular_file = "query_level_cache_data.pkl"
    struct_file = "struct_cache.pkl"
    tabular_data, struct_data = PickleUtils.load(data_header, tabular_file), PickleUtils.load(data_header, struct_file)
    n_op_types = len(struct_data["global_ops"])
    cache = PickleUtils.load(data_header, "struct_cache_extended.pkl")
    dag_dict = {"g_stage": cache["query_stages_dgl_dict"], "g_op": cache["stages_op_dgl_dict"]}
    col_dict, minmax_dict = tabular_data["col_dict"], tabular_data["minmax_dict"]
    struct2template = {v["sql_struct_id"]: v["template"] for v in struct_data["struct_dict"].values()}
    df = pd.concat(tabular_data["dfs"])
    op_groups = ["ch1_type"]
    op_feats = []

    misc = (df, dag_dict, op_groups, op_feats, struct2template, col_dict, minmax_dict)
    mp = ModelProxy(
        model_name=model_name,
        ckp_path=f"{ckp_header}/{ckp_sign}",
        obj_minmax=minmax_dict["obj"],
        device=th.device("cpu"),
        op_groups=op_groups,
        n_op_types=n_op_types
    )
    spark_knobs = SparkKnobs(meta_file="resources/knob-meta/spark.json")
    print("data prepared")
    return q_signs, mp, spark_knobs, misc

def predict_latency_and_cost(
        mp: ModelProxy,
        q_sign: str,
        misc: tuple,
        normalized_theta_q: th.FloatTensor,  # (N, 8)
        normalized_theta_s: th.FloatTensor,  # (N, m, 4) -- m is the number of stages
):
    """
    get the mean and std of a predicted latency
    :param mp: model proxy
    :param q_sign: query signature
    :param misc: other necessary variables for modeling
    :param knob_df: a pd.DataFrame of knobs
    :return:
    """
    assert normalized_theta_q.ndim == 2 and normalized_theta_q.shape[1] == 8
    assert normalized_theta_s.ndim == 3 and normalized_theta_s.shape[2] == 4
    assert normalized_theta_q.shape[0] == normalized_theta_s.shape[0]
    N = normalized_theta_q.shape[0]

    df, dag_dict, op_groups, op_feats, struct2template, col_dict, minmax_dict = misc
    # prepare g_stage, g_op, ch2, ch3
    record = df[df["q_sign"] == q_sign]
    sid = record.sql_struct_id.iloc[0]
    g_stage, g_op = dag_dict["g_stage"][sid], dag_dict["g_op"][sid]
    ch2_norm = get_tensor(norm_in_feat_inst(record[col_dict["ch2"]], minmax_dict["ch2"]).values).repeat(N, 1)
    ch3_norm = th.zeros((N, len(col_dict["ch3"])))  # as like in the idle env
    g_stages = [g_stage] * N
    g_ops = [g_op] * N
    batched_g_stage = dgl.batch(g_stages)
    batched_g_op = dgl.batch(list(itertools.chain(*g_ops)))
    normalized_theta_s = normalized_theta_s.reshape(-1, 4)
    lat = mp.get_lat(batched_g_stage, batched_g_op, ch2_norm, ch3_norm, normalized_theta_q, normalized_theta_s)
    return lat


def reco_configurations(mp, q_sign, spark_knobs, misc, n_samples, n_probes):
    # todo
    ...

q_signs, mp, spark_knobs, misc = data_preparation()
df, dag_dict, op_groups, op_feats, struct2template, col_dict, minmax_dict = misc

for i, q_sign in enumerate(q_signs):
    print(f"start solving {q_sign}")
    reset_seed(i)
    N = 1 # number of configurations for prediction
    record = df[df["q_sign"] == q_sign]
    sid = record.sql_struct_id.iloc[0]
    m = dag_dict["g_stage"][sid].num_nodes() # number of stages
    normalized_theta_q = th.rand(N, 8)
    normalized_theta_s = th.rand(N, m, 4)
    lat = predict_latency_and_cost(mp, q_sign, misc, normalized_theta_q, normalized_theta_s)
    print(lat.squeeze().item())
    print()
# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 31/03/2023

import dgl
import torch as th

import numpy as np
import pandas as pd
from utils.common import PickleUtils, ParquetUtils
from utils.model.utils import get_tensor
from utils.data.extractor import plot_dgl_graph, show_q


def message_func(edges):
    return {"sub_lat": edges.src["sub_lat"]}

def reduce_func(nodes):
    x = th.max(nodes.mailbox["sub_lat"], 1)
    return {"merged_lat": x[0]}

def apply_node_func_real(nodes):
    return {"sub_lat": nodes.data["merged_lat"] + nodes.data["latency"]}

def apply_node_func_pred(nodes):
    return {"sub_lat": nodes.data["merged_lat"] + nodes.data["latency_pred"]}

def get_wmape(y, y_hat):
    return np.abs(y-y_hat).sum() / y.sum()

def get_mape(y, y_hat):
    return np.mean(np.abs(y-y_hat)/ y)

def draw_topology():
    for sid in range(42):
        if sid not in qs_dag_dict:
            continue
        q = query_df[query_df["sql_struct_id"] == sid].sort_values("lat_err").iloc[-1]
        q_sign = q["q_sign"]
        appid = q.name
        # slats = stage_df[stage_df["q_sign"] == q_sign].sort_values("qs_id").stage_latency.values
        # labels = {i: f"{qs2stage_dict[sid][i]}\\n{f'{slats[i]:.1f}' if slats[i] < 1 else f'{slats[i]:.0f}'}s"
        #           for i in range(qs_dag_dict[sid].num_nodes())}
        labels = {i: i for i in range(qs_dag_dict[sid].num_nodes())}
        print(f"{sid}({q_sign}), {appid}, stage2query_latency: {q['stage2query_latency']:.1f}s, query_latency: {q['latency']:.1f}s")
        plot_dgl_graph(
            g=qs_dag_dict[sid],
            node_id2name=labels,
            dir_name="stage_topology/tpch_100_without_stage_mappings",
            title=f"{sid}({q_sign})",
            prefix=False,
            color="lightgreen",
            fillcolor="lightgreen",
            jupyter=False
        )

header = "examples/data/spark/cache/tpch_100"
query_data = PickleUtils.load(header, "query_level_cache_data.pkl")
query_df = pd.concat(query_data["dfs"])

stage_data = PickleUtils.load(header, "stage_level_cache_data.pkl")
stage_df_raw = ParquetUtils.parquet_read(header, "csvs_stage.parquet")
stage_df = pd.concat(stage_data["dfs"])
stage_df["sql_struct_svid"] = stage_df.sql_struct_svid.astype(int)
qs_dependencies_dict = stage_data["qs_dependencies_dict"]
qs2stage_dict = stage_data["qs2stage_dict"]
struct2template = query_df[["sql_struct_id", "template"]].drop_duplicates().set_index("sql_struct_id", drop=True).to_dict()["template"]
query_df = query_df.merge(stage_df[["id", "mapping_sign_id"]].drop_duplicates(), on="id")

qs_dag_dict = {}
id_dict = {}
for sid, deps in qs_dependencies_dict.items():
    src_ids = np.unique(deps, axis=0)[:, 0]
    dst_ids = np.unique(deps, axis=0)[:, 1]
    g = dgl.graph((src_ids, dst_ids))

    df_sid = stage_df[stage_df["sql_struct_id"] == sid]
    num_qs = len(df_sid.groupby("qs_id"))
    lat = np.array([df_sid[df_sid["qs_id"] == qs_id]["stage_latency"].values for qs_id in range(num_qs)])
    ids = np.array([df_sid[df_sid["qs_id"] == qs_id]["id"].values for qs_id in range(num_qs)])
    if len(lat) > 0:
        g.ndata["latency"] = get_tensor(lat)
        qs_dag_dict[sid] = g
        id_dict[sid] = ids[0]

appids = np.concatenate(list(id_dict.values())).tolist()
assert len(query_df) == len(appids)
query_df = query_df.set_index("id").loc[appids]

# stage topology to query latency
# 1
for sid, gi in qs_dag_dict.items():
    g = gi.clone()
    g.ndata["sub_lat"] = th.zeros_like(g.ndata["latency"])
    g.ndata["merged_lat"] = th.zeros_like(g.ndata["latency"])
    dgl.prop_nodes_topo(g, message_func=message_func, reduce_func=reduce_func, apply_node_func=apply_node_func_real)
    query_df.loc[query_df[query_df["sql_struct_id"] == sid].index, "stage2query_latency"] = g.ndata["sub_lat"].max(0)[
        0].numpy()

query_df["stage2query_latency"] += query_df["cpl_opt_time"]
query_df["lat_err"] = np.abs(query_df["stage2query_latency"] - query_df["latency"])
print(f"wmape = {get_wmape(query_df.latency, query_df.stage2query_latency)}")
# 2
stage_dt = stage_df[["id", "stage_dt"]].groupby("id").sum()
query_df.loc[stage_dt.index, "stage_dt_sum"] = stage_dt.values
query_df["stage2query_latency2"] = query_df["cpl_opt_time"] + query_df["stage_dt_sum"] / (query_df["k2"] * query_df["k3"]) / 1000
query_df["lat_err2"] = np.abs(query_df["stage2query_latency2"] - query_df["latency"])
print(f"wmape = {get_wmape(query_df.latency, query_df.stage2query_latency2)}")
# 3
stage_df["stage_dt_contribution"] = stage_df["stage_dt"] / np.minimum(stage_df["task_num"], stage_df["k2"] * stage_df["k3"])
stage_dt = stage_df[["id", "stage_dt_contribution"]].groupby("id").sum() / 1000
query_df.loc[stage_dt.index, "stage2query_latency3"] = query_df["cpl_opt_time"].values + stage_dt.values.squeeze()
query_df["lat_err3"] = np.abs(query_df["stage2query_latency3"] - query_df["latency"])
print(f"wmape = {get_wmape(query_df.latency, query_df.stage2query_latency3)}")

# breakdown of each sid
wmape_dict, mape_dict = {}, {}
print("sid\twmape\tmape")
for sid, ids in id_dict.items():
    qlats = query_df.loc[ids].latency.values
    s2qlats = query_df.loc[ids]["stage2query_latency"].values
    wmape = get_wmape(qlats, s2qlats)
    wmape_dict[sid] = wmape
    mape = get_mape(qlats, s2qlats)
    mape_dict[sid] = mape
    print(f"{sid}({struct2template[sid]})\t{wmape:.3f}\t{mape:.3f}")

qs2stage_dict = stage_data["qs2stage_dict"]
stage_df["br_time"] = stage_df["stage_latency"] - stage_df["stage_latency_wo_broadcast"]
stage_br = stage_df[stage_df["br_time"] > 0].copy()
stage_br["stage_id"] = stage_br.apply(lambda x: max(qs2stage_dict[x["mapping_sign_id"]][x["qs_id"]]), axis=1)
stage_df_raw = stage_df_raw.merge(stage_br[["id", "stage_id", "br_time"]], on=["id", "stage_id"], how="left").fillna(0)
stage_df_raw["stage_latency"] = stage_df_raw["stage_latency"] + stage_df_raw["br_time"]

for sid in range(42):
    if sid not in qs_dag_dict:
        continue
    q = query_df[query_df["sql_struct_id"] == sid].sort_values("lat_err").iloc[-1]
    q_sign = q["q_sign"]
    appid = q.name
    slats = stage_df[stage_df["q_sign"] == q_sign].sort_values("qs_id").stage_latency.values
    mapping_id = query_df.loc[appid, "mapping_sign_id"]
    labels = {i: f"{qs2stage_dict[mapping_id][i]}\\n{slats[i]:.1f}s"
              for i in range(qs_dag_dict[sid].num_nodes())}
    print(f"{sid}({q_sign}), {appid}, stage2query_latency: {q['stage2query_latency']:.1f}s, query_latency: {q['latency']:.1f}s")
    plot_dgl_graph(
        g=qs_dag_dict[sid],
        node_id2name=labels,
        dir_name="stage_topology/tpch_100",
        title=f"{sid}({q_sign})",
        prefix=False,
        color="lightgreen",
        fillcolor="lightgreen",
        jupyter=False
    )
    show_q(query_df, stage_df_raw, sid, q_sign, save_to="application_graphs/stage_topology/tpch_100/timelines")
    # show_q(query_df, stage_df_raw, sid, q_sign, save_to=None)


# sid = 4
# for q_sign in ["q3-685", "q3-25", "q3-1", "q3-5", "q3-15"]:
#     q = query_df[query_df["q_sign"] == q_sign].iloc[0]
#     appid = q.name
#     slats = stage_df[stage_df["q_sign"] == q_sign].sort_values("qs_id").stage_latency.values
#     mapping_id = query_df.loc[appid, "mapping_sign_id"]
#     labels = {i: f"{qs2stage_dict[mapping_id][i]}\\n{slats[i]:.1f}s"
#               for i in range(qs_dag_dict[sid].num_nodes())}
#
#     plot_dgl_graph(
#         g=qs_dag_dict[sid],
#         node_id2name=labels,
#         dir_name="stage_topology/tpch_100/debug",
#         title=f"{sid}({q_sign})",
#         prefix=False,
#         color="lightgreen",
#         fillcolor="lightgreen",
#         jupyter=True
#     )
#     show_q(query_df, stage_df_raw, sid, q_sign, save_to="application_graphs/stage_topology/tpch_100/debug/timelines")
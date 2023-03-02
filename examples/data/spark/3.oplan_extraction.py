# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: target output
# (1) cbo feat cache for ALL queries: (tid, qid) -> a lsit of (size, nrow)
# (2) a dict mapping from sql_struct_id to a mapping from physical_nids to logical_nids
#
# Created at 01/02/2023

import argparse, os, time
import re

import numpy as np
import pandas as pd

from utils.common import BenchmarkUtils, FileUtils, ParquetUtils, PickleUtils
from utils.data.extractor import get_csvs_tr_val_te, get_csvs, list_strip, LogicalStruct, SqlStructBefore, \
    extract_ofeats, format_size, plot_nx_graph_augment
from utils.data.feature import L2P_MAP


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--src-path-header", type=str, default="resources/dataset/tpch_100_query_traces")
        self.parser.add_argument("--oplan-path-header", type=str, default="resources/dataset/tpch_100_oplans")
        self.parser.add_argument("--oplan-name", type=str, default="logical_plans.parquet")
        self.parser.add_argument("--cache-header", type=str, default="examples/data/spark/cache")
        self.parser.add_argument("--cbo-cache-name", type=str, default="cbo_cache.pkl")
        self.parser.add_argument("--debug", type=int, default=1)
        self.parser.add_argument("--seed", type=int, default=42)

    def parse(self):
        return self.parser.parse_args()


args = Args().parse()
print(args)
bm = args.benchmark.lower()
sf = args.scale_factor
src_path_header = args.src_path_header
oplan_path_header = args.oplan_path_header
cache_header = f"{args.cache_header}/{bm}_{sf}"
debug = False if args.debug == 0 else True
seed = args.seed

# preprocess the tpch_100_oplans
oplan_name = args.oplan_name
try:
    lp_df = ParquetUtils.parquet_read(cache_header, oplan_name)
    print(f"found {oplan_name}")
except:
    print(f"cannot find {oplan_name}, start generating...")
    start = time.time()
    files = os.listdir(oplan_path_header)
    file_dict = {file.split("_oplan")[0]: FileUtils.read_file(f"{oplan_path_header}/{file}") for file in files}
    lp_df = pd.DataFrame.from_dict(file_dict, orient="index")
    lp_df = lp_df.reset_index()
    lp_df.columns = ["q_sign", "logical_plan"]
    ParquetUtils.parquet_write(lp_df, cache_header, oplan_name)
    print(f"{oplan_name} generated, cost {time.time() - start}s")

# (1) cbo feat cache for ALL queries: (tid, qid) -> a lsit of (size, nrow)
cbo_cache_name = args.cbo_cache_name
try:
    cbo_cache = PickleUtils.load(cache_header, cbo_cache_name)
    print(f"{cbo_cache_name} found")
except:
    print(f"{cbo_cache_name} not found, generating...")
    df = lp_df
    df["size"], df["nrows"] = zip(*df.apply(lambda x: extract_ofeats(x["logical_plan"].splitlines()), axis=1))
    df = df[["q_sign", "size", "nrows"]]
    df["template"] = df.q_sign.apply(lambda x: x.split("-")[0])
    df["qid"] = df.q_sign.apply(lambda x: int(x.split("-")[1]))
    ofeat_dict = {}
    templates = [f"q{i}" for i in BenchmarkUtils.get(bm)]
    for tid in templates:
        xx = df[df["template"] == tid]
        xx = xx.sort_values("qid")
        xx_feat = xx[["qid", "size", "nrows"]].explode(["size", "nrows"])
        xx_feat["size"] = xx_feat["size"].apply(lambda x: format_size(x))
        xx_feat["nrows"] = xx_feat["nrows"].astype(float)
        feat_np = xx_feat[["size", "nrows"]].values.reshape(len(xx), -1, 2)
        ofeat_dict[tid] = np.concatenate([feat_np, np.log(feat_np + 1e-6)], axis=2)

    df_tr, _, _ = get_csvs_tr_val_te(templates, src_path_header, cache_header, seed)
    ofeats = np.concatenate([v[df_tr.loc[f"q{1}"].q_sign.apply(lambda x: int(x.split("-")[1])).values].reshape(-1, 4)
                             for k, v in ofeat_dict.items()], axis=0)
    minmax = {"min": ofeats.min(0), "max": ofeats.max(0)}
    PickleUtils.save({
        "ofeat_dict": ofeat_dict,
        "minmax": minmax
    }, cache_header, cbo_cache_name)
    print(f"{cbo_cache_name} generated.")

# (2) a dict mapping from sql_struct_id to a mapping from physical_nids to logical_nids
struct_cache = PickleUtils.load(cache_header, "struct_cache.pkl")
struct_dict = struct_cache["struct_dict"]
struct_dgl_dict = struct_cache["struct_dgl_dict"]
global_ops = struct_cache["global_ops"]
dgl_dict = struct_cache["dgl_dict"]
q2struct = struct_cache["q2struct"]

l_dict, p_dict = {}, {}
for k, v in struct_dgl_dict.items():
    q_sign = v.d["q_sign"]
    sa, sb = v.struct_after, v.struct_before
    v.p2.g.in_degrees()
    ls = LogicalStruct(lp_df.set_index("q_sign").loc[q_sign].logical_plan.splitlines())
    # ls.struct.plot("tpch_100_from_logical", f"{k}-{q_sign}")
    #
    # rmapping = {vv: kk for kk, vv in v.mapping.items()}
    # plot_nx_graph_augment(
    #     G=v.p1.G,
    #     node_id2name=v.p1.node_id2name,
    #     dir_name="tpch_100_from_metric_augment",
    #     title=f"{k}-{q_sign}",
    #     nodes_desc={rmapping[kk]: vv for kk, vv in sb.nodes_desc.items()}
    # )
    l_dict[q_sign] = ls

# map from the ids in struct_after's dgl (p1) to logical nids
p2l = L2P_MAP[bm]
for k, v in struct_dgl_dict.items():
    assert len(v.get_nnames()) == len(p2l[k])


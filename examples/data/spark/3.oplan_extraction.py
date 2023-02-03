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
        self.parser.add_argument("--oplan-path-header", type=str, default="resources/dataset/tpch_oplans")
        self.parser.add_argument("--cache-header", type=str, default="examples/data/spark/cache")
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

# preprocess the tpch_oplans
oplan_name = "logical_plans.parquet"
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
cbo_cache_name = "cbo_cache.pkl"
try:
    cbo_cache = PickleUtils.load(cache_header, cbo_cache_name)
    print(f"{cbo_cache_name} found")
except:
    print(f"{cbo_cache_name} not found, generating...")
    x = lp_df
    x["size"], x["nrows"] = zip(*x.apply(lambda x: extract_ofeats(x["logical_plan"].splitlines()), axis=1))
    x = x[["q_sign", "size", "nrows"]]
    x["template"] = x.q_sign.apply(lambda x: x.split("-")[0])
    x["qid"] = x.q_sign.apply(lambda x: int(x.split("-")[1]))
    cbo_cache = {}
    for tid in BenchmarkUtils.get(bm):
        xx = x[x["template"] == f"q{tid}"]
        xx = xx.sort_values("qid")
        xx_feat = xx[["qid", "size", "nrows"]].explode(["size", "nrows"])
        xx_feat["size"] = xx_feat["size"].apply(lambda x: format_size(x))
        xx_feat["nrows"] = xx_feat["nrows"].astype(float)
        cbo_cache[int(tid)] = xx_feat[["size", "nrows"]].values.reshape(len(xx), -1, 2)
    PickleUtils.save(cbo_cache, cache_header, cbo_cache_name)

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

# debug
# if debug:
#     q_signs = BenchmarkUtils.get_sampled_q_signs(bm)
#     df1 = lp_df[lp_df.q_sign.isin(q_signs)].set_index("q_sign").loc[q_signs]
# else:
#     df1 = lp_df.set_index("q_sign")
# templates = [f"q{i}" for i in BenchmarkUtils.get(bm)]
# df2 = get_csvs(templates, src_path_header, cache_header, samplings=["lhs", "bo"]).reset_index().set_index("q_sign")
#
# def get_plans(q_sign, df1, df2):
#     lp = df1.loc[q_sign].splitlines()
#     desc = df2.loc[q_sign]["planDescription"]
#     plans = list_strip(re.compile("={2,}").split(desc))
#     tree_str, details_str = list_strip(plans[1].split("\\n\\n\\n"))
#     pp = tree_str.split("\\n")
#     if len(plans) == 4:
#         ...
#     return lp, pp

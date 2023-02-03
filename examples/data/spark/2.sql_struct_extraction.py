# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description:
#
# Created at 12/23/22

import argparse, random

import numpy as np
from utils.common import BenchmarkUtils, PickleUtils
from utils.data.extractor import get_csvs, SqlStruct, get_tr_val_te_masks
from utils.data.feature import CH1_FEATS, CH2_FEATS, CH3_FEATS, CH4_FEATS, OBJS


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--src-path-header", type=str, default="resources/dataset/tpch_100_query_traces")
        self.parser.add_argument("--cache-header", type=str, default="examples/data/spark/cache")
        self.parser.add_argument("--debug", type=int, default=0)
        self.parser.add_argument("--if-plot", type=int, default=1)
        self.parser.add_argument("--seed", type=int, default=42)

    def parse(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = Args().parse()
    bm = args.benchmark.lower()
    sf = args.scale_factor
    src_path_header = args.src_path_header
    cache_header = f"{args.cache_header}/{bm}_{sf}"
    debug = False if args.debug == 0 else True
    if_plot = False if args.if_plot == 0 else True
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)

    templates = [f"q{i}" for i in BenchmarkUtils.get(bm)]
    df = get_csvs(templates, src_path_header, cache_header, samplings=["lhs", "bo"])

    # 1. get unique query structures with operator types.
    struct_cache_name = "struct_cache.pkl"
    try:
        struct_cache = PickleUtils.load(cache_header, struct_cache_name)
        struct_dict = struct_cache["struct_dict"]
        struct_dgl_dict = struct_cache["struct_dgl_dict"]
        global_ops = struct_cache["global_ops"]
        dgl_dict = struct_cache["dgl_dict"]
        q2struct = struct_cache["q2struct"]
        print(f"find cached structures at {cache_header}/{struct_cache_name}")
    except:
        print(f"cannot find cached structure, start generating...")
        struct_dict = df.loc[df.sql_struct_id.drop_duplicates().index].to_dict(orient="index")
        struct_dgl_dict = {d["sql_struct_id"]: SqlStruct(d) for d in struct_dict.values()}
        global_ops = sorted(list(set.union(*[set(v.get_nnames()) for k, v in struct_dgl_dict.items()])))
        dgl_dict = {k: v.get_dgl(global_ops) for k, v in struct_dgl_dict.items()}
        q2struct = {x[0]: x[1] for x in df[["q_sign", "sql_struct_id"]].values}
        PickleUtils.save({
            "struct_dict": struct_dict,
            "struct_dgl_dict": struct_dgl_dict,
            "global_ops": global_ops,
            "dgl_dict": dgl_dict,
            "q2struct": q2struct
        }, header=cache_header, file_name=struct_cache_name)
        print(f"generated cached structure at {cache_header}/{struct_cache_name}")

    # generate data for query-level modeling

    head_cols = ["id", "q_sign", "knob_sign", "template", "sampling", "start_timestamp"]
    ch1_cols, ch2_cols, ch3_cols, ch4_cols = CH1_FEATS, CH2_FEATS, CH3_FEATS, CH4_FEATS
    obj_cols = OBJS

    df[ch4_cols] = df.knob_sign.str.split(",", expand=True)
    df[["k6", "k7", "s4"]] = (df[["k6", "k7", "s4"]] == "True") + 0
    df[ch4_cols] = df[ch4_cols].astype(float)

    selected_cols = head_cols + ch1_cols + ch2_cols + ch3_cols + ch4_cols + obj_cols
    df4model = df[selected_cols]
    tr_mask, val_mask, te_mask = get_tr_val_te_masks(
        df=df4model, groupby_col1="template", groupby_col2="template",
        frac_val_per_group=0.1, frac_te_per_group=0.1, seed=seed)
    df_tr, df_val, df_te = df4model[tr_mask], df4model[val_mask], df4model[te_mask]
    col_dict = {"ch1": ch1_cols, "ch2": ch2_cols, "ch3": ch3_cols, "ch4": ch4_cols, "obj": obj_cols}
    minmax_dict = {}
    for ch in ["ch1", "ch2", "ch3", "ch4", "obj"]:
        df_ = df_tr[col_dict[ch]]
        min_, max_ = df_.min(), df_.max()
        minmax_dict[ch] = {"min": min_, "max": max_}
    cache_data = {
        "full_cols": selected_cols, "col_dict": col_dict, "minmax_dict": minmax_dict,
        "dfs": [df_tr, df_val, df_te]
    }
    PickleUtils.save(cache_data, cache_header, "query_level_cache_data.pkl")


    # 2. get structure mappings
    #   - QueryStage -> [operators]
    #   - SQL -> [QueryStages] + {QueryStageDep - a DGL} / blocked by Exchanges and Subqueries
    #   - QueryStage -> [stages]

    # 4. generate data for queryStage-level modeling

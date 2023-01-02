# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description:
#
# Created at 12/23/22

import argparse, random, time
import json
import os

import numpy as np
import pandas as pd
from utils.common import BenchmarkUtils, PickleUtils
from utils.data.extractor import get_csvs, SqlStruct, SqlStuctBefore, replace_symbols, evals_self, evals

# Word2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import RegexpTokenizer


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
        print(f"find cached structures at {cache_header}/{struct_cache_name}")
    except:
        print(f"cannot find cached structure, start generating...")
        struct_dict = df.loc[df.sql_struct_id.drop_duplicates().index].to_dict(orient="index")
        struct_dgl_dict = {d["sql_struct_id"]: SqlStruct(d) for d in struct_dict.values()}
        global_ops = sorted(list(set.union(*[set(v.get_nnames()) for k, v in struct_dgl_dict.items()])))
        dgl_dict = {k: v.get_dgl(global_ops) for k, v in struct_dgl_dict.items()}
        PickleUtils.save({
            "struct_dict": struct_dict,
            "struct_dgl_dict": struct_dgl_dict,
            "global_ops": global_ops,
            "dgl_dict": dgl_dict
        }, header=cache_header, file_name=struct_cache_name)
        print(f"generated cached structure at {cache_header}/{struct_cache_name}")


    # generate data

    # 2. get structure mappings
    #   - QueryStage -> [operators]
    #   - SQL -> [QueryStages] + {QueryStageDep - a DGL} / blocked by Exchanges and Subqueries
    #   - QueryStage -> [stages]

    # 3. generate data for query-level modeling

    # 4. generate data for queryStage-level modeling

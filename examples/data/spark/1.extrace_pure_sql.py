# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description:
#
# Created at 12/23/22

import argparse
import json
import os

import numpy as np
import pandas as pd

from utils.common import BenchmarkUtils, JsonUtils
from utils.data.extractor import get_csvs, SqlStruct


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--sampling", type=str, default="lhs")
        self.parser.add_argument("--src-path-header", type=str, default="resources/dataset/tpch_100_query_traces")
        self.parser.add_argument("--cache-header", type=str, default="examples/data/spark/cache/tpch_100")

    def parse(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = Args().parse()
    bm = args.benchmark.lower()
    sf = args.scale_factor
    sampling = args.sampling
    src_path_header = args.src_path_header
    cache_header = args.cache_header
    templates = [f"q{i}" for i in BenchmarkUtils.get(bm)]
    df = get_csvs(templates, src_path_header, cache_header, samplings=["lhs", "bo"])

    # 1. generate operator features
    #   - operator type (needs to get ALL types)
    #   - CBO features (get the estimated candidates, average row size and cost)
    #   - doc2vec to summarize the plan description
    struct_dict = df.loc[df.sql_struct_id.drop_duplicates().index].to_dict(orient="index")
    struct_dgl_dict = {d["sql_struct_id"]: SqlStruct(d) for d in struct_dict.values()}

    # 2. get structure mappings
    #   - QueryStage -> [operators]
    #   - SQL -> [QueryStages] + {QueryStageDep - a DGL} / blocked by Exchanges and Subqueries
    #   - QueryStage -> [stages]


    # 3. generate data for query-level modeling


    # 4. generate data for queryStage-level modeling



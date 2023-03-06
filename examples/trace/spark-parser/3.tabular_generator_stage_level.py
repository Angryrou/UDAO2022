# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: generate CSV merged between Query-level traces and Machine Systems states
#
# Created at 12/12/22

import argparse
import os

import numpy as np
import pandas as pd

from utils.common import ParquetUtils, JsonUtils


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--sampling", type=str, default="lhs")
        self.parser.add_argument("--dst-path-header", type=str, default="examples/trace/spark-parser/outs")
        self.parser.add_argument("--dst-path-matches", type=str, default="*query_traces*.parquet")
        self.parser.add_argument("--tabular-tmp-name", type=str, required=False, default=None)

    def parse(self):
        return self.parser.parse_args()


def match_inds(x, y):
    """
    for each x[i], get the index j of y so that y[j] is the maximum among those <x[i], O(logn + m)
    :param x: a sorted timestamp list of queries, size=m
    :param y: a sorted timestamp list of machine traces (every 5s), size=n
    :return: a list of the index of the most recent machine trace to each query
    """
    m, n = len(x), len(y)
    assert x[-1] < y[-1] - 5
    # O(logn + m)
    ret = np.zeros(m).astype(int)
    j = max(np.where(x[0] > y)[0])
    ret[0] = j
    for i, xi in enumerate(x[1:]):
        while xi > y[j + 1]:  # break when xi >= y[j + 1]
            j += 1
        ret[i + 1] = j

    # O(mlogn)
    # ret = np.array([max(np.where(x_i > y)[0]) for x_i in x])
    return ret


if __name__ == "__main__":
    args = Args().parse()
    bm = args.benchmark.lower()
    sf = args.scale_factor
    sampling = args.sampling
    dst_path_header = args.dst_path_header
    matches = args.dst_path_matches
    mach_path = f"{dst_path_header}/{bm}_{sf}_{sampling}/1.mach"
    tabular_path = f"{dst_path_header}/{bm}_{sf}_{sampling}/3.tabular_stages"
    tabular_tmp_name = args.tabular_tmp_name
    dst = f"{tabular_path}/query_traces"
    os.makedirs(dst, exist_ok=True)

    tmp_cols = ["id", "name", "q_sign", "knob_sign", "planDescription", "nodes", "edges",
                "start_timestamp", "latency", "err"]
    if tabular_tmp_name is None:
        df_tabular = ParquetUtils.parquet_read_multiple(tabular_path, matches)
    else:
        df_tabular = ParquetUtils.parquet_read(tabular_path, tabular_tmp_name)
    print(f"originally, get {df_tabular.shape[0]} stages in {df_tabular.id.unique().size} queries to parse")
    df_meta = df_tabular.groupby(["id", "stage_id"]).size()
    failed_ids = df_meta[df_meta>1].reset_index().id.unique().tolist()
    JsonUtils.save_json(failed_ids, f"{dst}/failed_appids.txt")
    df_tabular = df_tabular[~df_tabular.id.isin(failed_ids)]
    df_tabular = df_tabular[df_tabular.err.isna()].sort_values("first_task_launched_time")
    x = df_tabular["first_task_launched_time"].values
    print(f"After filtering failures, get {len(x)} stages in {df_tabular.id.unique().size} queries to parse")

    mach_cols = ["timestamp", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]
    df_mach = ParquetUtils.parquet_read(mach_path, "mach_traces.parquet")
    # original columns: ["timestamp", "cpu_utils", "mem_utils", "disk_busy", "disk_bsize/s",
    #                    "disk_KB/s", "disk_xfers/s", "net_KB/s", "net_xfers/s"]
    df_mach.columns = mach_cols
    y = df_mach["timestamp"].values

    yinds = match_inds(x, y)
    df_tabular[mach_cols] = df_mach.iloc[yinds].values
    df_tabular["template"] = df_tabular.q_sign.str.split("-").str[0]
    df_dict = {k: v for k, v in df_tabular.groupby("template")}
    for k, v in df_tabular.groupby("template"):
        v.to_csv(f"{dst}/{k}_{sampling}.csv", sep="\u0001", index=False)
    print(f"saved for csvs at {dst}/*")

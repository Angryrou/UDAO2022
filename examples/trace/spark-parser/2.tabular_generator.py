import argparse, csv, traceback
import os, json, time
from multiprocessing import Pool

import pandas as pd

from utils.common import JsonUtils, TimeUtils, ParquetUtils


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--sampling", type=str, default="lhs")
        self.parser.add_argument("--dst-path-header", type=str, default="examples/trace/spark-parser/outs")
        self.parser.add_argument("--tabular-tmp-name", type=str, required=True)

    def parse(self):
        return self.parser.parse_args()

if __name__ == '__main__':
    args = Args().parse()
    bm = args.benchmark.lower()
    sf = args.scale_factor
    sampling = args.sampling
    dst_path_header = args.dst_path_header
    mach_path = f"{dst_path_header}/{bm}_{sf}_{sampling}/1.mach"
    tabular_path = f"{dst_path_header}/{bm}_{sf}_{sampling}/2.tabular"
    tabular_tmp_name = args.tabular_tmp_name

    df_mach = ParquetUtils.parquet_read(mach_path, "mach_traces.parquet")
    df_tabular_tmp = ParquetUtils.parquet_read(mach_path, tabular_tmp_name)

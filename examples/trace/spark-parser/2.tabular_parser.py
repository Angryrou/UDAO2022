import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.common import BenchmarkUtils, TimeUtils, ParquetUtils

class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--sampling", type=str, default="lhs")
        self.parser.add_argument("--dst", type=str, default="outs/lhs/2.tabular")



    def parse(self):
        return self.parser.parse_args()

if __name__ == '__main__':
    args = Args().parse()
    workers = BenchmarkUtils.get_workers(args.benchmark)
    dst_path = args.dst_path
    tz_ahead = args.timezone_ahead
    assert tz_ahead in (1, 2)
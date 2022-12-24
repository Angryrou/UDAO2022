# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: convert extracted nmon CSV files to a Table of our interested system metrics in Parquet
#
# Created at 12/08/22

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
        self.parser.add_argument("--src-path", type=str, default="outs/lhs/1.nmon")
        self.parser.add_argument("--dst-path", type=str, default="outs/tpch_100_lhs/1.mach")
        self.parser.add_argument("--timezone-ahead", type=int, default=2)

    def parse(self):
        return self.parser.parse_args()

def get_plot(df_dict, metric, workers, dst_path, one_out_of=1000, metric_prefix=""):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for w in workers:
        df_dict[w][metric][1::one_out_of].plot(label=w)
    ax.legend()
    plt.tight_layout()
    os.makedirs(dst_path, exist_ok=True)
    fig.savefig(f"{dst_path}/{metric_prefix}{metric.replace('/', '_per_')}_1_{one_out_of}.pdf")

FEQ = 5
DISK = "sdh"
NET = "ib0"
_TARGET_FILES = [
    "CPU_ALL.csv",
    # cpu_utils = (100 - idle) / 100
    "MEM.csv",
    # mem utils = current total memory usage / Total Memory
    # current total memory usage = Total Memory - (Free + Buffers + Cached)

    "DISKBUSY.csv",
    "DISKBSIZE.csv",
    # target disk: "sdh" [0, 1]
    # disk_busy: DISKBUSY, Percentage of time during which the disk is active.
    # disk_blocks: DISKBSIZE / 5, Disk Block Size - Total number of disk blocks that are read and written over the interval.
    # disk_bytes/s: DISKWRITE + DISKREAD
    # disk_reqs: DISKXFER, Disk transfers per second - Number of transfers per second.

    "NET.csv",  # ib0-write-KB/s + ib0-read-KB/s
    "NETPACKET.csv"  # ib0-write/s + ib0-read/s
]

def get_df(src_path, w, target):
    file = f"{src_path}/{w}/csv/{target}"
    df = pd.read_csv(file)
    return df

if __name__ == '__main__':
    args = Args().parse()
    workers = BenchmarkUtils.get_workers(args.benchmark)
    src_path = args.src_path
    dst_path = args.dst_path
    tz_ahead = args.timezone_ahead
    assert tz_ahead in (1, 2)

    df_dict = {}
    columns = ["timestamp", "cpu_utils", "mem_utils", "disk_busy", "disk_bsize/s",
               "disk_KB/s", "disk_xfers/s", "net_KB/s", "net_xfers/s"]
    for w in workers:
        df_w = pd.DataFrame(columns=columns)
        # cpu
        # cpu_utils = (100 - idle) / 100
        metric = "cpu_utils"
        df = get_df(src_path, w, target="CPU_ALL.csv")
        df_w[metric] = (100 - df["Idle%"] - df["Steal%"]) / 100
        df_w["timestamp"] = df[df.columns[0]].apply(lambda x: TimeUtils.get_utc_timestamp(x, tz_ahead))

        # mem
        # mem utils = current total memory usage / Total Memory
        # current total memory usage = Total Memory - (Free + Buffers + Cached)
        metric = "mem_utils"
        df = get_df(src_path, w, target="MEM.csv")
        df_w[metric] = (df["memtotal"] - df["memfree"] - df["buffers"] - df["cached"]) / df["memtotal"]

        # disk
        # disk_busy: DISKBUSY, Percentage of time during which the disk is active.
        df = get_df(src_path, w, target="DISKBUSY.csv")
        df_w["disk_busy"] = df[DISK] / 100
        # disk_bsize/s: DISKBSIZE / 5, Disk Block Size
        # - Total number of disk blocks that are read and written over the interval.
        df = get_df(src_path, w, target="DISKBSIZE.csv")
        df_w["disk_bsize/s"] = df[DISK] / FEQ
        # disk_bytes/s
        df_i = get_df(src_path, w, target="DISKREAD.csv")
        df_o = get_df(src_path, w, target="DISKWRITE.csv")
        df_w["disk_KB/s"] = df_i[DISK] + df_o[DISK]
        # disk_xfers/s
        df = get_df(src_path, w, target="DISKXFER.csv")
        df_w["disk_xfers/s"] = df[DISK]

        # network
        # net_bytes/s
        df = get_df(src_path, w, target="NET.csv")
        df_w["net_KB/s"] = df[f"{NET}-read-KB/s"] + df[f"{NET}-write-KB/s"]
        # net_xfers/s = ib0-write/s + ib0-read/s
        df = get_df(src_path, w, target="NETPACKET.csv")
        df_w["net_xfers/s"] = df[f"{NET}-read/s"] + df[f"{NET}-write/s"]

        dts = df_w["timestamp"].values[1:] - df_w["timestamp"].values[:-1]
        df_w.loc[np.where(dts < 0)[0][0] + 1:, "timestamp"] += 3600 # summer to winter
        dts = df_w["timestamp"].values[1:] - df_w["timestamp"].values[:-1]
        print(f"{w}, max_dt={np.max(dts)}, min_dt={np.min(dts)}")
        df_dict[w] = df_w

    n_ts = min([df_w.shape[0] for df_w in df_dict.values()])
    df_mach = pd.concat([df_w[columns[1:]][:n_ts] for df_w in df_dict.values()]).reset_index().groupby("index").mean()
    df_mach["timestamp"] = df_dict[workers[0]]["timestamp"][:n_ts]
    df_mach = df_mach[columns]

    ParquetUtils.parquet_write(df_mach, dst_path, "mach_traces.parquet", True)
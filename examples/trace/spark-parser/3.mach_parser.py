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
        self.parser.add_argument("--src-path", type=str, default="outs/lhs/3.nmon")
        self.parser.add_argument("--dst-path", type=str, default="outs/lhs/3.mach")
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

if __name__ == '__main__':
    args = Args().parse()
    workers = BenchmarkUtils.get_workers(args.benchmark)
    src_path = args.src_path
    dst_path = args.dst_path
    tz_ahead = args.timezone_ahead
    assert tz_ahead in (1, 2)

    # cpu
    # cpu_utils = (100 - idle) / 100
    target = "CPU_ALL.csv"
    metric = "cpu_utils"
    df_cpu = {}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target}"
        df = pd.read_csv(file)
        df[metric] = (100 - df["Idle%"] - df["Steal%"]) / 100
        df_cpu[w] = df

    # align the worker nodes on the beginning timestamp and the number of steps.
    ts_begin = min(TimeUtils.get_utc_timestamp(df.head(1).values[0, 0], tz_ahead) for w, df in df_cpu.items())
    steps = min([df_cpu[w].shape[0] for w in workers])
    ts_list = np.arange(ts_begin, ts_begin + steps * FEQ, FEQ)
    assert len(ts_list) == steps
    for w in workers:
        df_cpu[w] = df.head(steps)
    get_plot(df_cpu, metric, workers, dst_path, 1000)
    cpus = np.array([df_[metric].values for df_ in df_cpu.values()])
    cpu_mu = cpus.mean(0)
    assert len(cpu_mu) == steps
    df_mach = pd.DataFrame({"timestamp": ts_list, metric: cpu_mu})

    # mem
    # mem utils = current total memory usage / Total Memory
    # current total memory usage = Total Memory - (Free + Buffers + Cached)
    target = "MEM.csv"
    metric = "mem_utils"
    df_mem = {}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target}"
        df = pd.read_csv(file, nrows=steps)
        df[metric] = (df["memtotal"] - df["memfree"] - df["buffers"] - df["cached"]) / df["memtotal"]
        df_mem[w] = df
    mems = np.array([df_[metric].values for df_ in df_mem.values()])
    get_plot(df_mem, metric, workers, dst_path, 1000)
    mem_mu = mems.mean(0)
    assert len(mem_mu) == steps
    df_mach[metric] = mem_mu

    # disk
    # disk_busy: DISKBUSY, Percentage of time during which the disk is active.
    target = "DISKBUSY.csv"
    metric = "disk_busy"
    df_diskbusy = {}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target}"
        df = pd.read_csv(file, nrows=steps)
        df[metric] = df[DISK] / 100
        df_diskbusy[w] = df
    diskbusys = np.array([df_[metric].values for df_ in df_diskbusy.values()])
    get_plot(df_diskbusy, metric, workers, dst_path, 1000)
    diskbusy_mu = diskbusys.mean(0)
    assert len(diskbusy_mu) == steps
    df_mach[metric] = diskbusy_mu

    # disk_bsize/s: DISKBSIZE / 5, Disk Block Size - Total number of disk blocks that are read and written over the interval.
    target = "DISKBSIZE.csv"
    metric = "disk_bsize/s"
    df_diskbsize = {}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target}"
        df = pd.read_csv(file, nrows=steps)
        df[metric] = df[DISK] / FEQ
        df_diskbsize[w] = df
    diskbsizes = np.array([df_[metric].values for df_ in df_diskbsize.values()])
    get_plot(df_diskbsize, metric, workers, dst_path, 1000)
    diskbsize_mu = diskbsizes.mean(0)
    assert len(diskbsize_mu) == steps
    df_mach[metric] = diskbsize_mu

    # disk_bytes/s
    target_dict = {"r": "DISKREAD.csv", "w": "DISKWRITE.csv"}
    df_diskbyte = {"r": {}, "w": {}}
    metric = "disk_KB/s"
    for w in workers:
        file = f"{src_path}/{w}/csv/{target_dict['r']}"
        df = pd.read_csv(file, nrows=steps)
        df[metric] = df[DISK]
        df_diskbyte["r"][w] = df

        file = f"{src_path}/{w}/csv/{target_dict['w']}"
        df = pd.read_csv(file, nrows=steps)
        df[metric] = df[DISK]
        df_diskbyte["w"][w] = df
    diskbytes = np.array([df_[metric].values for df_ in df_diskbyte["r"].values()]) + \
                np.array([df_[metric].values for df_ in df_diskbyte["w"].values()])
    get_plot(df_diskbyte["r"], metric, workers, dst_path, 1000, metric_prefix="r_")
    get_plot(df_diskbyte["w"], metric, workers, dst_path, 1000, metric_prefix="w_")
    diskbyte_mu = diskbytes.mean(0)
    assert len(diskbyte_mu) == steps
    df_mach[metric] = diskbyte_mu

    # disk_xfers/s
    target = "DISKXFER.csv"
    metric = "disk_xfers/s"
    df_diskxfer = {}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target}"
        df = pd.read_csv(file, nrows=steps)
        df[metric] = df[DISK]
        df_diskxfer[w] = df
    diskxfers = np.array([df_[metric].values for df_ in df_diskxfer.values()])
    get_plot(df_diskxfer, metric, workers, dst_path, 1000)
    diskxfer_mu = diskxfers.mean(0)
    assert len(diskxfer_mu) == steps
    df_mach[metric] = diskxfer_mu

    # Network
    # net_bytes/s
    target = "NET.csv"
    metric = "net_KB/s"
    df_netbyte = {}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target}"
        df = pd.read_csv(file, nrows=steps)
        df[metric] = df[f"{NET}-read-KB/s"] + df[f"{NET}-write-KB/s"]
        df_netbyte[w] = df
    netbytes = np.array([df_[metric].values for df_ in df_netbyte.values()])
    get_plot(df_netbyte, metric, workers, dst_path, 1000)
    netbyte_mu = netbytes.mean(0)
    assert len(netbyte_mu) == steps
    df_mach[metric] = netbyte_mu

    # net_xfers/s = ib0-write/s + ib0-read/s
    target = "NETPACKET.csv"
    metric = "net_xfers/s"
    df_netxfer = {}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target}"
        df = pd.read_csv(file, nrows=steps)
        df[metric] = df[f"{NET}-read/s"] + df[f"{NET}-write/s"]
        df_netxfer[w] = df
    netxfers = np.array([df_[metric].values for df_ in df_netxfer.values()])
    get_plot(df_netxfer, metric, workers, dst_path, 1000)
    netxfer_mu = netxfers.mean(0)
    assert len(netxfer_mu) == steps
    df_mach[metric] = netxfer_mu

    ParquetUtils.parquet_write(df_mach, dst_path, "mach_traces.parquet", True)

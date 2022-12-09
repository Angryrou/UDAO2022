# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: convert extracted nmon CSV files to a Table of our interested system metrics in Parquet
#
# Created at 12/08/22

import numpy as np
import pandas as pd
import argparse
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
    df_cpu = {}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target}"
        df = pd.read_csv(file)
        df["cpu_utils"] = (100 - df["Idle%"] - df["Steal%"]) / 100
        df_cpu[w] = df

    # align the worker nodes on the beginning timestamp and the number of steps.
    ts_begin = min(TimeUtils.get_utc_timestamp(df.head(1).values[0, 0], tz_ahead) for w, df in df_cpu.items())
    steps = min([df_cpu[w].shape[0] for w in workers])
    ts_list = np.arange(ts_begin, ts_begin + steps * FEQ, FEQ)
    assert len(ts_list) == steps
    for w in workers:
        df_cpu[w] = df.head(steps)
    # for w in workers:
    #     df_cpu[w]["cpu_utils"][1:1000].plot(label=w)
    # plt.legend()
    cpus = np.array([df_["cpu_utils"].values for df_ in df_cpu.values()])
    cpu_mu = cpus.mean(0)
    assert len(cpu_mu) == steps
    df_mach = pd.DataFrame({"timestamp": ts_list, "cpu_utils": cpu_mu})

    # mem
    # mem utils = current total memory usage / Total Memory
    # current total memory usage = Total Memory - (Free + Buffers + Cached)
    target = "MEM.csv"
    df_mem = {}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target}"
        df = pd.read_csv(file, nrows=steps)
        df["mem_utils"] = (df["memtotal"] - df["memfree"] - df["buffers"] - df["cached"]) / df["memtotal"]
        df_mem[w] = df
    mems = np.array([df_["mem_utils"].values for df_ in df_mem.values()])
    # for w in workers:
    #     df_mem[w]["mem_utils"][1:1000].plot(label=w)
    # plt.legend()
    mem_mu = mems.mean(0)
    assert len(mem_mu) == steps
    df_mach["mem_utils"] = mem_mu

    # disk
    # disk_busy: DISKBUSY, Percentage of time during which the disk is active.
    target = "DISKBUSY.csv"
    df_diskbusy = {}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target}"
        df = pd.read_csv(file, nrows=steps)
        df["disk_busy"] = df[DISK] / 100
        df_diskbusy[w] = df
    diskbusys = np.array([df_["disk_busy"].values for df_ in df_diskbusy.values()])
    # for w in workers:
    #     df_diskbusy[w]["disk_busy"][1:1000].plot(label=w)
    # plt.legend()
    diskbusy_mu = diskbusys.mean(0)
    assert len(diskbusy_mu) == steps
    df_mach["diskbusy"] = diskbusy_mu

    # disk_bsize/s: DISKBSIZE / 5, Disk Block Size - Total number of disk blocks that are read and written over the interval.
    target = "DISKBSIZE.csv"
    df_diskbsize = {}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target}"
        df = pd.read_csv(file, nrows=steps)
        df["disk_bsize/s"] = df[DISK] / FEQ
        df_diskbsize[w] = df
    diskbsizes = np.array([df_["disk_bsize/s"].values for df_ in df_diskbsize.values()])
    # for w in workers:
    #     df_diskbsize[w]["disk_bsize/s"][1:1000].plot(label=w)
    # plt.legend()
    diskbsize_mu = diskbsizes.mean(0)
    assert len(diskbsize_mu) == steps
    df_mach["disk_bsize/s"] = diskbsize_mu

    # disk_bytes/s
    target_dict = {"r": "DISKREAD.csv", "w": "DISKWRITE.csv"}
    df_diskbyte = {"r": {}, "w": {}}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target_dict['r']}"
        df = pd.read_csv(file, nrows=steps)
        df["disk_KB/s"] = df[DISK]
        df_diskbyte["r"][w] = df

        file = f"{src_path}/{w}/csv/{target_dict['w']}"
        df = pd.read_csv(file, nrows=steps)
        df["disk_KB/s"] = df[DISK]
        df_diskbyte["w"][w] = df
    diskbytes = np.array([df_["disk_KB/s"].values for df_ in df_diskbyte["r"].values()]) + \
                np.array([df_["disk_KB/s"].values for df_ in df_diskbyte["w"].values()])
    # for w in workers:
    #     df_diskbyte["r"][w]["disk_KB/s"][1:1000].plot(label=w)
    # plt.legend()
    # plt.show()
    # for w in workers:
    #     df_diskbytes"w"][w]["disk_KB/s"][1:1000].plot(label=w)
    # plt.legend()
    # plt.show() # <200MB/s on average
    diskbyte_mu = diskbytes.mean(0)
    assert len(diskbyte_mu) == steps
    df_mach["disk_KB/s"] = diskbyte_mu

    # disk_xfers/s
    target = "DISKXFER.csv"
    df_diskxfer = {}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target}"
        df = pd.read_csv(file, nrows=steps)
        df["disk_xfers/s"] = df[DISK]
        df_diskxfer[w] = df
    diskxfers = np.array([df_["disk_xfers/s"].values for df_ in df_diskxfer.values()])
    # for w in workers:
    #     df_diskxfer[w]["disk_xfers/s"][1:1000].plot(label=w)
    # plt.legend()
    diskxfer_mu = diskxfers.mean(0)
    assert len(diskxfer_mu) == steps
    df_mach["disk_xfers/s"] = diskxfer_mu

    # Network
    # net_bytes/s
    target = "NET.csv"
    df_netbyte = {}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target}"
        df = pd.read_csv(file, nrows=steps)
        df["net_KB/s"] = df[f"{NET}-read-KB/s"] + df[f"{NET}-write-KB/s"]
        df_netbyte[w] = df
    netbytes = np.array([df_["net_KB/s"].values for df_ in df_netbyte.values()])
    # for w in workers:
    #     df_netbyte[w]["net_KB/s"][1:1000].plot(label=w)
    # plt.legend()
    netbyte_mu = netbytes.mean(0)
    assert len(netbyte_mu) == steps
    df_mach["net_KB/s"] = netbyte_mu

    # net_xfers/s = ib0-write/s + ib0-read/s
    target = "NETPACKET.csv"
    df_netxfer = {}
    for w in workers:
        file = f"{src_path}/{w}/csv/{target}"
        df = pd.read_csv(file, nrows=steps)
        df["net_xfers/s"] = df[f"{NET}-read/s"] + df[f"{NET}-write/s"]
        df_netxfer[w] = df
    netxfers = np.array([df_["net_xfers/s"].values for df_ in df_netxfer.values()])
    # for w in workers:
    #     df_netxfer[w]["net_xfers/s"][1:1000].plot(label=w)
    # plt.legend()
    netxfer_mu = netxfers.mean(0)
    assert len(netxfer_mu) == steps
    df_mach["net_xfers/s"] = netxfer_mu

    ParquetUtils.parquet_write(df_mach, dst_path, "mach_traces.parquet", True)

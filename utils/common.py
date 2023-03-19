# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Created at 9/16/22

import os, json, pickle, datetime, ciso8601, glob, re
import time
import urllib.request
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


class JsonUtils(object):

    @staticmethod
    def load_json(file):
        assert os.path.exists(file), FileNotFoundError(file)
        with open(file) as f:
            try:
                return json.load(f)
            except:
                raise Exception(f"{f} cannot be parsed as a JSON file")

    @staticmethod
    def load_json_from_str(s: str):
        return json.loads(s)

    @staticmethod
    def print_dict(d: dict):
        print(json.dumps(d, indent=2))

    @staticmethod
    def load_json_from_url(url_str, timeout=10):
        try:
            with urllib.request.urlopen(url_str, timeout=timeout) as url:
                data = json.load(url)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            raise Exception(f"failed to load from {url_str}")
        return data

    @staticmethod
    def extract_json_list(l, keys: list):
        for key in keys:
            for l_ in l:
                assert key in l_, f"{key} is not in {l_}"
        return [[l_[key] for l_ in l] for key in keys]

    @staticmethod
    def save_json(d, des_file):
        json_data = json.dumps(d, indent=2)
        with open(des_file, "w") as f:
            f.write(json_data)

class PickleUtils(object):

    @staticmethod
    def save(obj, header, file_name, overwrite=False):
        path = f"{header}/{file_name}"
        if os.path.exists(path) and not overwrite:
            return f"{path} already exists"
        os.makedirs(header, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(header, file_name):
        path = f"{header}/{file_name}"
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_file(path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            return pickle.load(f)

class BenchmarkUtils(object):

    @staticmethod
    def get(benchmark: str):
        if benchmark.lower() == "tpch":
            return [str(i) for i in range(1, 23)]
        elif benchmark.lower() == "tpcds":
            return "1 2 3 4 5 6 7 8 9 10 11 12 13 14a 14b 15 16 17 18 19 20 21 22 23a 23b 24a 24b 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39a 39b 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99".split(
                " ")
        elif benchmark.lower() == "tpcxbb":
            return [str(i) for i in range(1, 31)]
        else:
            raise ValueError(f"{benchmark} is not supported")

    @staticmethod
    def get_workers(input: str):
        if input.lower() == "tpch":
            return ["node2", "node3", "node4", "node5", "node6"]
        elif input.lower() == "tpcds":
            return ["node8", "node9", "node10", "node11", "node12"]
        elif input.lower() == "debug":
            return ["node14", "node15", "node16", "node17", "node18"]
        elif input.lower() == "hex1":
            return ["node2", "node3", "node4", "node5", "node6"]
        elif input.lower() == "hex2":
            return ["node8", "node9", "node10", "node11", "node12"]
        elif input.lower() == "hex3":
            return ["node14", "node15", "node16", "node17", "node18"]
        else:
            raise ValueError(f"{input} is not supported")

    @staticmethod
    def get_sampled_q_signs(benchmark: str):
        if benchmark.lower() == "tpch":
            return "q1-1,q2-1,q3-3,q4-1,q5-2,q6-1,q7-2,q8-19,q9-18,q10-1,q11-1,q12-5,q13-1,q14-1,q15-1,q16-1,q17-1,q18-1,q19-1,q20-6,q21-2,q22-1".split(",")
        else:
            raise ValueError(benchmark)

    @staticmethod
    def extract_sampled_q_sign(benchmark: str, sign: str):
        if bool(re.match(r"^q[0-9]+-[0-9]+$", sign)):
            q_sign = sign
        else:
            try:
                q_sign = BenchmarkUtils.get_sampled_q_signs(benchmark)[int(sign) - 1]
            except:
                raise ValueError(sign)
        return q_sign

    @staticmethod
    def get_tid(q_sign) -> int:
        tid = int("".join([d for d in q_sign.split("-")[0] if d.isdigit()]))
        return tid

    @staticmethod
    def get_qid(q_sign) -> int:
        qid = int(q_sign.split("-")[1])
        return qid

class TimeUtils(object):

    @staticmethod
    def get_current_iso():
        ct = datetime.datetime.utcnow()
        # "2022-09-23T17:05:09.589GMT"
        return ct.astimezone().isoformat()

    @staticmethod
    def get_utc_timestamp(s: str, tz_ahead: int = 0) -> int:
        t = ciso8601.parse_datetime(f"{s}+0{tz_ahead}00").utctimetuple()
        return int(time.mktime(t))


class ParquetUtils(object):

    @staticmethod
    def parquet_read(header, file_name):
        path = f"{header}/{file_name}"
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return pq.read_table(path).to_pandas()

    @staticmethod
    def parquet_read_multiple(header, matches=None):
        """
        read all the parquet files under the directory header
        :param header: str
        :return: a dataFrame concatenated over all parquet files under header
        """
        names = [n for n in glob.glob(f"{header}/{'*.parquet' if matches is None else matches}")]
        if len(names) == 0:
            return None
        return pd.concat([ParquetUtils.parquet_read(header, name)
                          for name in os.listdir(header) if name[-8:] == ".parquet"])


    @staticmethod
    def parquet_write(df, header, file_name, overwrite=False):
        path = f"{header}/{file_name}"
        if os.path.exists(path) and not overwrite:
            return f"{path} already exists"
        os.makedirs(header, exist_ok=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)


class FileUtils(object):

    @staticmethod
    def read_file(filename):
        with open(filename, "r") as f:
            rows = f.read().strip()
        return rows

    @staticmethod
    def read_file_as_rows(filename):
        with open(filename, "r") as f:
            rows = f.read().strip().split("\n")
        return rows

    @staticmethod
    def read_1st_row(filename):
        with open(filename, "r") as f:
            row = f.readline().strip()
        return row

    @staticmethod
    def write_str(filename, s):
        with open(filename, "w") as f:
            f.write(s)

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.
    Defined in :numref:`sec_calculus`"""
    # use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.
    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.
    Defined in :numref:`sec_calculus`"""

    def has_one_axis(X):  # True if `X` (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None: axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
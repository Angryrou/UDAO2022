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
        self.parser.add_argument("--dst-path", type=str, default="outs/tpch_100_lhs/2.tabular")
        self.parser.add_argument("--url-header", type=str,
                                 default="http://10.0.0.1:18088/api/v1/applications/application_1663600377480")
        self.parser.add_argument("--url-suffix-start", type=int, default=3827, help="the number is inclusive")
        self.parser.add_argument("--url-suffix-end", type=int, default=83840, help="the number is inclusive")
        self.parser.add_argument("--num-processes", type=int, default=6)

    def parse(self):
        return self.parser.parse_args()


def extract_tabular(appid):
    """
    return ["id", "name", "q_sign", "knob_sign",
            "planDescription", "nodes", "edges",
            "start_timestamp", "latency", "err"]
    """
    url = f"{url_header}_{f'{appid:04}' if appid < 10000 else str(appid)}"
    try:
        data = JsonUtils.load_json_from_url(url)
        query = JsonUtils.load_json_from_url(url + "/sql")[1]
        _, q_sign, knob_sign = data["name"].split("_")
        return [
            appid, data["name"], q_sign, knob_sign,
            json.dumps(query["planDescription"]), json.dumps(query["nodes"]), json.dumps(query["edges"]),
            TimeUtils.get_utc_timestamp(query["submissionTime"][:-3]), query["duration"] / 1000, None
        ]
    except Exception as e:
        traceback.print_exc()
        print(f"{e} when url={url}/sql")
        return [
            None, None, None, None,
            None, None, None,
            None, None, str(e)
        ]


if __name__ == '__main__':
    args = Args().parse()
    bm = args.benchmark.lower()
    sf = args.scale_factor
    sampling = args.sampling
    dst_path = args.dst_path
    url_header = args.url_header
    url_suffix_start = args.url_suffix_start
    url_suffix_end = args.url_suffix_end
    n_processes = args.num_processes

    arg_list = [(appid, ) for appid in range(url_suffix_start, url_suffix_end + 1)]
    begin = time.time()
    with Pool(processes=n_processes) as pool:
        res = pool.starmap(extract_tabular, arg_list)
    print(f"generating urls cots {time.time() - begin}s")
    columns = ["id", "name", "q_sign", "knob_sign",
               "planDescription", "nodes", "edges", "start_timestamp", "latency", "err"]
    df_tmp = pd.DataFrame(res, columns=columns)
    os.makedirs(dst_path, exist_ok=True)
    df_tmp.to_csv(f"{dst_path}/{url_suffix_start}_{url_suffix_end}.csv", sep="\u0001")
import argparse
import os, json, time, glob
import sys

import pandas as pd

from utils.common import JsonUtils, TimeUtils, ParquetUtils, FileUtils


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--sampling", type=str, default="lhs")
        self.parser.add_argument("--dst-path-header", type=str, default="examples/trace/spark-parser/outs")
        self.parser.add_argument("--url-header", type=str,
                                 default="http://10.0.0.1:18088/api/v1/applications/application_1663600377480")
        self.parser.add_argument("--url-suffix-start", type=int, default=3827, help="the number is inclusive")
        self.parser.add_argument("--url-suffix-end", type=int, default=83840, help="the number is inclusive")
        self.parser.add_argument("--lamda", type=int, default=100)
        self.parser.add_argument("--target-url-path", type=str, default=None)
        # "examples/trace/spark-parser/outs/tpch_100_lhs/2.tabular/*_failed_urls.txt"
        self.parser.add_argument("--debug", type=int, default=0)

    def parse(self):
        return self.parser.parse_args()


if __name__ == '__main__':
    args = Args().parse()
    bm = args.benchmark.lower()
    sf = args.scale_factor
    sampling = args.sampling
    dst_path_header = args.dst_path_header
    dst_path = f"{dst_path_header}/{bm}_{sf}_{sampling}/2.tabular"
    os.makedirs(dst_path, exist_ok=True)
    lamda = args.lamda
    debug = False if args.debug == 0 else True

    existed_df_tabular = ParquetUtils.parquet_read_multiple(dst_path)
    existed_df_tabular = existed_df_tabular[existed_df_tabular.err.isna()]
    existed_appids = set(existed_df_tabular["id"]) if existed_df_tabular is not None else set()

    begin = time.time()
    if args.target_url_path is not None:
        files = glob.glob(args.target_url_path)
        urls = []
        for file in files:
            rows = [u[:-4] for u in FileUtils.read_file_as_rows(file)]
            urls += rows
        path_sign = args.target_url_path.split("/")[-1]
    else:
        url_header = args.url_header
        url_suffix_start = args.url_suffix_start
        url_suffix_end = args.url_suffix_end
        urls = [f"{url_header}_{f'{url_suffix:04}' if url_suffix < 10000 else str(url_suffix)}"
                for i, url_suffix in enumerate(range(url_suffix_start, url_suffix_end + 1))]
        path_sign = f"{url_suffix_start}_{url_suffix_end}"

    n_queries = len(urls)
    print(f"Got {n_queries} queries to parse, with {len(existed_appids)} existed.")
    res = [None] * len(urls)
    for i, url in enumerate(urls):
        appid = url.split("/")[-1]
        if appid in existed_appids:
            if debug:
                print(f"found {appid} in the existing Parquets.")
            continue
        try:
            data = JsonUtils.load_json_from_url(url)
            query = JsonUtils.load_json_from_url(url + "/sql", 30)[1]
            _, q_sign, knob_sign = data["name"].split("_")
            if debug:
                print(f"extract {appid} from urls.")
            elif (i + 1) % (n_queries // lamda) == 0:
                print(f"finished {i + 1}/{n_queries}, cost {time.time() - begin}s")
            res[i] = [
                appid, data["name"], q_sign, knob_sign,
                json.dumps(query["planDescription"]), json.dumps(query["nodes"]), json.dumps(query["edges"]),
                TimeUtils.get_utc_timestamp(query["submissionTime"][:-3]), query["duration"] / 1000, None
            ]
        except KeyboardInterrupt:
            if args.target_url_path is not None:
                sys.exit(1)
            else:
                url_suffix_end = int(url.split("_")[-1]) - 1
                path_sign = f"{url_suffix_start}_{url_suffix_end}"
                res = [r for r in res[:i] if r is not None]
                break
        except Exception as e:
            print(f"{e} when url={url}")
            res[i] = [
                appid, None, None, None,
                None, None, None,
                None, None, str(e)
            ]
            with open(f"{dst_path}/{int(begin)}_failed_urls.txt", "a+") as f:
                f.write(f"{url}/sql\n")

    res = [r for r in res if r is not None]
    print(f"generating {len(res)} urls cots {time.time() - begin}s")
    columns = ["id", "name", "q_sign", "knob_sign",
               "planDescription", "nodes", "edges", "start_timestamp", "latency", "err"]
    df_tmp = pd.DataFrame(res, columns=columns)
    ParquetUtils.parquet_write(
        df_tmp, dst_path, f"{int(begin)}_query_traces_{path_sign}.parquet")
    print(f"results written into {dst_path}/{int(begin)}_query_traces_{path_sign}.parquet")

# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: download the traces from JSON of REST APIs to CSV files
#
# Created at 12/12/22

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
    dst_path = f"{dst_path_header}/{bm}_{sf}_{sampling}/3.tabular_stages"
    os.makedirs(dst_path, exist_ok=True)
    lamda = args.lamda
    debug = False if args.debug == 0 else True

    existed_df_tabular = ParquetUtils.parquet_read_multiple(dst_path)
    if existed_df_tabular is None:
        existed_appids = set()
    else:
        existed_df_tabular = existed_df_tabular[existed_df_tabular.err.isna()]
        existed_appids = set(existed_df_tabular["id"])

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
    res = []
    for i, url in enumerate(urls):
        appid = url.split("/")[-1]
        if appid in existed_appids:
            if debug:
                print(f"found {appid} in the existing Parquets.")
            continue
        try:
            data = JsonUtils.load_json_from_url(url)
            stages = JsonUtils.load_json_from_url(url + "/stages", 30)
            cur_res = []
            for s in stages:
                if s["status"] != "COMPLETE":
                    continue
                stage_lat = TimeUtils.get_utc_timestamp(s["completionTime"][:-3]) - \
                            TimeUtils.get_utc_timestamp(s["firstTaskLaunchedTime"][:-3])

                stage = JsonUtils.load_json_from_url(url + "/stages/" + str(s["stageId"]) + "/" + str(s["attemptId"]))
                stage_dt = sum([v["taskTime"] for k, v in stage["executorSummary"].items()])
                cur_res.append([
                    appid, s["stageId"],
                    TimeUtils.get_utc_timestamp(s["firstTaskLaunchedTime"][:-3]), stage_lat, stage_dt,
                    s["numTasks"], s["inputBytes"], s["inputRecords"], s["shuffleReadBytes"], s["shuffleReadRecords"],
                    s["outputBytes"], s["outputRecords"], s["ShuffleWriteBytes"], s["shuffleWriteRecords"], None
                ])
            if debug:
                print(f"extract {appid} from urls: \n {cur_res}")
            elif (i + 1) % (n_queries // lamda) == 0:
                print(f"finished {i + 1}/{n_queries}, cost {time.time() - begin}s")
            res += cur_res
        except KeyboardInterrupt:
            if args.target_url_path is not None:
                sys.exit(1)
            else:
                url_suffix_end = int(url.split("_")[-1]) - 1
                path_sign = f"{url_suffix_start}_{url_suffix_end}"
                break
        except Exception as e:
            print(f"{e} when url={url}")
            res += [
                appid, None,
                None, None, None,
                None, None, None, None, None,
                None, None, None, None, str(e)
            ]
            with open(f"{dst_path}/{int(begin)}_failed_urls.txt", "a+") as f:
                f.write(f"{url}/stages\n")
            if debug:
                break

    print(f"generating {len(res)} stages from {len(urls)} urls costs {time.time() - begin}s")
    columns = ["id", "stage_id", "first_task_launched_time", "stage_latency", "stage_dt",
               "task_num", "input_bytes", "input_records", "sr_bytes", "sr_records",
               "output_bytes", "output_records", "sw_bytes", "sw_records", "err"]
    df_tmp = pd.DataFrame(res, columns=columns)
    ParquetUtils.parquet_write(
        df_tmp, dst_path, f"{int(begin)}_query_traces_{path_sign}.parquet")
    print(f"results written into {dst_path}/{int(begin)}_query_traces_{path_sign}.parquet")

# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: dump LHS results
#
# Created at 10/28/22

# maintain a directory for <n_templates> CSV files.
import argparse
import os.path

import numpy as np
import pandas as pd

from utils.common import BenchmarkUtils, PickleUtils, JsonUtils, FileUtils
from trace.parser.spark import get_cloud_cost

class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("-k", "--knob-meta-file", type=str, default="resources/knob-meta/spark.json")
        self.parser.add_argument("-s", "--seed", type=int, default=42)
        self.parser.add_argument("--script-header", type=str, default="resources/scripts/tpch-lhs")
        self.parser.add_argument("--cache-header", type=str, default="examples/trace/spark/cache")
        self.parser.add_argument("--log-header", type=str, default="examples/trace/spark/6.run_all_pressure_test_tpcxbb/log")
        self.parser.add_argument("--num-templates", type=int, default=22)
        self.parser.add_argument("--debug", type=int, default=0)

    def parse(self):
        return self.parser.parse_args()

def get_obj(tid, qid, conf_i, log_header):
    knob_sign = conf_i["knob_sign"]
    file_prefix = f"q{tid}-{qid}_{knob_sign}"
    file_path = f"{log_header}/{file_prefix}.log"
    if os.path.exists(file_path):
        log = FileUtils.read_file(file_path).lower()
        if "error" in log:
            return np.inf, np.inf
        rows = log.split("\n")
        appid, url_head = rows[0], rows[1]
        url_str = f"{url_head}/api/v1/applications/{appid}"
        data = JsonUtils.load_json_from_url(url_str)
        if data is not None:
            lat = data["attempts"][0]["duration"] / 1000  # seconds
            cost = get_cloud_cost(
                lat=lat,
                mem=int(conf_i["spark.executor.memory"][:-1]),
                cores=int(conf_i["spark.executor.cores"]),
                nexec=int(conf_i["spark.executor.instances"])
            )
            return lat, cost
        else:
            raise Exception(f"failed to analyze {url_str}")
    else:
        return -1, -1

if __name__ == '__main__':

    args = Args().parse()

    benchmark = args.benchmark
    seed = args.seed
    script_header = args.script_header
    out_header = args.out_header
    cache_header = os.path.join(args.cache_header, benchmark.lower())
    log_header = args.log_header
    n_templates = args.num_templates
    debug = args.debug

    templates = BenchmarkUtils.get(benchmark)
    qpt = 100
    assert benchmark == "TPCxBB" and n_templates == 30
    conf_df = pd.concat(PickleUtils.load(cache_header, f"lhs_{n_templates}x{qpt}.pkl"))
    conf_df_tmp = conf_df.groupby(level=[0, 1]).size()
    conf_df_off_mask = conf_df_tmp[conf_df_tmp > 10]
    conf_df_off = conf_df.reset_index(level=2).loc[conf_df_off_mask.index].rename(columns={"level_2": "knob_sign"})

    obj_df_dict = {}
    for tid in templates:
        df_t = conf_df_off.loc[tid]
        qids = df_t.index.unique().to_list()
        obj_df_dict_t = {}
        for qid in qids:
            cache_header_q = f"{cache_header}/lhs_{n_templates}x{qpt}_objs"
            try:
                df_q = PickleUtils.load(cache_header_q, f"q{tid}-{qid}.pkl")
                print(f"found df_q at {cache_header_q}/q{tid}-{qid}.pkl")
            except:
                print(f"not found df_q, start generating")
                df_q = df_t.loc[qid]
                lat_list, cost_list = [-1] * len(df_q), [-1] * len(df_q)
                for i, (_, conf_i) in enumerate(df_q.iterrows()):
                    if (i + 1) % 100 == 0:
                        print(f"finished parsing {i+1}/{len(df_q)} configurations")
                    lat_list[i], cost_list[i] = get_obj(tid, qid, df_q, log_header)
                df_q["lat"] = lat_list
                df_q["cost"] = cost_list
                PickleUtils.save(df_q, cache_header_q, f"q{tid}-{qid}.pkl")
                print(f"geneerated df_q at {cache_header_q}/q{tid}={qid}.pkl")
            obj_df_dict_t[qid] = df_q
        obj_df_dict[tid] = obj_df_dict_t
    PickleUtils.save(obj_df_dict, cache_header, f"lhs_{n_templates}x{qpt}_objs.pkl")

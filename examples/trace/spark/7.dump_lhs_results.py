# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: dump LHS results
#
# Created at 10/28/22

# maintain a directory for <n_templates> CSV files.
import argparse
import os.path

import pandas as pd

from utils.common import BenchmarkUtils, PickleUtils, JsonUtils
from trace.parser.spark import get_cloud_cost

class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("-k", "--knob-meta-file", type=str, default="resources/knob-meta/spark.json")
        self.parser.add_argument("-s", "--seed", type=int, default=42)
        self.parser.add_argument("--script-header", type=str, default="resources/scripts/tpch-lhs")
        self.parser.add_argument("--out-header", type=str, default="examples/trace/spark/7.run_all_pressure_test")
        self.parser.add_argument("--cache-header", type=str, default="examples/trace/spark/cache")
        self.parser.add_argument("--num-templates", type=int, default=22)
        self.parser.add_argument("--num-queries-per-template-to-run", type=int, default=400)
        self.parser.add_argument("--url-header", type=str,
                                 default="node1-opa:18088/api/v1/applications/application_1663600377480")
        self.parser.add_argument("--url-suffix-start", type=int, default=3827, help="the number is inclusive")
        self.parser.add_argument("--url-suffix-end", type=int, default=83841, help="the number is inclusive")
        self.parser.add_argument("--debug", type=int, default=0)

    def parse(self):
        return self.parser.parse_args()


if __name__ == '__main__':

    args = Args().parse()

    benchmark = args.benchmark
    seed = args.seed
    script_header = args.script_header
    out_header = args.out_header
    cache_header = args.cache_header
    n_templates = args.num_templates
    qpt = args.num_queries_per_template_to_run
    debug = args.debug
    url_header = args.url_header
    url_suffix_start = args.url_suffix_start
    url_suffix_end = args.url_suffix_end

    templates = BenchmarkUtils.get(benchmark)
    if benchmark == "TPCH":
        assert n_templates == 22
        qpt_total = 3637
    elif benchmark == "TPCDS":
        assert n_templates == 103
        qpt_total = 777
    else:
        raise ValueError(benchmark)
    if debug:
        qpt_total = qpt

    assert os.path.exists(os.path.join(cache_header, f"lhs_{n_templates}x{qpt_total}.pkl")), \
        f'lhs conf file not found {os.path.join(cache_header, f"lhs_{n_templates}x{qpt_total}.pkl")}'
    conf_df_dict = PickleUtils.load(cache_header, f"lhs_{n_templates}x{qpt_total}.pkl")
    total_confs = sum(v.shape[0] for k, v in conf_df_dict.items())
    assert total_confs == (url_suffix_end - url_suffix_start + 1)

    lhs_dict = {k: {} for k in conf_df_dict.keys()}
    for i, appid in enumerate(range(url_suffix_start, url_suffix_end + 1)):
        appid_str = f"{appid:04}" if appid < 10000 else str(appid)
        url_str = f"{url_header}_{appid_str}"
        data = JsonUtils.load_json_from_url(url_str)
        if data is not None:
            _, q_sign, knob_sign = data["name"].split("_")
            tid = q_sign.split("-")[0]
            lat = data["attempts"][0]["duration"] # seconds
            conf_dict = conf_df_dict[tid].loc[knob_sign].to_dict()
            cost = get_cloud_cost(
                lat=lat,
                mem=int(conf_dict["spark.executor.memory"][:-1]),
                cores=int(conf_dict["spark.executor.cores"]),
                nexec=int(conf_dict["spark.executor.instances"])
            )
            lhs_dict[tid][knob_sign] = {"l": lat, "c": cost}
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{n_templates * qpt_total} traces has been analyzed.")

    for tid in lhs_dict.keys():
        assert len(lhs_dict[tid]) == qpt_total

    obj_df_dict = {k: pd.DataFrame.from_dict(v, orient="index") for k, v in lhs_dict.items()}
    PickleUtils.save(obj_df_dict, cache_header, f"lhs_{n_templates}x{qpt_total}_objs.pkl")

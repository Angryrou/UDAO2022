import argparse, csv
import os, json

from utils.common import JsonUtils, TimeUtils


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--sampling", type=str, default="lhs")
        self.parser.add_argument("--dst-path", type=str, default="outs/lhs/2.tabular")
        self.parser.add_argument("--url-header", type=str,
                                 default="http://10.0.0.1:18088/api/v1/applications/application_1663600377480")
        self.parser.add_argument("--url-suffix-start", type=int, default=3827, help="the number is inclusive")
        self.parser.add_argument("--url-suffix-end", type=int, default=83840, help="the number is inclusive")

    def parse(self):
        return self.parser.parse_args()


def extract_tabular(url):
    JsonUtils.load_json_from_url(url)


if __name__ == '__main__':
    args = Args().parse()
    benchmark = args.benchmark
    sf = args.scale_factor
    dst_path = args.dst_path
    url_header = args.url_header
    url_suffix_start = args.url_suffix_start
    url_suffix_end = args.url_suffix_end

    dbname = f"{benchmark}_{sf}"
    all = [{}] * (url_suffix_end - url_suffix_start + 1)

    for i, appid in enumerate(range(url_suffix_start, url_suffix_end + 1)):
        appid_str = f"{appid:04}" if appid < 10000 else str(appid)
        url_str = f"{url_header}_{appid_str}"
        print(url_str)

    os.makedirs(dst_path, exist_ok=True)
    with open(f"{dst_path}/tabular_{url_suffix_start}_{url_suffix_end}.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter='\u0001')
        writer.writerow(
            ["q_sign", "knob_sign", "timestamp_begin_all", "latency_all", "planDescription", "nodes", "edges",
             "timestamp_begin_query", "latency_query"])

        for i, appid in enumerate(range(url_suffix_start, url_suffix_end + 1)):
            appid_str = f"{appid:04}" if appid < 10000 else str(appid)
            url_str = f"{url_header}_{appid_str}"
            data = JsonUtils.load_json_from_url(url_str)
            if data is not None:
                _, q_sign, knob_sign = data["name"].split("_")
            else:
                raise Exception(f"failed to analyze {url_str}")
            query = JsonUtils.load_json_from_url(f"{url_str}/sql")[1]
            assert query["status"] == "COMPLETED"

            writer.writerow([
                q_sign,
                knob_sign,
                TimeUtils.get_utc_timestamp(data["attempts"][0]["startTime"][:-3]),
                data["attempts"][0]["duration"] / 1000,
                json.dumps(query["planDescription"]),
                json.dumps(query["nodes"]),
                json.dumps(query["edges"]),
                TimeUtils.get_utc_timestamp(query["submissionTime"][:-3]),
                query["duration"] / 1000
            ])

import argparse, json, os, traceback

from utils.common import JsonUtils, TimeUtils
from pyspark.sql import SparkSession


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--sampling", type=str, default="lhs")
        self.parser.add_argument("--dst-path", type=str, default="examples/traces/spark-parser/outs/lhs/2.tabular")
        self.parser.add_argument("--mach-traces-path", type=str,
                                 default="examples/trace/spark-parser/outs/lhs/1.mach/mach_traces.parquet")
        self.parser.add_argument("--url-header", type=str,
                                 default="http://10.0.0.1:18088/api/v1/applications/application_1663600377480")
        self.parser.add_argument("--url-suffix-start", type=int, default=3827, help="the number is inclusive")
        self.parser.add_argument("--url-suffix-end", type=int, default=83840, help="the number is inclusive")

    def parse(self):
        return self.parser.parse_args()


# def extract_tabular(url):
#     data = JsonUtils.load_json_from_url(url)
#     query = JsonUtils.load_json_from_url(url + "/sql")[1]
#     _, q_sign, knob_sign = data["name"].split("_")
#     return json.dumps({
#         "id": data["id"],
#         "name": data["name"],
#         "q_sign": q_sign,
#         "knob_sign": knob_sign,
#         "startTime_all": TimeUtils.get_utc_timestamp(data["attempts"][0]["startTime"][:-3]),
#         "latency_all": data["attempts"][0]["duration"] / 1000,
#         "planDescription": query["planDescription"],
#         "nodes": query["nodes"],
#         "edges": query["edges"],
#         "startTime_query": TimeUtils.get_utc_timestamp(query["submissionTime"][:-3]),
#         "latency_query": query["duration"] / 1000
#     })


def extract_tabular(url):
    try:
        data = JsonUtils.load_json_from_url(url)
        data2 = JsonUtils.load_json_from_url(url + "/sql")
        query = data2[1]
        _, q_sign, knob_sign = data["name"].split("_")
        res = json.dumps({
            "id": data["id"],
            "name": data["name"],
            "q_sign": q_sign,
            "knob_sign": knob_sign,
            "startTime_all": TimeUtils.get_utc_timestamp(data["attempts"][0]["startTime"][:-3]),
            "latency_all": data["attempts"][0]["duration"] / 1000,
            "planDescription": query["planDescription"],
            "nodes": query["nodes"],
            "edges": query["edges"],
            "startTime_query": TimeUtils.get_utc_timestamp(query["submissionTime"][:-3]),
            "latency_query": query["duration"] / 1000
        })
        return res
    except Exception as e:
        print(f"{e} when url={url}, data2={data2}")
        traceback.print_exc()
        return None


if __name__ == '__main__':
    args = Args().parse()
    benchmark = args.benchmark
    sf = args.scale_factor
    dst_path = args.dst_path
    mach_trace = args.mach_traces_path
    url_header = args.url_header
    url_suffix_start = args.url_suffix_start
    url_suffix_end = args.url_suffix_end
    dbname = f"{benchmark}_{sf}"

    columns = ["id", "name", "q_sign", "knob_sign", "startTime_all", "latency_all", "planDescription",
               "nodes", "edges", "startTime_query", "latency_query"]

    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    sc = spark.sparkContext
    urls = sc.parallelize([f"{url_header}_{f'{appid:04}' if appid < 10000 else str(appid)}"
                           for i, appid in enumerate(range(url_suffix_start, url_suffix_end + 1))])

    rdd = urls.map(lambda url: extract_tabular(url))
    df = spark.read.json(rdd).select(columns)
    df.cache()
    df.count()
    df.show()
    df.write.format("csv").option("header", True).mode("overwrite").option("sep", "\OOO1").save(
        f"file://{os.getcwd()}/{dst_path}/{dbname}_tabular.csv")

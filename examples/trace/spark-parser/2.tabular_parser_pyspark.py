import argparse, json, os, traceback, time, ciso8601
import urllib.request

from pyspark import sql
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StructField, StructType, StringType, LongType


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--sampling", type=str, default="lhs")
        self.parser.add_argument("--url-header", type=str,
                                 default="http://10.0.0.1:18088/api/v1/applications/application_1663600377480")
        self.parser.add_argument("--url-suffix-start", type=int, default=3827, help="the number is inclusive")
        self.parser.add_argument("--url-suffix-end", type=int, default=83840, help="the number is inclusive")

    def parse(self):
        return self.parser.parse_args()

def load_json_from_url(url_str):
    try:
        with urllib.request.urlopen(url_str) as url:
            data = json.load(url)
    except:
        traceback.print_exc()
        print(f"failed to load from {url_str}")
        return None
    return data

def get_utc_timestamp(s: str, tz_ahead: int = 0) -> int:
    t = ciso8601.parse_datetime(f"{s}+0{tz_ahead}00").utctimetuple()
    return int(time.mktime(t))

def extract_tabular(url):
    remaining_trials = 5
    appid = url.split("/")[-1]
    try:
        data = load_json_from_url(url)
        while remaining_trials > 0:
            data2 = load_json_from_url(url + "/sql")
            if len(data2) == 2:
                break
            remaining_trials -= 1
            print(f"{url}/sql failed, remaining trials: {remaining_trials}")
            time.sleep(1)
        query = data2[1]
        _, q_sign, knob_sign = data["name"].split("_")
        return json.dumps({
            "id": appid,
            "name": data["name"],
            "q_sign": q_sign,
            "knob_sign": knob_sign,
            "startTime_all": get_utc_timestamp(data["attempts"][0]["startTime"][:-3]),
            "latency_all": data["attempts"][0]["duration"] / 1000,
            "planDescription": json.dumps(query["planDescription"]),
            "nodes": json.dumps(query["nodes"]),
            "edges":json.dumps(query["edges"]),
            "startTime_query": get_utc_timestamp(query["submissionTime"][:-3]),
            "latency_query": query["duration"] / 1000,
            "error": None
        })
    except Exception as e:
        traceback.print_exc()
        print(f"{e} when url={url}/sql, data2={data2}")
        return json.dumps({
            "id": appid,
            "name": None,
            "q_sign": None,
            "knob_sign": None,
            "startTime_all": None,
            "latency_all": None,
            "planDescription": None,
            "nodes": None,
            "edges": None,
            "startTime_query": None,
            "latency_query": None,
            "error": str(e)
        })


if __name__ == '__main__':
    args = Args().parse()
    bm, sf, sampling = args.benchmark.lower(), args.scale_factor, args.sampling
    dbname = f"{bm}_{sf}_traces"
    url_header = args.url_header
    url_suffix_start = args.url_suffix_start
    url_suffix_end = args.url_suffix_end

    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    sc = spark.sparkContext
    spark.sql(f"create database if not exists {dbname}")
    spark.sql(f"use {dbname}")
    print(f"use {dbname}")

    # get R1 (mach_traces)
    df_mach = spark.read.parquet(f"/user/spark_benchmark/{bm}_{sf}/traces/{sampling}_mach_traces.parquet").toDF(
        "timestamp", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8")
    df_mach.write.mode("overwrite").saveAsTable(f"{dbname}.{sampling}_mach_traces")

    # get R2 (query_traces_tmp)
    query_tmp_schema = StructType([
        StructField('id', StringType(), False),
        StructField('name', StringType(), True),
        StructField('q_sign', StringType(), True),
        StructField('knob_sign', StringType(), True),
        StructField('startTime_all', LongType(), True),
        StructField('latency_all', DoubleType(), True),
        StructField('planDescription', StringType(), True),
        StructField('nodes', StringType(), True),
        StructField('edges', StringType(), True),
        StructField('startTime_query', LongType(), True),
        StructField('latency_query', DoubleType(), True),
        StructField('error', StringType(), True)
    ])
    urls = sc.parallelize([f"{url_header}_{f'{appid:04}' if appid < 10000 else str(appid)}"
                           for i, appid in enumerate(range(url_suffix_start, url_suffix_end + 1))])
    rdd = urls.map(lambda url: extract_tabular(url))
    rdd.cache()
    df_tmp = spark.read.json(rdd, schema=query_tmp_schema)
    # print(f"failed to parse {df_tmp.filter('error is not null').count()}")
    df_tmp.createOrReplaceTempView("R2")
    df_query = spark.sql(f"""\
    select \
        R2.id, R2.name, R2.q_sign, R2.knob_sign, R2.planDescription, R2.nodes, R2.edges, \
        R1.m1, R1.m2, R1.m3, R1.m4, R1.m5, R1.m6, R1.m7, R1.m8, R2.startTime_query, R2.latency_query, \
        split(R2.q_sign, "-")[0] as tid \
    from {sampling}_mach_traces R1, R2, ( \
        select R2.id as id, R2.startTime_query as t1, max(R1.timestamp) as t2 \
        from {sampling}_mach_traces R1, R2 \
        where R2.error is null and R1.timestamp < R2.startTime_query \
        group by R2.id, R2.id, R2.startTime_query \
    ) as R3 \
    where R1.timestamp = R3.t2 and R2.id = R3.id and R2.error is null
    """)
    df_query.write.mode("overwrite").saveAsTable(f"{dbname}.{sampling}_query_traces")
    df_query.repartition("tid").write.mode("overwrite").partitionBy("tid").format("csv"). \
        options(delimiter="\u0001").save(f"/user/spark_benchmark/{bm}_{sf}/traces/{sampling}_query_traces")
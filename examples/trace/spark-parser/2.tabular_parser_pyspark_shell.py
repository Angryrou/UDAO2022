import json, traceback, time, ciso8601
import urllib.request

from pyspark import sql
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StructField, StructType, StringType, LongType

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

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
sc = spark.sparkContext

# url_suffix_start = 3827
# url_suffix_end = 3926
url_suffix_start = 83840 - 100 + 1
url_suffix_end = 83840
url_header = "http://10.0.0.1:18088/api/v1/applications/application_1663600377480"
columns = ["id", "name", "q_sign", "knob_sign", "startTime_all", "latency_all", "planDescription",
           "nodes", "edges", "startTime_query", "latency_query"]


dbname = "tpch_100_lhs"
sql(f"create database if not exists {dbname}").show()
sql(f"use {dbname}").show()
df_mach = spark.read.parquet("/user/spark_benchmark/tpch_100/traces/lhs/mach_traces.parquet").toDF(
    "timestamp", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8")
df_mach.write.mode("overwrite").saveAsTable(f"{dbname}.mach_traces")


# root (original)
#  |-- id: string (nullable = true)
#  |-- name: string (nullable = true)
#  |-- q_sign: string (nullable = true)
#  |-- knob_sign: string (nullable = true)
#  |-- startTime_all: long (nullable = true)
#  |-- latency_all: double (nullable = true)
#  |-- planDescription: string (nullable = true)
#  |-- nodes: array (nullable = true)
#  |    |-- element: struct (containsNull = true)
#  |    |    |-- metrics: array (nullable = true)
#  |    |    |    |-- element: struct (containsNull = true)
#  |    |    |    |    |-- name: string (nullable = true)
#  |    |    |    |    |-- value: string (nullable = true)
#  |    |    |-- nodeId: long (nullable = true)
#  |    |    |-- nodeName: string (nullable = true)
#  |    |    |-- wholeStageCodegenId: long (nullable = true)
#  |-- edges: array (nullable = true)
#  |    |-- element: struct (containsNull = true)
#  |    |    |-- fromId: long (nullable = true)
#  |    |    |-- toId: long (nullable = true)
#  |-- startTime_query: long (nullable = true)
#  |-- latency_query: double (nullable = true)

schema = StructType([
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
df = spark.read.json(rdd, schema=schema)
df.cache()
df.createOrReplaceTempView("R2")

df_query = sql(f"""\
select \
    R2.id, R2.name, R2.q_sign, R2.knob_sign, R2.planDescription, R2.nodes, R2.edges, \
    R1.m1, R1.m2, R1.m3, R1.m4, R1.m5, R1.m6, R1.m7, R1.m8, R2.startTime_query, R2.latency_query, \
    split(R2.q_sign, "-")[0] as tid \
from mach_traces R1, R2, ( \
    select R2.id as id, R2.startTime_query as t1, max(R1.timestamp) as t2 \
    from mach_traces R1, R2 \
    where R2.error is null and R1.timestamp < R2.startTime_query \
    group by R2.id, R2.id, R2.startTime_query \
) as R3 \
where R1.timestamp = R3.t2 and R2.id = R3.id and R2.error is null
""")
df_query.cache()
df_query.repartition("tid").write.mode("overwrite").partitionBy("tid").format("csv").\
    options(delimiter="\u0001").save("/user/spark_benchmark/tpch_100/traces/lhs/query_trace")
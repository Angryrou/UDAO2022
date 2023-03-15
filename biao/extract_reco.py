# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 14/03/2023
import glob
import os
import pandas as pd

from trace.parser.spark import get_cloud_cost
from utils.common import PickleUtils, FileUtils, JsonUtils
from utils.data.configurations import SparkKnobs

def sqldt_from_appid(url_header, appid, if_full_plan=False):
    url_str = f"{url_header}/{appid}"
    try:
        data = JsonUtils.load_json_from_url(url_str)
        sql = JsonUtils.load_json_from_url(f"{url_str}/sql/1")
        if sql["status"] != "COMPLETED":
            print(f"failure detected at {url_str}/sql/1")
            if if_full_plan:
                return "", "", -1, -1, None
            return "", "", -1, -1
        lat = sql["duration"] / 1000  # secs
        _, q_sign, knob_sign = data["name"].split("_")
        knobs = knob_sign.split(",")
        k1, k2, k3 = knobs[0], knobs[1], knobs[2]
        mem = int(k1) * 2
        cores = int(k2)
        nexec = int(k3)
        cost = get_cloud_cost(lat, mem, cores, nexec)
        print(f"got {q_sign}_{knob_sign}")
        if if_full_plan:
            return q_sign, knob_sign, lat, cost, sql
        return q_sign, knob_sign, lat, cost
    except Exception as e:
        print(f"failed to get {url_str}/sql, {e}")
        raise Exception(e)

DATA_COLNS = ["q_sign", "knob_sign", "lat", "cost"]
bm, sf, pj = "tpch", 100, "tpch_100"
debug = False
query_header = "resources/tpch-kit/spark-sqls"
aqe_sign = "aqe_off"
reco_header = "biao/outs"
seed = 42
api_header="http://10.0.0.1:18088/api/v1/applications"
spark_knobs = SparkKnobs(meta_file="resources/knob-meta/spark.json")

pkls = os.listdir(reco_header)
for pkl in pkls:
    res_pkl = f"res_{pkl}"
    res_dict = {}
    for q_sign, df in PickleUtils.load(reco_header, pkl).items():
        sh = f"examples/trace/spark/internal/2.knob_hp_tuning/{bm}_{aqe_sign}/{q_sign}"
        conf_df = spark_knobs.df_knob2conf(df)
        knob_signs = conf_df.index.to_list()
        for knob_sign in knob_signs:
            file_prefix = f"{q_sign}_{knob_sign}"
            assert len(glob.glob(f"{sh}/{file_prefix}*.dts")) > 0
            try:
                obj_df_i = PickleUtils.load(sh, f"{file_prefix}_objs.pkl")
                print(f"found {file_prefix}_obj.pkl")
            except:
                appids = [FileUtils.read_1st_row(p) for p in glob.glob(f"{sh}/{file_prefix}*.log")]
                ret = [sqldt_from_appid(api_header, appid) for appid in appids]
                obj_df_i = pd.DataFrame(data=ret, columns=DATA_COLNS)
                obj_df_i = obj_df_i[obj_df_i["lat"] != -1]
                PickleUtils.save(obj_df_i, sh, file_name=f"{file_prefix}_objs.pkl")
                
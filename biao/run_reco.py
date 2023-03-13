# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 12/03/2023

from utils.common import PickleUtils, BenchmarkUtils
import os
import pandas as pd

from utils.data.collect import run_q_confs
from utils.data.configurations import SparkKnobs

bm, sf, pj = "tpch", 100, "tpch_100"
debug = False
query_header = "resources/tpch-kit/spark-sqls"
aqe_sign = "aqe_off"
reco_header = "biao/outs"
seed = 42
workers = BenchmarkUtils.get_workers("hex1")

pkls = os.listdir(reco_header)
reco_all_dict = {}
for pkl in pkls:
     for q_sign, df in PickleUtils.load(reco_header, pkl).items():
        if q_sign in reco_all_dict:
            reco_all_dict[q_sign] = pd.concat([df, reco_all_dict[q_sign]])
        else:
            reco_all_dict[q_sign] = df

print("total runs")
for q_sign, df in reco_all_dict.items():
    print(f"{q_sign}: {df.drop_duplicates().shape[0]}")

spark_knobs = SparkKnobs(meta_file="resources/knob-meta/spark.json")
for pkl in pkls:
    reco_dict = PickleUtils.load(reco_header, pkl)
    print(f"start running for {pkl}")
    for q_sign, df in reco_dict.items():
        print(f"prepared to run {len(df)} recommended PO configurations for {q_sign}")
        script_header = f"examples/trace/spark/internal/2.knob_hp_tuning/{bm}_{aqe_sign}/{q_sign}"
        conf_df = spark_knobs.df_knob2conf(df)
        run_q_confs(
            bm=bm, sf=sf, spark_knobs=spark_knobs, query_header=query_header,
            out_header=script_header, seed=seed, workers=workers,
            n_trials=3, debug=debug, q_sign=q_sign, conf_df=conf_df, if_aqe=False)
    print(f"finished running for {pkl}")
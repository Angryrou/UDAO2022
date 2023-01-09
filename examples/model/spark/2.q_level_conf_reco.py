# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: used weighted sum to recommend a Pareto set.
#
# Created at 08/01/2023

import os
import time

from trace.parser.spark import get_cloud_cost
from utils.common import BenchmarkUtils, PickleUtils, TimeUtils
from utils.data.collect import run_q_confs
from utils.data.configurations import SparkKnobs
from utils.model.args import ArgsRecoQ
from utils.model.parameters import set_data_params, get_gpus
from utils.model.proxy import ModelProxy
from utils.model.utils import expose_data, analyze_cols, add_pe, prepare_data_for_opt, \
    get_sample_spark_knobs
from utils.optimization.moo_utils import is_pareto_efficient

import numpy as np
import pandas as pd

args = ArgsRecoQ().parse()
print(args)

bm, sf, pj = args.benchmark, args.scale_factor, f"{args.benchmark}_{args.scale_factor}"
debug = False if args.debug == 0 else True
run = False if args.run == 0 else True
seed = args.seed
data_header = f"{args.data_header}/{pj}"
query_header, out_header = args.query_header, args.out_header
assert os.path.exists(data_header), f"data not exists at {data_header}"
assert args.granularity in ("Q", "QS")
assert args.model_name == "GTN"
model_name = args.model_name
obj = args.obj
ckp_sign = args.ckp_sign
n_samples = args.n_samples
gpus, device = get_gpus(args.gpu)

print("1. preparing data and model")
data_params = set_data_params(args)
dfs, ds_dict, col_dict, minmax_dict, dag_dict, n_op_types, op_feats_data = expose_data(
    header=data_header,
    tabular_file=f"{'query_level' if args.granularity == 'Q' else 'stage_level'}_cache_data.pkl",
    struct_file="struct_cache.pkl",
    op_feats_file=...,
    debug=debug,
    ori=True
)
add_pe(model_name, dag_dict)
print("data loaded")
op_groups, picked_groups, picked_cols = analyze_cols(data_params, col_dict)
ckp_header = f"examples/model/spark/ckp/{pj}/{model_name}/{obj}/" \
             f"{'_'.join([data_params[f] for f in ['ch1_type', 'ch1_cbo', 'ch1_enc', 'ch2', 'ch3', 'ch4']])}"
ckp_path = os.path.join(ckp_header, ckp_sign)
mp = ModelProxy(model_name, ckp_path, minmax_dict["obj"], device, op_groups, n_op_types)
tid, qid = args.template, args.template_query

df = pd.concat(dfs)
q_sign = f"q{tid}-{qid}"
stage_emb, ch2_norm, ch3_norm = prepare_data_for_opt(
    df, q_sign, dag_dict, mp.hp_params["ped"], op_groups, mp, col_dict, minmax_dict)

spark_knobs = SparkKnobs(meta_file="resources/knob-meta/spark.json")
knobs = spark_knobs.knobs

cache_header = f"{out_header}/{tid}-{qid}"
cache_conf_name = f"po_points_{n_samples}.pkl"
cache_res_name = f"po_points_{n_samples}_res.pkl"

if os.path.exists(f"{cache_header}/{cache_conf_name}"):
    cache = PickleUtils.load(cache_header, cache_conf_name)
    knob_df_pareto = cache["knob_df"]
    knob_sign_pareto = cache["knob_sign"]
    conf_df_pareto = cache["conf_df"]
    objs_pareto = cache["objs_pred"]
else:
    knob_df, ch4_norm = get_sample_spark_knobs(knobs, n_samples, seed)
    conf_df = spark_knobs.df_knob2conf(knob_df)
    start = time.time()
    lats = mp.get_lat(ch1=stage_emb, ch2=ch2_norm, ch3=ch3_norm, ch4=ch4_norm, out_fmt="numpy")
    costs = get_cloud_cost(
        lat=lats,
        mem=conf_df["spark.executor.memory"].str[:-1].astype(int).values.reshape(-1, 1),
        cores=conf_df["spark.executor.cores"].astype(int).values.reshape(-1, 1),
        nexec=conf_df["spark.executor.instances"].astype(int).values.reshape(-1, 1)
    )
    objs = np.hstack([lats, costs])
    print(f"get {len(objs)} objs, cost {time.time() - start}s")
    # inds = ws_return(objs, n_weights, seed)
    inds_pareto = is_pareto_efficient(objs)
    objs_pareto = objs[inds_pareto]
    sorted_inds = np.argsort(objs_pareto[:, 0])
    objs_pareto = objs_pareto[sorted_inds]
    conf_df_pareto = conf_df.iloc[inds_pareto]
    PickleUtils.save({
        "knob_df": knob_df.iloc[inds_pareto],
        "knob_sign": knob_df.iloc[inds_pareto].index.to_list(),
        "conf_df": conf_df_pareto,
        "objs_pred": objs_pareto
    }, cache_header, cache_conf_name)

print(f"prepared to run {len(conf_df_pareto)} recommended PO configurations")

if run:
    objs = run_q_confs(
        bm=bm, sf=sf, spark_knobs=spark_knobs, query_header=args.query_header, out_header=args.out_header, seed=seed,
        workers=BenchmarkUtils.get_workers(args.worker),
        n_trials=3, debug=debug, tid=tid, qid=qid, conf_df=conf_df_pareto)
    PickleUtils.save({
        "e2e_objs": objs
    }, cache_header, f"{cache_res_name}.{TimeUtils.get_current_iso()}")
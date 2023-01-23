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
from utils.model.proxy import ModelProxy, ws_return
from utils.model.utils import expose_data, analyze_cols, add_pe, prepare_data_for_opt, \
    get_sample_spark_knobs
from utils.optimization.moo_utils import is_pareto_efficient

import numpy as np
import pandas as pd

args = ArgsRecoQ().parse()
print(args)

bm, sf, pj = args.benchmark.lower(), args.scale_factor, f"{args.benchmark.lower()}_{args.scale_factor}"
debug = False if args.debug == 0 else True
run = False if args.run == 0 else True
seed = args.seed
data_header = f"{args.data_header}/{pj}"
query_header = args.query_header
assert os.path.exists(data_header), f"data not exists at {data_header}"
assert args.granularity in ("Q", "QS")
assert args.model_name == "GTN"
model_name = args.model_name
obj = args.obj
ckp_sign = args.ckp_sign
n_samples = args.n_samples
gpus, device = get_gpus(args.gpu)
if_robust = False if args.if_robust == 0 else True
alpha = args.alpha

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

out_header = f"{ckp_path}/{tid}-{qid}"
assert args.moo in ("bf", "ws")
moo = args.moo
moo_suffix = moo if moo == "bf" else f"{moo}_{args.n_weights}"
cache_sign = f"po_points_{n_samples}_{moo_suffix}"
if if_robust:
    cache_sign += f"_alpha_{alpha:.1f}"
cache_conf_name = f"{cache_sign}.pkl"
cache_res_name = f"{cache_sign}_res.pkl"

if os.path.exists(f"{out_header}/{cache_conf_name}"):
    cache = PickleUtils.load(out_header, cache_conf_name)
    knob_df_pareto = cache["knob_df"]
    knob_sign_pareto = cache["knob_sign"]
    conf_df_pareto = cache["conf_df"]
    print(f"found cached {len(conf_df_pareto)} PO configurations at {out_header}/{cache_conf_name}")
else:
    knob_df, ch4_norm = get_sample_spark_knobs(knobs, n_samples, seed)
    conf_df = spark_knobs.df_knob2conf(knob_df)
    start = time.time()
    if if_robust:
        lats_mu, lats_std = mp.get_lat(ch1=stage_emb, ch2=ch2_norm, ch3=ch3_norm, ch4=ch4_norm,
                                       out_fmt="numpy", dropout=True)
        costs_mu = get_cloud_cost(
            lat=lats_mu,
            mem=conf_df["spark.executor.memory"].str[:-1].astype(int).values.reshape(-1, 1),
            cores=conf_df["spark.executor.cores"].astype(int).values.reshape(-1, 1),
            nexec=conf_df["spark.executor.instances"].astype(int).values.reshape(-1, 1)
        )
        costs_std = get_cloud_cost(
            lat=lats_std,
            mem=conf_df["spark.executor.memory"].str[:-1].astype(int).values.reshape(-1, 1),
            cores=conf_df["spark.executor.cores"].astype(int).values.reshape(-1, 1),
            nexec=conf_df["spark.executor.instances"].astype(int).values.reshape(-1, 1)
        )
    else:
        lats_mu = mp.get_lat(ch1=stage_emb, ch2=ch2_norm, ch3=ch3_norm, ch4=ch4_norm, out_fmt="numpy")
        lats_std = np.zeros_like(lats_mu)
        costs_mu = get_cloud_cost(
            lat=lats_mu,
            mem=conf_df["spark.executor.memory"].str[:-1].astype(int).values.reshape(-1, 1),
            cores=conf_df["spark.executor.cores"].astype(int).values.reshape(-1, 1),
            nexec=conf_df["spark.executor.instances"].astype(int).values.reshape(-1, 1)
        )
        costs_std = np.zeros_like(costs_mu)

    objs_mu = np.hstack([lats_mu, costs_mu])
    objs_std = np.hstack([lats_std, costs_std])
    objs = objs_mu + alpha * objs_std
    print(f"get {len(objs)} objs, cost {time.time() - start}s")
    if args.moo == "bf":
        inds_pareto = is_pareto_efficient(objs)
    elif args.moo == "ws":
        inds_pareto = ws_return(objs, args.n_weights, seed)
    else:
        raise ValueError(args.moo)
    sorted_inds = np.argsort(objs[inds_pareto, 0])
    inds_pareto = np.array(inds_pareto)[sorted_inds]

    objs_pareto = objs[inds_pareto]
    conf_df_pareto = conf_df.iloc[inds_pareto]
    PickleUtils.save({
        "knob_df": knob_df.iloc[inds_pareto],
        "knob_sign": knob_df.iloc[inds_pareto].index.to_list(),
        "conf_df": conf_df_pareto,
        "objs_pred": {
            "objs_pareto": objs_pareto,
            "objs_pareto_mu": objs_mu[inds_pareto],
            "objs_pareto_std": objs_std[inds_pareto],
            "alpha": alpha
        }
    }, out_header, cache_conf_name)
    print(f"generated {len(conf_df_pareto)} PO configurations, cached at {out_header}/{cache_conf_name}")

if run:
    print(f"prepared to run {len(conf_df_pareto)} recommended PO configurations")
    if_aqe = False if args.if_aqe == 0 else True
    aqe_sign = "aqe_on" if if_aqe else "aqe_off"
    script_header = f"examples/trace/spark/internal/2.knob_hp_tuning/{bm.lower()}_{aqe_sign}/{tid}-{qid}"
    objs = run_q_confs(
        bm=bm, sf=sf, spark_knobs=spark_knobs, query_header=query_header,
        out_header=script_header, seed=seed, workers=BenchmarkUtils.get_workers(args.worker),
        n_trials=3, debug=debug, tid=tid, qid=qid, conf_df=conf_df_pareto, if_aqe=if_aqe)
    PickleUtils.save({
        "e2e_objs": objs
    }, out_header, f"{cache_res_name}_{'aqe_on' if if_aqe else 'aqe_off'}.{TimeUtils.get_current_iso()}")
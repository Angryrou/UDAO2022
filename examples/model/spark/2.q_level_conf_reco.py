# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: used weighted sum to recommend a Pareto set.
#
# Created at 08/01/2023

import os
import time

from trace.parser.spark import get_cloud_cost
from utils.common import BenchmarkUtils, PickleUtils
from utils.data.configurations import SparkKnobs
from utils.model.args import ArgsRecoQ
from utils.model.parameters import set_data_params, get_gpus
from utils.model.proxy import ModelProxy, ws_return, bf_return
from utils.model.utils import expose_data, analyze_cols, add_pe, prepare_data_for_opt, \
    get_sample_spark_knobs

import numpy as np
import pandas as pd

args = ArgsRecoQ().parse()
print(args)

bm, sf, pj = args.benchmark.lower(), args.scale_factor, f"{args.benchmark.lower()}_{args.scale_factor}"
debug = False if args.debug == 0 else True
seed = args.seed
data_header = f"{args.data_header}/{pj}"
query_header = args.query_header
assert os.path.exists(data_header), f"data not exists at {data_header}"
assert args.model_name == "GTN"
model_name = args.model_name
obj = args.obj
ckp_sign = args.ckp_sign
n_samples = args.n_samples
gpus, device = get_gpus(args.gpu)
q_signs = BenchmarkUtils.get_sampled_q_signs(bm) if args.q_signs is None else \
    [BenchmarkUtils.extract_sampled_q_sign(bm, sign) for sign in args.q_signs.split(",")]

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
df = pd.concat(dfs)

spark_knobs = SparkKnobs(meta_file="resources/knob-meta/spark.json")
knobs = spark_knobs.knobs

def get_cache_conf(inds_pareto, objs, conf_df, knob_df, objs_mu, objs_std, alpha):
    sorted_inds = np.argsort(objs[inds_pareto, 0])
    inds_pareto = np.array(inds_pareto)[sorted_inds]

    objs_pareto = objs[inds_pareto]
    conf_df_pareto = conf_df.iloc[inds_pareto]
    return {
        "knob_df": knob_df.iloc[inds_pareto],
        "knob_sign": knob_df.iloc[inds_pareto].index.to_list(),
        "conf_df": conf_df_pareto,
        "objs_pred": {
            "objs_pareto": objs_pareto,
            "objs_pareto_mu": objs_mu[inds_pareto],
            "objs_pareto_std": objs_std[inds_pareto],
            "alpha": alpha
        }
    }


for q_sign in q_signs:
    start = time.time()
    out_header = f"{ckp_path}/{q_sign}"
    cache_prefix = f"rs({n_samples}x100)"
    objs_cache_name = f"{cache_prefix}_objs.pkl"

    try:
        objs_dict = PickleUtils.load(out_header, objs_cache_name)
        print(f"found {objs_cache_name}")
        lats_mu, lats_std = objs_dict["lats"]["mu"], objs_dict["lats"]["std"]
        costs_mu, costs_std = objs_dict["costs"]["mu"], objs_dict["costs"]["std"]
        conf_df, knob_df = objs_dict["conf_df"], objs_dict["knob_df"]
        dt_pred = objs_dict["duration"]
    except:
        start = time.time()
        stage_emb, ch2_norm, ch3_norm = prepare_data_for_opt(
            df, q_sign, dag_dict, mp.hp_params["ped"], op_groups, mp, col_dict, minmax_dict)

        # get predicted data
        knob_df, ch4_norm = get_sample_spark_knobs(knobs, n_samples, seed=BenchmarkUtils.get_tid(q_sign))
        conf_df = spark_knobs.df_knob2conf(knob_df)
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
        dt_pred = time.time() - start
        PickleUtils.save({
            "lats": {"mu": lats_mu, "std": lats_std},
            "costs": {"mu": costs_mu, "std": costs_std},
            "conf_df": conf_df,
            "knob_df": knob_df,
            "duration": dt_pred,
            "_NOTE": "dropout on with 100 local samples for each input configuration in the NN"
        }, out_header, objs_cache_name)

    objs_mu = np.hstack([lats_mu, costs_mu])
    objs_std = np.hstack([lats_std, costs_std])
    options = [("vc", "ws", 0), ("vc", "bf", 0)] + \
              [("robust", algo_, alpha_) for alpha_ in [-3, -2, 0, 2, 3] for algo_ in ["ws", "bf"]]

    total_cache = {}
    for algo, moo, alpha in options:
        assert moo in ["ws", "bf"]
        moo_sign = moo if moo == "bf" else f"{moo}({args.n_weights})"
        start = time.time()
        if algo == "vc":
            objs = np.hstack([objs_mu, (objs_std / objs_mu).sum(1, keepdims=True)])
        else:
            objs = objs_mu + alpha * objs_std

        inds_pareto = bf_return(objs) if moo == "bf" else ws_return(objs, args.n_weights, seed)
        sorted_inds = np.argsort(objs[inds_pareto, 0])
        inds_pareto = np.array(inds_pareto)[sorted_inds]

        objs_pareto = objs[inds_pareto]
        conf_df_pareto = conf_df.iloc[inds_pareto]
        cache_conf_name = f"{cache_prefix}_po_{algo}_{moo_sign}_alpha({alpha:.0f}).pkl"
        dt = time.time() - start
        conf_cache = {
            "knob_df": knob_df.iloc[inds_pareto],
            "knob_sign": knob_df.iloc[inds_pareto].index.to_list(),
            "conf_df": conf_df_pareto,
            "objs_pred": {
                "objs_pareto": objs_pareto,
                "objs_pareto_mu": objs_mu[inds_pareto],
                "objs_pareto_std": objs_std[inds_pareto],
                "alpha": alpha
            },
            "duration_pred": dt_pred,
            "duration_reco": dt
        }
        total_cache[(algo, moo, alpha)] = conf_cache
        PickleUtils.save(conf_cache, out_header, cache_conf_name)
        print(f"({algo}, {moo}, {alpha}): generated {len(conf_df_pareto)} PO configurations, "
              f"cached at {out_header}/{cache_conf_name}, cost {dt:.0f}s")

    stats = [[k[0], k[1], k[2], len(v["conf_df"])] for k, v in total_cache.items()]
    df_save = pd.DataFrame(data=stats, columns=["algo", "moo", "alpha", "n_reco_conf"])
    df_save.to_csv(f"{out_header}/{cache_prefix}_bf_ws({args.n_weights}).csv", index=False)
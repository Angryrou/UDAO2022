# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: A request from YL
#
# have a few plots of the Pareto front between latency and cost for the same query.
# Plot 1: take a default latency model and generate the Pareto front
# Plot 2: for the same query, simulate changes of the data characteristics, and regenerate the Pareto front
# Plot 3: for the same query, simulate changes of the machine state, and regenerate the Pareto front
#
# need plots to get the intuition of how much the Pareto front can change
# when data characteristics and system state change.
#
# Created at 05/05/2023
import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from trace.parser.spark import get_cloud_cost
from utils.common import BenchmarkUtils, PickleUtils
from utils.data.configurations import SparkKnobs
from utils.data.feature import L2P_MAP
from utils.model.args import ArgsRecoQ
from utils.model.parameters import get_gpus, set_data_params
from utils.model.proxy import ModelProxy
from utils.model.utils import analyze_cols, expose_data, prepare_emb, norm_in_feat_inst, get_sample_spark_knobs
from utils.optimization.moo_utils import is_pareto_efficient

args = ArgsRecoQ().parse()
print(args)
bm, sf, pj = "tpch", 100, "tpch_100"
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
op_feats_file = {}
if data_params["ch1_cbo"] == "on":
    op_feats_file["cbo"] = "cbo_cache.pkl"
if data_params["ch1_enc"] != "off":
    ch1_enc = data_params["ch1_enc"]
    op_feats_file["enc"] = f"enc_cache_{ch1_enc}.pkl"
dfs, ds_dict, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, op_feats, clf_feat = expose_data(
    header=data_header,
    tabular_file=f"{'query_level' if args.granularity == 'Q' else 'stage_level'}_cache_data.pkl",
    struct_file="struct_cache.pkl",
    op_feats_file=op_feats_file,
    debug=True,
    ori=True,
    model_name=model_name,
    clf_feat_file=data_params["clf_feat"]
)
if data_params["ch1_cbo"] == "on":
    op_feats["cbo"]["l2p"] = L2P_MAP[args.benchmark.lower()]

print("data loaded")
op_groups, picked_groups, picked_cols = analyze_cols(data_params, col_dict)
ckp_header = f"examples/model/spark/ckp/{pj}/{model_name}/{obj}/" \
             f"{'_'.join([data_params[f] for f in ['ch1_type', 'ch1_cbo', 'ch1_enc', 'ch2', 'ch3', 'ch4']])}"
ckp_path = os.path.join(ckp_header, ckp_sign)
mp = ModelProxy(model_name, ckp_path, minmax_dict["obj"], device, op_groups, n_op_types, clf_feat)
df = pd.concat(dfs)
cpu_l, cpu_h = df["m1"].min(), df["m1"].max()
mem_l, mem_h = df["m2"].min(), df["m2"].max()


spark_knobs = SparkKnobs(meta_file="resources/knob-meta/spark.json")
knobs = spark_knobs.knobs


def norm_in_feat_inst_local(x, minmax):
    return (x - minmax["min"].values.reshape(1, -1)) / (minmax["max"] - minmax["min"]).values.reshape(1, -1)

def plot_pred(fig_header, q_sign, objs_pred, labels, colors, linestyles, markers, note="ch2", if_show=True):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    X, Y = [o[:, 0] for o in objs_pred], [o[:, 1] for o in objs_pred]
    for x, y, c, ls, m, label in zip(X, Y, colors, linestyles, markers, labels):
        ax.plot(x, y, c=c, ls=ls, marker=m, label=label)
    ax.legend(handletextpad=0.3, borderaxespad=0.2, fontsize=8.5)
    ax.grid()
    ax.set_title("PO Frontiers in the predicted space")
    ax.set_xlabel("latency (s)")
    ax.set_ylabel("cost($)")
    plt.tight_layout()
    figpath = f"{fig_header}/{q_sign}_{note}.pdf"
    os.makedirs(fig_header, exist_ok=True)
    fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
    if if_show:
        plt.show()
    plt.close()

# without variance concern
dts_all = {}
print("qsign\tsample_5k\tprep_emb\tprep_ch2ch3\tmodel\tpo_filter")
dt1_s, dt2_s, dt3_s, dt4_s, dt5_s = [], [], [], [], []
for q_sign in q_signs:
    dts = {}
    out_header = f"{ckp_path}/with_different_inputs/{q_sign}"
    cache_prefix = f"rs({n_samples})"
    cache_conf_name = f"{cache_prefix}_po_bf_(ch2_ch3)x8.pkl"
    try:
        cache = PickleUtils.load(out_header, cache_conf_name)
        # print(f"found {cache_conf_name}")
        po_objs_dict = cache["po_obj_dict"]
        dts = cache["dts"]
    except:

        t1 = time.time()
        knob_df, ch4_norm = get_sample_spark_knobs(knobs, n_samples, bm, q_sign, seed=BenchmarkUtils.get_tid(q_sign))
        conf_df = spark_knobs.df_knob2conf(knob_df)
        dts["sample_conf"] = time.time() - t1

        t1 = time.time()
        emb, record = prepare_emb(df, q_sign, dag_dict, mp.hp_params["ped"], op_groups, op_feats, struct2template, mp)
        dts["prep_emb"] = time.time() - t1

        t1 = time.time()
        ch2_default = np.repeat(record[col_dict["ch2"]].values, 5, 0)
        ch2_default[:, 0] = ch2_default[:, 0] * np.array([1, 4, 2, 0.5, 0.25])
        ch2_default[:, 1] = ch2_default[:, 1] * np.array([1, 4, 2, 0.5, 0.25])
        ch2_default[:, [2, 3]] = np.log(ch2_default[:, [0, 1]] + 1)
        ch2_norms = norm_in_feat_inst_local(ch2_default, minmax_dict["ch2"])
        ch3_default = np.repeat(record[col_dict["ch3"]].values, 5, 0)
        ch3_default[1:, 0] = [cpu_l, cpu_l, cpu_h, cpu_h]
        ch3_default[1:, 1] = [mem_l, mem_h, mem_l, mem_h]
        ch3_norms = norm_in_feat_inst_local(ch3_default, minmax_dict["ch3"])
        dts["prep_ch2ch3"] = time.time() - t1
        dts["model_objs"] = {}
        dts["po_filter"] = {}
        po_objs_dict = {}

        for cat, ch2_norm, ch3_norm in zip(
            ["default", "cpu-mem-", "cpu-mem+", "cpu+mem-", "cpu+mem+", "4x_input", "2x_input", "1/2_input", "1/4_input"],
            ch2_norms[[0, 0, 0, 0, 0, 1, 2, 3, 4]],
            ch3_norms[[0, 1, 2, 3, 4, 0, 0, 0, 0]]
        ):
            t1 = time.time()
            ch2_norm = ch2_norm.reshape(1, -1)
            ch3_norm = ch3_norm.reshape(1, -1)
            lats = mp.get_lat(ch1=emb, ch2=ch2_norm, ch3=ch3_norm, ch4=ch4_norm, out_fmt="numpy", dropout=False)
            costs = get_cloud_cost(
                lat=lats,
                mem=conf_df["spark.executor.memory"].str[:-1].astype(int).values.reshape(-1, 1),
                cores=conf_df["spark.executor.cores"].astype(int).values.reshape(-1, 1),
                nexec=conf_df["spark.executor.instances"].astype(int).values.reshape(-1, 1)
            )
            objs = np.hstack([lats, costs])
            dts["model_objs"][cat] = time.time() - t1

            t1 = time.time()
            mask = is_pareto_efficient(objs)
            inds_pareto = np.arange(len(objs))[mask]
            sorted_inds = np.argsort(objs[inds_pareto, 0])
            inds_pareto = np.array(inds_pareto)[sorted_inds]
            objs_pareto = objs[inds_pareto]
            dts["po_filter"][cat] = time.time() - t1
            po_objs_dict[cat] = objs_pareto

        cache = {
            "po_obj_dict": po_objs_dict,
            "dts": dts,
            "NOTE_": "using accurate model over 8 different (ch2, ch3) variance"
        }
        PickleUtils.save(cache, out_header, cache_conf_name)

    dts_all[q_sign] = po_objs_dict
    dt1, dt2, dt3, dt4, dt5 = dts['sample_conf'], dts['prep_emb'], dts['prep_ch2ch3'], \
        np.mean(list(dts['model_objs'].values())), np.mean(list(dts['po_filter'].values()))
    print(f"{q_sign}\t{dt1}\t{dt2}\t{dt3}\t{dt4}\t{dt5}")
    dt1_s.append(dt1)
    dt2_s.append(dt2)
    dt3_s.append(dt3)
    dt4_s.append(dt4)
    dt5_s.append(dt5)

    labels_ch2 = ["default", "4x_input", "2x_input", "1/2_input", "1/4_input"]
    colors_ch2 = ["black"] + sns.color_palette("pastel", 4)
    plot_pred(
        fig_header=f"{ckp_path}/with_different_inputs/figs",
        q_sign=q_sign,
        objs_pred=[po_objs_dict[l] for l in labels_ch2],
        labels=labels_ch2,
        colors=colors_ch2,
        linestyles=["--"] * len(labels_ch2),
        markers=["o"] * len(labels_ch2),
        note="ch2"
    )

    labels_ch3 = ["default", "cpu-mem-", "cpu-mem+", "cpu+mem-", "cpu+mem+"]
    colors_ch3 = ["black"] + sns.color_palette("rocket", 4)
    plot_pred(
        fig_header=f"{ckp_path}/with_different_inputs/figs",
        q_sign=q_sign,
        objs_pred=[po_objs_dict[l] for l in labels_ch3],
        labels=labels_ch3,
        colors=colors_ch3,
        linestyles=["--"] * len(labels_ch3),
        markers=["o"] * len(labels_ch3),
        note="ch3"
    )

print(f"mu(std)\t"
      f"{np.mean(dt1_s):.4f}({np.std(dt1_s):.4f})\t"
      f"{np.mean(dt2_s):.4f}({np.std(dt2_s):.4f})\t"
      f"{np.mean(dt3_s):.4f}({np.std(dt3_s):.4f})\t"
      f"{np.mean(dt4_s):.4f}({np.std(dt4_s):.4f})\t"
      f"{np.mean(dt5_s):.4f}({np.std(dt5_s):.4f})\t")
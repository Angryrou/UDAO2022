# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: construct the uncertain PO according to the paper https://www.researchgate.net/publication/241815491
#
# Created at 11/05/2023
import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as patches

from trace.parser.spark import get_cloud_cost
from utils.common import BenchmarkUtils, PickleUtils
from utils.data.configurations import SparkKnobs
from utils.data.feature import L2P_MAP
from utils.model.args import ArgsRecoQ
from utils.model.parameters import get_gpus, set_data_params
from utils.model.proxy import ModelProxy
from utils.model.utils import analyze_cols, expose_data, prepare_emb, norm_in_feat_inst, get_sample_spark_knobs, \
    prepare_data_for_opt
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

def plot_uncertain_po(objs, fig_header, nbucks = 30, cmap=plt.cm.Reds, note="uncertain_po", if_show=True):
    heatmap, xedges, yedges = np.histogram2d(objs[:, 0], objs[:, 1], bins=(nbucks, nbucks))
    heatmap = heatmap / n_model_param_samples
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    plt.imshow(heatmap.T, origin='lower', cmap=plt.cm.Reds)
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_xlim([-0.5, nbucks])
    ax.set_ylim([-0.5, nbucks])
    x_ticks = np.linspace(0, len(xedges) - 1, 5).astype(int)
    y_ticks = np.linspace(0, len(yedges) - 1, 5).astype(int)
    # Define the tick labels for each dimension
    x_tick_labels = ['{:0.0f}'.format(val) for val in xedges[x_ticks]]
    y_tick_labels = ['{:0.4f}'.format(val) for val in yedges[y_ticks]]
    # Set the tick positions and labels
    ax.set_xticks(x_ticks, x_tick_labels)
    ax.set_yticks(y_ticks, y_tick_labels)

    # Define the indices of the cells to be highlighted
    heatmap_flatten = np.array([[i, j, -rc] for i, r in enumerate(heatmap.T) for j, rc in enumerate(r) if rc > 0])
    heatmap_mask = is_pareto_efficient(heatmap_flatten)
    objs_pareto = heatmap_flatten[heatmap_mask]
    highlighted_cells = objs_pareto[:, :2]
    # Highlight the cells by coloring their edges
    for cell in highlighted_cells:
        rect = patches.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, linewidth=1, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(rect)

    ax.set_title("Uncertain Pareto Points")
    ax.set_xlabel("latency (s)")
    ax.set_ylabel("cost($)")
    plt.tight_layout()
    os.makedirs(fig_header, exist_ok=True)
    figpath = f"{fig_header}/{q_sign}_{note}.pdf"
    fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
    if if_show:
        plt.show()
    plt.close()

print("qsign\tsample_5k\tprep_mci\tmodel\tpo_filter")
dt1_s, dt2_s, dt3_s, dt4_s = [], [], [], []

n_model_param_samples = args.n_model_samples
for q_sign in q_signs:
    dts = {}
    out_header = f"{ckp_path}/uncertain_po/{q_sign}"
    cache_prefix = f"rs({n_samples})"
    cache_conf_name = f"{cache_prefix}_[{n_model_param_samples}model_params]"
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
        emb, ch2_norm, ch3_norm = prepare_data_for_opt(
            df, q_sign, dag_dict, mp.hp_params["ped"], op_groups, op_feats, struct2template, mp, col_dict, minmax_dict)
        dts["prep_mci"] = time.time() - t1

        po_objs_dict = {}

        # for model_seed in range(n_model_param_samples):

        dts["model_objs"] = {}
        dts["po_filter"] = {}
        for model_seed in range(n_model_param_samples):
            t1 = time.time()
            mp.manual_dropout(model_seed)
            lats = mp.get_lat(ch1=emb, ch2=ch2_norm, ch3=ch3_norm, ch4=ch4_norm, out_fmt="numpy", dropout=False)
            costs = get_cloud_cost(
                lat=lats,
                mem=conf_df["spark.executor.memory"].str[:-1].astype(int).values.reshape(-1, 1),
                cores=conf_df["spark.executor.cores"].astype(int).values.reshape(-1, 1),
                nexec=conf_df["spark.executor.instances"].astype(int).values.reshape(-1, 1)
            )
            objs = np.hstack([lats, costs])
            dts["model_objs"][model_seed] = time.time() - t1

            t1 = time.time()
            mask = is_pareto_efficient(objs)
            inds_pareto = np.arange(len(objs))[mask]
            sorted_inds = np.argsort(objs[inds_pareto, 0])
            inds_pareto = np.array(inds_pareto)[sorted_inds]
            objs_pareto = objs[inds_pareto]
            dts["po_filter"][model_seed] = time.time() - t1
            po_objs_dict[model_seed] = objs_pareto

        cache = {
            "po_obj_dict": po_objs_dict,
            "dts": dts,
            "NOTE_": "using accurate model over 8 different (ch2, ch3) variance"
        }
        PickleUtils.save(cache, out_header, cache_conf_name)

    dt1, dt2, dt3, dt4 = dts['sample_conf'], dts['prep_mci'], \
        np.mean(list(dts['model_objs'].values())), np.mean(list(dts['po_filter'].values()))
    print(f"{q_sign}\t{dt1:.4f}\t{dt2:.4f}\t{dt3:.4f}\t{dt4:.4f}")
    dt1_s.append(dt1)
    dt2_s.append(dt2)
    dt3_s.append(dt3)
    dt4_s.append(dt4)

    objs = np.concatenate(list(po_objs_dict.values()))
    plot_uncertain_po(
        objs,
        fig_header=f"{ckp_path}/uncertain_po/figs",
        nbucks=30,
        cmap=plt.cm.Reds,
        note="uncertain_po",
        if_show=True
    )

print(f"mu(std)\t"
      f"{np.mean(dt1_s):.4f}({np.std(dt1_s):.4f})\t"
      f"{np.mean(dt2_s):.4f}({np.std(dt2_s):.4f})\t"
      f"{np.mean(dt3_s):.4f}({np.std(dt3_s):.4f})\t"
      f"{np.mean(dt4_s):.4f}({np.std(dt4_s):.4f})")
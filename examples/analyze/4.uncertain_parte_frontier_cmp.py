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
from utils.model.utils import analyze_cols, expose_data, get_sample_spark_knobs, prepare_data_for_opt
from utils.optimization.moo_utils import is_pareto_efficient

class Args(ArgsRecoQ):
    def __init__(self):
        super(Args, self).__init__()
        self.parser.add_argument("--topK", type=float, default=0.2)

args = Args().parse()
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

topK = args.topK if (args.topK is None or args.topK < 1) else int(np.ceil(args.topK))
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

def plot_uncertain_po_cmp(objs, objs_mu, fig_header, nbucks = 30, cmap=plt.cm.Reds,
                          note="uncertain_po", if_show=True, topK=0.1):
    heatmap, xedges, yedges = np.histogram2d(objs[:, 0], objs[:, 1], bins=(nbucks, nbucks))
    heatmap = heatmap / n_model_param_samples
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    plt.imshow(heatmap.T, origin='lower', cmap=cmap)
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
    if topK is None:
        n_top = len(objs_mu)
        objs_pareto = objs_pareto[objs_pareto[:, 2].argsort()[:n_top]]
    elif topK < 1:
        n_top = np.ceil(len(objs_pareto) * topK).astype(int)
        objs_pareto = objs_pareto[objs_pareto[:, 2].argsort()[:n_top]]
    elif topK > 1:
        assert topK == int(topK)
        objs_pareto = objs_pareto[objs_pareto[:, 2].argsort()[:topK]]
    highlighted_cells = objs_pareto[:, :2]
    # Highlight the cells by coloring their edges
    for cell in highlighted_cells:
        rect = patches.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, linewidth=1, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(rect)


    edges_min = np.array([xedges[0], yedges[0]])
    edges_max = np.array([xedges[-1], yedges[-1]])
    objs_mu_norm = (objs_mu - edges_min) / (edges_max - edges_min) * nbucks
    line, = plt.plot(objs_mu_norm[:, 0], objs_mu_norm[:, 1], "gx--", label="PO")

    # Create custom legend handles
    highlight_patch = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='blue', facecolor='none',
                                         label='Uncertain PO')
    plt.legend(handles=[line, highlight_patch])

    ax.set_title("Uncertain Pareto Points")
    ax.set_xlabel("latency (s)")
    ax.set_ylabel("cost($)")
    plt.tight_layout()
    if topK is None:
        fig_header += f"_adapt"
    elif topK < 1:
        fig_header += f"_{topK:.1f}"
    elif topK > 1:
        fig_header += f"_{topK}"
    os.makedirs(fig_header, exist_ok=True)
    figpath = f"{fig_header}/{q_sign}_{note}.pdf"
    fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
    if if_show:
        plt.show()
    plt.close()

print("qsign\tsample_5k\tprep_mci\tmodel\tpo_filter")
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
        objs_dict = cache["obj_dict"]
    except:
        raise Exception("please run 3.uncertain_parte_frontier.py first")

    objs_mu = np.array([l for l in objs_dict.values()]).mean(0)
    # objs_std = np.array([l for l in objs_dict.values()]).mean(1)
    mask = is_pareto_efficient(objs_mu)
    inds_pareto = np.arange(len(objs_mu))[mask]
    sorted_inds = np.argsort(objs_mu[inds_pareto, 0])
    inds_pareto = np.array(inds_pareto)[sorted_inds]
    po_objs_mu = objs_mu[inds_pareto]

    po_objs = np.concatenate(list(po_objs_dict.values()))
    plot_uncertain_po_cmp(
        po_objs,
        po_objs_mu,
        fig_header=f"{ckp_path}/uncertain_po_cmp/figs",
        nbucks=30,
        cmap=plt.cm.Reds,
        note="uncertain_po",
        if_show=True,
        topK=topK
    )

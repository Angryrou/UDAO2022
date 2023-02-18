# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 18/02/2023

import os
import torch as th
import pandas as pd
import numpy as np

from utils.common import PickleUtils
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


def plot_err_dist_by_tid(fig_header, fig_name_prefix, df_te):
    sns.set_theme(style="ticks")

    # Initialize the figure with a logarithmic x axis
    fig, ax = plt.subplots(figsize=(12, 3))
    # ax.set_yscale("log")

    # Plot the orbital period with horizontal boxes
    sns.boxplot(x="tid", y="err", data=df_te, orient="v", width=.8, palette="vlag")

    # # Add in points to show each observation
    # sns.stripplot(x="tid", y="err", data=df_te, orient="v",
    #               size=2, color=".3", linewidth=0)

    ax.yaxis.grid(True)
    ax.set(xlabel="")
    # sns.despine(trim=True, left=True)

    ax.set_title(f"Absolute Error Distribution on {fig_name_prefix}")
    plt.tight_layout()
    figpath = f"{fig_header}/{fig_name_prefix}_dist_tid.pdf"
    fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
    plt.show()
    plt.close()


def plot_err_dist_by_range(fig_header, fig_name_prefix, df_te, bsize=10):
    sns.set_theme(style="ticks")

    # Initialize the figure with a logarithmic x axis
    fig, ax = plt.subplots(figsize=(12, 3))
    # ax.set_yscale("log")

    # Plot the orbital period with horizontal boxes
    df_te[f"lat_bucket_{bsize}_tick"] = df_te[f"lat_bucket_{bsize}"] * bsize
    sns.boxplot(x=f"lat_bucket_{bsize}_tick", y="err", data=df_te, orient="v", width=.8, palette="vlag")

    #     # Add in points to show each observation
    #     sns.stripplot(x=f"lat_bucket_{bsize}", y="err", data=df_te, orient="v",
    #                   size=2, color=".3", linewidth=0)

    ax.yaxis.grid(True)
    ax.set(xlabel="")
    if bsize <= 10:
        plt.xticks(rotation=45)

    # sns.despine(trim=True, left=True)

    ax.set_title(f"Absolute Error Distribution on latency bucket = {bsize}")
    plt.tight_layout()
    figpath = f"{fig_header}/{fig_name_prefix}_dist_latency(bucket={bsize}).pdf"
    fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
    plt.show()
    plt.close()

def plot_mae_over_hps(fig_header, fig_name_prefix, df_list, signs, xlabel, title, fsign):
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    df = pd.concat(df_list, axis=1)
    df.columns = signs
    df = df.sort_values(signs[0], ascending=False)

    n = len(signs)
    colors = sns.color_palette("mako", n)
    labels = signs
    err_list = [df[sign].values for sign in signs]
    xnames = df.index.to_list()

    barWidth = 0.4
    r_list = [
        np.arange(len(xnames)) - barWidth * ((n - 1) / 2) + barWidth * i
        for i in range(n)
    ]

    fig, ax = plt.subplots(figsize=(12, 3))
    rect_list = [None] * n
    for i in range(n):
        rect_list[i] = ax.bar(r_list[i], err_list[i], color=colors[i], width=barWidth,
                              edgecolor='white', label=labels[i])
        bar_labels = [f"{e:.1f}" for e in err_list[i]]
        ax.bar_label(rect_list[i], padding=2, fontsize=6, labels=bar_labels, rotation=0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("err")

    ax.set_xticks(np.arange(len(xnames)))
    ax.set_xticklabels(xnames)
    ax.set_xlim(
        [-1 * n / 2 * barWidth - (1 - n * barWidth) / 2, len(xnames) - 1 + n / 2 * barWidth + (1 - n * barWidth) / 2])
    ax.set_ylim([0.05, df.values.max() * 1.1])

    ax.legend(loc="upper right")

    if fsign == "latency(bucket=10)":
        plt.xticks(rotation=45)

    ax.set_title(title)
    plt.tight_layout()
    figpath = f"{fig_header}/{fig_name_prefix}_mae_{fsign}.pdf"
    fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
    plt.show()
    plt.close()


bm = "tpch"
out_header = "examples/analyze/1.analyze_model_error"
script_header = "examples/trace/spark/internal/2.knob_hp_tuning"
data_header = "examples/data/spark/cache/tpch_100"
model_name = "GTN"

tabular_data = PickleUtils.load(data_header, "query_level_cache_data.pkl")
dfs = tabular_data["dfs"]

df_dict = {}
for (model_name, feat_sign, ckp_sign) in [
    ("GTN", "on_on_w2v_on_on_on", "039a48af62fd3e0a"),
    ("AVGMLP", "on_on_w2v_on_on_on", "4ab746c8c1ddda18")
]:
    ckp_header = f"examples/model/spark/ckp/tpch_100/{model_name}/latency/{feat_sign}"
    ckp_path = f"{ckp_header}/{ckp_sign}"
    fig_header = f"{out_header}/lat_dist"
    fig_name_prefix = f"{model_name}_{feat_sign}_{ckp_sign}"
    os.makedirs(fig_header, exist_ok=True)
    results_pth_sign = f"{ckp_path}/results.pth"
    results = th.load(results_pth_sign, map_location=th.device("cpu"))
    df_te = dfs[2][["latency"]].copy()
    df_te["latency_hat"] = results["y_te_hat"]
    df_te["err"] = np.abs(df_te["latency_hat"] - df_te["latency"])
    df_te = df_te.reset_index().rename(columns={"level_0": "tid", "level_1": "qid"})
    for bsize in [5, 10, 20, 50]:
        df_te[f"lat_bucket_{bsize}"] = (df_te["latency"] // bsize).astype(int)

    df_dict[(model_name, feat_sign, ckp_sign)] = df_te

    plot_err_dist_by_tid(fig_header, fig_name_prefix, df_te)
    plot_err_dist_by_range(fig_header, fig_name_prefix, df_te, 5)
    plot_err_dist_by_range(fig_header, fig_name_prefix, df_te, 10)
    plot_err_dist_by_range(fig_header, fig_name_prefix, df_te, 20)
    plot_err_dist_by_range(fig_header, fig_name_prefix, df_te, 50)

signs, dfs_tid, dfs_latbuck = [], [], []
bsize = 10
for sign, df in df_dict.items():
    signs.append(" / ".join(sign))
    dfs_tid.append(df.groupby("tid")["err"].mean())
    dfs_latbuck.append(df.groupby(f"lat_bucket_{bsize}_tick")["err"].mean())

plot_mae_over_hps(fig_header,
                  fig_name_prefix="MAE",
                  df_list=dfs_tid,
                  signs=signs,
                  xlabel="",
                  title="Mean Absolute Error (MAE) of each template",
                  fsign="tid")
plot_mae_over_hps(fig_header,
                  fig_name_prefix="MAE",
                  df_list=dfs_latbuck,
                  signs=signs,
                  xlabel="",
                  title="Mean Absolute Error (MAE) of each latency bucket (10)",
                  fsign="latency(bucket=10)")
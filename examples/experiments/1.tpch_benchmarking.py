# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: An initial comparison over tuning
# (1) res knobs only (20 samples from lhs);
# (2) sql knobs only (20 samples from lhs);
# (3) 12 knobs together by using PO solutions based on a Q-level model (WS_10000_samples_1000_weights)
#
# 2nd version - latency with new implemented dropout model and weighted sum
#
# Created at 10/01/2023
import os
import argparse
import time
import glob

import pandas as pd

from utils.common import JsonUtils, PickleUtils, FileUtils, BenchmarkUtils
from trace.parser.spark import get_cloud_cost
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from utils.data.configurations import SparkKnobs, KnobUtils
from utils.data.feature import L2P_MAP
from utils.model.proxy import ModelProxy
from utils.model.utils import expose_data, prepare_data_for_opt, add_pe
from utils.model.parameters import DEFAULT_DEVICE
from utils.optimization.moo_utils import is_pareto_efficient

DATA_COLNS = ["q_sign", "knob_sign", "lat", "cost"]
SIGN = "1.tpch_benchmarking"

def get_mp(data_header, ckp_header, ckp_sign, op_feats_file, bm, model_name):
    dfs, ds_dict, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, op_feats_data = expose_data(
        header=data_header,
        tabular_file=f"query_level_cache_data.pkl",
        struct_file="struct_cache.pkl",
        op_feats_file=op_feats_file,
        debug=False,
        ori=True,
        model_name=model_name
    )
    if "cbo" in op_feats_data:
        op_feats_data["cbo"]["l2p"] = L2P_MAP[bm.lower()]

    df = pd.concat(dfs)
    ckp_path = os.path.join(ckp_header, ckp_sign)
    op_groups = ["ch1_type"]
    mp = ModelProxy("GTN", ckp_path, minmax_dict["obj"], DEFAULT_DEVICE, op_groups, n_op_types)
    spark_knobs = SparkKnobs(meta_file="resources/knob-meta/spark.json")
    misc = df, dag_dict, mp, op_groups, col_dict, minmax_dict, spark_knobs, struct2template, op_feats_data
    return misc

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

def get_obj_df_with_knob_signs(knob_signs, q_sign, sh, api_header="http://10.0.0.1:18088/api/v1/applications"):
    obj_df_list = []
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
        obj_df_list.append(obj_df_i)
    obj_df = pd.concat(obj_df_list)
    return obj_df


def get_objs(q_sign, knob_list, misc):
    df, dag_dict, mp, op_groups, col_dict, minmax_dict, spark_knobs, struct2template, op_feats = misc
    knobs = spark_knobs.knobs
    stage_emb, ch2_norm, ch3_norm = prepare_data_for_opt(
        df, q_sign, dag_dict, mp.hp_params["ped"], op_groups, op_feats, struct2template, mp, col_dict, minmax_dict)
    knob_df = pd.DataFrame(data=knob_list, columns=spark_knobs.knob_names)
    ch4_norm = KnobUtils.knob_normalize(knob_df, knobs)
    conf_df = spark_knobs.df_knob2conf(knob_df)
    lats = mp.get_lat(ch1=stage_emb, ch2=ch2_norm, ch3=ch3_norm, ch4=ch4_norm, out_fmt="numpy")
    costs = get_cloud_cost(
        lat=lats,
        mem=conf_df["spark.executor.memory"].str[:-1].astype(int).values.reshape(-1, 1),
        cores=conf_df["spark.executor.cores"].astype(int).values.reshape(-1, 1),
        nexec=conf_df["spark.executor.instances"].astype(int).values.reshape(-1, 1)
    )
    objs = np.hstack([lats, costs])
    sorted_inds = np.argsort(objs[:, 0])  # sorted by the latency
    objs = objs[sorted_inds]
    return objs

def get_po_obj_and_obj_hat_default(sh, spark_knobs, q_sign, misc,
                                   api_header="http://10.0.0.1:18088/api/v1/applications"):
    try:
        obj_df = PickleUtils.load(sh, file_name="default_objs.pkl")
    except:
        knobs = spark_knobs.knobs
        conf_dict = {k.name: k.default for k in knobs}
        conf_df = pd.DataFrame.from_records([conf_dict])
        df = spark_knobs.df_conf2knob(conf_df)
        knob_signs = df.apply(lambda x: KnobUtils.knobs2sign(x, spark_knobs.knobs), axis=1)
        obj_df = get_obj_df_with_knob_signs(knob_signs, q_sign, sh, api_header)
        PickleUtils.save(obj_df, sh, file_name=f"default_objs.pkl")

    df = obj_df.groupby(["q_sign", "knob_sign"]).mean().loc[q_sign]
    d = df.values
    d_knobs = [KnobUtils.sign2knobs(s, spark_knobs.knobs) for s in df.index.tolist()]
    d_pred = get_objs(q_sign, d_knobs, misc)
    return d, d_pred


def get_po_obj_and_obj_hat_heuristic(sh, spark_knobs, q_sign, misc, hsign="res",
                                     api_header="http://10.0.0.1:18088/api/v1/applications"):
    try:
        obj_df = PickleUtils.load(sh, file_name=f"lhs_{hsign}_objs.pkl")
    except:
        conf_df = pd.read_csv(f"{sh}/lhs_{hsign}.csv")
        conf_df["spark.shuffle.compress"] = conf_df["spark.shuffle.compress"].map(lambda x: "true" if x else "false")
        df = spark_knobs.df_conf2knob(conf_df)
        knob_signs = df.apply(lambda x: KnobUtils.knobs2sign(x, spark_knobs.knobs), axis=1)
        obj_df = get_obj_df_with_knob_signs(knob_signs, q_sign, sh, api_header)
        PickleUtils.save(obj_df, sh, file_name=f"lhs_{hsign}_objs.pkl")

    obj_mu = obj_df.groupby(["q_sign", "knob_sign"]).mean().loc[[q_sign]]
    po_mask = is_pareto_efficient(obj_mu.loc[q_sign].values)
    po_df = obj_mu.loc[q_sign][po_mask].sort_values("lat")
    po = po_df.values
    po_knob_signs = po_df.index.tolist()
    po_knobs = [KnobUtils.sign2knobs(s, spark_knobs.knobs) for s in po_knob_signs]
    po_pred = get_objs(q_sign, po_knobs, misc)
    return po, po_pred

def get_tuned(sh, pred_header, pred_name, q_sign):
    tuned_meta = PickleUtils.load(f"{pred_header}/{q_sign}", f"{pred_name}.pkl")
    knob_signs = tuned_meta["knob_sign"]
    tuned_pred = tuned_meta["objs_pred"]["objs_pareto"]
    tuned = get_obj_df_with_knob_signs(knob_signs, q_sign, sh)[["q_sign", "knob_sign", "lat", "cost"]] \
        .groupby(["q_sign", "knob_sign"]).mean().loc[q_sign].loc[knob_signs].values
    return tuned, tuned_pred

def get_objs_all(q_sign, meta, aqe_sign):
    mp_misc, spark_knobs, bm, script_header, out_header, pred_header, pred_name_prefix = meta
    sh = os.path.join(script_header, f"{bm}_{aqe_sign}", q_sign)
    try:
        obj = PickleUtils.load(sh, f"{SIGN}.pkl")
    except:
        default, default_pred = get_po_obj_and_obj_hat_default(
            sh=sh, spark_knobs=spark_knobs, q_sign=q_sign, misc=mp_misc)
        res_po, res_po_pred = get_po_obj_and_obj_hat_heuristic(
            sh=sh, spark_knobs=spark_knobs, q_sign=q_sign, misc=mp_misc, hsign="res")
        sql_po, sql_po_pred = get_po_obj_and_obj_hat_heuristic(
            sh=sh, spark_knobs=spark_knobs, q_sign=q_sign, misc=mp_misc, hsign="sql")

        vc_po, vc_po_pred = get_tuned(sh, pred_header, f"{pred_name_prefix}_vc_ws(5000)_alpha(0)", q_sign)

        tuned_dict, tuned_pred = {}, None
        for a in [-3, -2, 0, 2, 3]:
            pred_name = f"{pred_name_prefix}_robust_ws(5000)_alpha({a})"
            if a == 0:
                tuned0, tuned_pred = get_tuned(sh, pred_header, pred_name, q_sign)
                tuned_dict[0] = tuned0
            else:
                tuned, _ = get_tuned(sh, pred_header, f"{pred_name_prefix}_robust_ws(5000)_alpha({a})", q_sign)
                tuned_dict[a] = tuned

        obj = {
            "default": default,
            "default_pred": default_pred,
            "res": res_po,
            "res_pred": res_po_pred,
            "sql": sql_po,
            "sql_pred": sql_po_pred,
            "vc": vc_po,
            "vc_pred": vc_po_pred,
            "tuned": tuned_dict,
            "tuned_pred": tuned_pred
        }
        PickleUtils.save(obj, sh, f"{SIGN}.pkl")
    return obj

def get_wmape(y, y_hat):
    y_err = np.abs(y - y_hat)
    return y_err.sum() / y.sum()


def plot_pred(fig_header, q_sign, objs, objs_pred, if_show=True):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    X, Y = [o[:, 0] for o in objs_pred], [o[:, 1] for o in objs_pred]
    wmapes = [get_wmape(o[:, 0], o_pred[:, 0]) for o, o_pred in zip(objs, objs_pred)]
    labels = [f"{p}({w:.2f})" for p, w in zip(["default", "res_po", "sql_po", "ws"], wmapes)]
    fmts = ["ko", "go--", "bo--", "ro--"]
    for x, y, fmt, label in zip(X, Y, fmts, labels):
        ax.plot(x, y, fmt, label=label)
    ax.legend(handletextpad=0.3, borderaxespad=0.2, fontsize=8.5)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    ax.set_title("PO Frontiers in the predicted space")
    ax.set_xlabel("latency (s)")
    ax.set_ylabel("cost($)")
    plt.tight_layout()
    figpath = f"{fig_header}/{q_sign}_po_pred.pdf"
    fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
    if if_show:
        plt.show()
    plt.close()

def get_po_points(objs):
    mask = is_pareto_efficient(objs)
    po_objs = objs[mask]
    sorted_inds = np.argsort(po_objs[:, 0])  # sorted by the latency
    po_objs = po_objs[sorted_inds]
    return po_objs

def get_mimmax(obj_off, obj_on):
    tuned_off = {k: get_po_points(v) for k, v in obj_off["tuned"].items()}
    tuned_on = {k: get_po_points(v) for k, v in obj_on["tuned"].items()}
    vc_off = get_po_points(obj_off["vc"])
    vc_on = get_po_points(obj_on["vc"])
    objs_2d = [obj_off["default"], obj_off["res"], obj_off["sql"], vc_off] + \
              [obj_on["default"], obj_on["res"], obj_on["sql"], vc_on] + \
              [tuned_off[k] for k in [-3, -2, 0, 2, 3]] + \
              [tuned_on[k] for k in [-3, -2, 0, 2, 3]]
    po_all_2d = np.concatenate(objs_2d)
    return po_all_2d.min(0), po_all_2d.max(0)


def calculate_dom_space(po, anchors):
    po = po[po[:, 0].argsort()]  # sort po by latency.
    obj_u, obj_n = anchors
    space = (obj_n - obj_u).prod()
    dom_space = 0
    for i in range(len(po)):
        dom_space += (obj_n - po[i]).prod()
        obj_n = np.array([obj_n[0], po[i, 1]])
    return dom_space / space

def plot_actual(objs, aqe_sign, fig_header, q_sign, if_po=True, if_show=True):
    tuned = objs["tuned"]
    if if_po:
        tuned = {k: get_po_points(v) for k, v in tuned.items()}
        objs_vc = get_po_points(objs["vc"])
    else:
        objs_vc = objs["vc"]

    objs_2d = [objs["default"], objs["res"], objs["sql"]] + [tuned[k] for k in [-3, -2, 0, 2, 3]] + [objs_vc]
    X, Y = [o[:, 0] for o in objs_2d], [o[:, 1] for o in objs_2d]
    labels = ["default", "res_po", "sql_po", "ws(-3)", "ws(-2)", "ws", "ws(+2)", "ws(+3)", "vc"]
    if if_po:
        # calculate dominated space for each algo.
        po_all_2d = np.concatenate(objs_2d)
        anchors = po_all_2d.min(0), po_all_2d.max(0)
        dom_spaces = [calculate_dom_space(po, anchors) for po in objs_2d]
        labels = [f"{l}, {ds * 100:.0f}%" for l, ds in zip(labels, dom_spaces)]

    colors = ["black", "green", "blue"] + sns.color_palette("rocket", 5) + ["purple"]
    linestyles = ["--"] * len(X)
    markers = ["o", "o", "o", "v", "^", "X", "<", ">", "+"]

    fig_name = f"{q_sign}_po({aqe_sign})_local_anchors" if if_po else f"{q_sign}_po({aqe_sign})_raw"
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for x, y, c, ls, m, label in zip(X, Y, colors, linestyles, markers, labels):
        ax.plot(x, y, c=c, ls=ls, marker=m, label=label)

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(ncol=2, handletextpad=0.3, borderaxespad=0.2, fontsize=8.5)
    ax.set_xlabel("latency (s)")
    ax.set_ylabel("cost($)")
    ax.set_title(f"PO Frontiers ({aqe_sign.upper()})")
    ax.grid()
    plt.tight_layout()
    figpath = f"{fig_header}/{fig_name}.pdf"
    fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
    if if_show:
        plt.show()
    plt.close()
    if if_po:
        return dom_spaces

def plot_actual_glb_anchors(objs, aqe_sign, fig_header, q_sign, if_po=True, anchors=None, if_show=True):
    tuned = objs["tuned"]
    assert if_po

    tuned = {k: get_po_points(v) for k, v in tuned.items()}
    objs_vc = get_po_points(objs["vc"])
    assert anchors is not None

    objs_2d = [objs["default"], objs["res"], objs["sql"]] + [tuned[k] for k in [-3, -2, 0, 2, 3]] + [objs_vc]
    X, Y = [o[:, 0] for o in objs_2d], [o[:, 1] for o in objs_2d]
    labels = ["default", "res_po", "sql_po", "ws(-3)", "ws(-2)", "ws", "ws(+2)", "ws(+3)", "vc"]
    # calculate dominated space for each algo.
    dom_spaces = [calculate_dom_space(po, anchors) for po in objs_2d]
    labels = [f"{l}, {ds * 100:.0f}%" for l, ds in zip(labels, dom_spaces)]


    colors = ["black", "green", "blue"] + sns.color_palette("rocket", 5) + ["magenta"]
    linestyles = ["--"] * len(X)
    markers = ["o", "o", "o", "v", "^", "X", "<", ">", "+"]

    fig_name = f"{q_sign }_po({aqe_sign})_glb_anchors"
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for x, y, c, ls, m, label in zip(X, Y, colors, linestyles, markers, labels):
        ax.plot(x, y, c=c, ls=ls, marker=m, label=label)

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(ncol=2, handletextpad=0.3, borderaxespad=0.2, fontsize=8.5)
    ax.set_xlabel("latency (s)")
    ax.set_ylabel("cost($)")
    ax.set_title(f"PO Frontiers ({aqe_sign.upper()})")
    ax.grid()
    plt.tight_layout()
    figpath = f"{fig_header}/{fig_name}.pdf"
    fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
    if if_show:
        plt.show()
    plt.close()
    return dom_spaces


def plot_q_all(q_sign, meta, fig_header, if_show=True):
    obj_off = get_objs_all(q_sign, meta, "aqe_off")
    obj_on = get_objs_all(q_sign, meta, "aqe_on")

    # 1. predicted space
    plot_pred(
        fig_header, q_sign,
        objs=[obj_off["default"], obj_off["res"], obj_off["sql"], obj_off["tuned"][0]],
        objs_pred=[obj_off["default_pred"], obj_off["res_pred"], obj_off["sql_pred"], obj_off["tuned_pred"]],
        if_show=True)

    dom_space_dict = {}
    anchors = get_mimmax(obj_off, obj_on)
    # 2. AQE_OFF obj_space
    plot_actual(obj_off, "aqe_off", fig_header, q_sign, if_po=False, if_show=if_show)
    dom_space_dict["off_local"] = plot_actual(obj_off, "aqe_off", fig_header, q_sign, if_po=True, if_show=if_show)
    dom_space_dict["off_glb"] = plot_actual_glb_anchors(
        obj_off, "aqe_off", fig_header, q_sign, if_po=True, anchors=anchors, if_show=if_show)

    # 3. AQE_ON obj_space
    plot_actual(obj_on, "aqe_on", fig_header, q_sign, if_po=False, if_show=if_show)
    dom_space_dict["on_local"] = plot_actual(obj_on, "aqe_on", fig_header, q_sign, if_po=True, if_show=if_show)
    dom_space_dict["on_glb"] = plot_actual_glb_anchors(
        obj_on, "aqe_on", fig_header, q_sign, if_po=True, anchors=anchors, if_show=if_show)

    return dom_space_dict

class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--q-signs", type=str, default=None)

    def parse(self):
        return self.parser.parse_args()


def main():
    bm = "tpch"
    out_header = "examples/experiments/1.tpch_benchmarking"
    script_header = "examples/trace/spark/internal/2.knob_hp_tuning"
    ckp_header = "examples/model/spark/ckp/tpch_100/GTN/latency/on_off_off_on_on_on"
    ckp_sign = "b7698e80492e5d72"
    pred_header = os.path.join(ckp_header, ckp_sign)
    pred_name_prefix = "rs(10000x100)_po"
    fig_header = os.path.join(ckp_header, ckp_sign, "fig")
    os.makedirs(fig_header, exist_ok=True)
    data_header = "examples/data/spark/cache/tpch_100"
    model_name = "GTN"
    op_feats_file = {}
    mp_misc = get_mp(data_header, ckp_header, ckp_sign, op_feats_file, bm, model_name)

    spark_knobs = SparkKnobs()
    meta = mp_misc, spark_knobs, bm, script_header, out_header, pred_header, pred_name_prefix

    args = Args().parse()
    q_signs = BenchmarkUtils.get_sampled_q_signs(bm) if args.q_signs is None else \
        [BenchmarkUtils.extract_sampled_q_sign(bm, sign) for sign in args.q_signs.split(",")]
    summary = {q_sign: [] for q_sign in q_signs}

    header = ["Q", "default", "res", "sql", "ws(-3)", "ws(-2)", "ws", "ws(+2)", "ws(+3)", "vc"]

    for q_sign in q_signs:
        summary[q_sign] = plot_q_all(q_sign=q_sign, meta=meta, fig_header=fig_header, if_show=False)

    for mode in ["off_local", "off_glb", "on_local", "on_glb"]:
        print(mode)
        print("&\t".join(header) + "\\\\")
        s_mode = []
        for tid, s_ in summary.items():
            print(f"{tid}&\t" + "&\t".join([f"{i * 100:.0f}\%" for i in s_[mode]]) + "\t\\\\")
            s_mode.append(s_[mode])
        s_mode = np.array(s_mode)
        s_mode_mu, s_mode_std = s_mode.mean(0), s_mode.std(0)
        print("AVG&\t" + "&\t".join(f"{mu_ * 100:.0f}\%" for mu_ in s_mode_mu) + "\\\\")
        print("STD&\t" + "&\t".join(f"{std_ * 100:.0f}\%" for std_ in s_mode_std) + "\\\\")
        print()


if __name__ == '__main__':
    main()

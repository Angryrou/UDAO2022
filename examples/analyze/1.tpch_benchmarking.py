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

from utils.common import JsonUtils, PickleUtils, plot, FileUtils
from trace.parser.spark import get_cloud_cost
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from utils.data.configurations import SparkKnobs, KnobUtils
from utils.model.proxy import ModelProxy
from utils.model.utils import expose_data, prepare_data_for_opt, add_pe
from utils.model.parameters import DEFAULT_DEVICE
from utils.optimization.moo_utils import is_pareto_efficient

DATA_COLNS = ["q_sign", "knob_sign", "lat", "cost"]
QSIGNS = [f"q{i}-1" for i in range(1, 23)]
SIGN = "1.tpch_benchmarking"

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
        print(f"got {q_sign}-{knob_sign}")
        if if_full_plan:
            return q_sign, knob_sign, lat, cost, sql
        return q_sign, knob_sign, lat, cost
    except Exception as e:
        print(f"failed to get {url_str}/sql, {e}")
        raise Exception(e)


def get_default_objs(out_header, file_name):
    try:
        obj_dict = PickleUtils.load(out_header, file_name)
        print(f"found default objs")
    except:
        print(f"not found default objs, generating...")
    #         start = time.time()
    #         query_urls = [
    #             ("aqe_off", "http://10.0.0.1:18088/api/v1/applications/application_1667868712223", 1),
    #             ("aqe_on", "http://10.0.0.1:18088/api/v1/applications/application_1667868712223", 67),
    #         ]
    #         obj_dict = {}
    #         for aqe_, url_header, url_suffix_start in query_urls:
    #             url_suffix_end = url_suffix_start + 66 - 1
    #             ret = [sqldt_from_appid(url_header, appid) for appid in range(url_suffix_start, url_suffix_end + 1)]
    #             obj_dict[aqe_] = pd.DataFrame(data=ret, columns=DATA_COLNS)
    #         PickleUtils.save(obj_dict, out_header, file_name)
    #         print(f"finished generating default objs, cost {time.time() - start}s")
    return obj_dict


def get_mp(data_header, ckp_header, ckp_sign):
    dfs, ds_dict, col_dict, minmax_dict, dag_dict, n_op_types, op_feats_data = expose_data(
        header=data_header,
        tabular_file=f"query_level_cache_data.pkl",
        struct_file="struct_cache.pkl",
        op_feats_file=...,
        debug=False,
        ori=True
    )
    add_pe("GTN", dag_dict)
    df = pd.concat(dfs)
    ckp_path = os.path.join(ckp_header, ckp_sign)
    op_groups = ["ch1_type"]
    mp = ModelProxy("GTN", ckp_path, minmax_dict["obj"], DEFAULT_DEVICE, op_groups, n_op_types)
    spark_knobs = SparkKnobs(meta_file="resources/knob-meta/spark.json")
    misc = df, dag_dict, mp, op_groups, col_dict, minmax_dict, spark_knobs
    return misc


def get_objs(q_sign, knob_list, misc):
    df, dag_dict, mp, op_groups, col_dict, minmax_dict, spark_knobs = misc
    knobs = spark_knobs.knobs
    stage_emb, ch2_norm, ch3_norm = prepare_data_for_opt(
        df, q_sign, dag_dict, mp.hp_params["ped"], op_groups, mp, col_dict, minmax_dict)
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


def get_obj_and_obj_hat_default(default_objs, q_sign, misc):
    df, dag_dict, mp, op_groups, col_dict, minmax_dict, spark_knobs = misc
    obj_default_mu = default_objs.groupby(["q_sign", "knob_sign"]).mean().loc[QSIGNS]
    d = obj_default_mu.loc[q_sign].values
    d_signs = obj_default_mu.loc[q_sign].sort_values("lat").index.tolist()
    d_conf = [KnobUtils.sign2knobs(s, spark_knobs.knobs) for s in d_signs]
    d_pred = get_objs(q_sign, d_conf, misc)
    return d, d_pred


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


def get_po_obj_and_obj_hat_heuristic(sh, spark_knob, q_sign, misc, hsign="res",
                                     api_header="http://10.0.0.1:18088/api/v1/applications"):
    try:
        obj_df = PickleUtils.load(sh, file_name=f"lhs_{hsign}_objs.pkl")
    except:
        conf_df = pd.read_csv(f"{sh}/lhs_{hsign}.csv")
        conf_df["spark.shuffle.compress"] = conf_df["spark.shuffle.compress"].map(lambda x: "true" if x else "false")
        df = spark_knob.df_conf2knob(conf_df)
        knob_signs = df.apply(lambda x: KnobUtils.knobs2sign(x, spark_knob.knobs), axis=1)
        obj_df = get_obj_df_with_knob_signs(knob_signs, q_sign, sh, api_header)
        PickleUtils.save(obj_df, sh, file_name=f"lhs_{hsign}_objs.pkl")

    df, dag_dict, mp, op_groups, col_dict, minmax_dict, spark_knobs = misc
    obj_mu = obj_df.groupby(["q_sign", "knob_sign"]).mean().loc[[q_sign]]
    po_mask = is_pareto_efficient(obj_mu.loc[q_sign].values)
    po_df = obj_mu.loc[q_sign][po_mask].sort_values("lat")
    po = po_df.values
    po_knob_signs = po_df.index.tolist()
    po_knobs = [KnobUtils.sign2knobs(s, spark_knobs.knobs) for s in po_knob_signs]
    po_pred = get_objs(q_sign, po_knobs, misc)
    return po, po_pred


def get_tuned(sh, pred_header, tid, pred_name, q_sign):
    tuned_meta = PickleUtils.load(f"{pred_header}/{tid}-1", f"{pred_name}.pkl")
    knob_signs = tuned_meta["knob_sign"]
    tuned_pred = tuned_meta["objs_pred"]["objs_pareto"]
    tuned = get_obj_df_with_knob_signs(knob_signs, q_sign, sh)[["q_sign", "knob_sign", "lat", "cost"]] \
        .groupby(["q_sign", "knob_sign"]).mean().loc[q_sign].loc[knob_signs].values
    return tuned_pred, tuned


def get_objs_all(tid, meta, aqe_sign):
    q_sign = QSIGNS[tid - 1]
    default_obj_dict, mp_misc, spark_knob, bm, script_header, out_header, fig_header, pred_header, pred_name = meta
    sh = os.path.join(script_header, f"{bm}_{aqe_sign}", f"{tid}-1")
    try:
        obj = PickleUtils.load(sh, f"{SIGN}.pkl")
    except:
        res_po, res_po_pred = get_po_obj_and_obj_hat_heuristic(
            sh=sh, spark_knob=spark_knob, q_sign=q_sign, misc=mp_misc, hsign="res")
        sql_po, sql_po_pred = get_po_obj_and_obj_hat_heuristic(
            sh=sh, spark_knob=spark_knob, q_sign=q_sign, misc=mp_misc, hsign="sql")
        tuned_pred, tuned0 = get_tuned(sh, pred_header, tid, pred_name, q_sign)
        tuned_dict = {0: tuned0}
        for a in [-3, -2, 2, 3]:
            _, tuned = get_tuned(sh, pred_header, tid, f"{pred_name}_alpha_{a:.1f}", q_sign)
            tuned_dict[a] = tuned

        obj = {
            "res": res_po,
            "res_pred": res_po_pred,
            "sql": sql_po,
            "sql_pred": sql_po_pred,
            "tuned": tuned_dict,
            "tuned_pred": tuned_pred
        }
        PickleUtils.save(obj, sh, f"{SIGN}.pkl")
    return obj


def get_wmape(y, y_hat):
    y_err = np.abs(y - y_hat)
    return y_err.sum() / y.sum()


def plot_pred(fig_header, tid, objs, objs_pred):
    d, res_po, sql_po, tuned_a0 = objs
    d_pred, res_po_pred, sql_po_pred, tuned_pred_a0 = objs_pred

    d_wmape = get_wmape(d[:, 0], d_pred[:, 0])
    res_wmape = get_wmape(res_po[:, 0], res_po_pred[:, 0])
    sql_wmape = get_wmape(sql_po[:, 0], sql_po_pred[:, 0])
    tuned_wmape = get_wmape(tuned_a0[:, 0], tuned_pred_a0[:, 0])

    fig, ax = plt.subplots()
    plot(
        X=[d_pred[:, 0], res_po_pred[:, 0], sql_po_pred[:, 0], tuned_pred_a0[:, 0]],
        Y=[d_pred[:, 1], res_po_pred[:, 1], sql_po_pred[:, 1], tuned_pred_a0[:, 1]],
        xlabel="latency (s)", ylabel="cost($)",
        legend=[f"default({d_wmape:.3f})", f"res_po({res_wmape:.3f})", f"sql_po({sql_wmape:.3f})",
                r"ws,$\mu_i$" + f"({tuned_wmape:.3f})"],
        fmts=["ko", "go--", "bo--", "ro--"],
        axes=ax, figsize=(4.5, 3.5))
    ax.set_title("PO Frontiers in the predicted space")
    plt.tight_layout()
    figpath = f"{fig_header}/q{tid}_po_pred.pdf"
    fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
    plt.show()
    plt.close()


def get_po_points(objs):
    mask = is_pareto_efficient(objs)
    po_objs = objs[mask]
    sorted_inds = np.argsort(po_objs[:, 0])  # sorted by the latency
    po_objs = po_objs[sorted_inds]
    return po_objs

def plot_actual(d, objs, aqe_sign, fig_header, tid, if_po=True):
    tuned = objs["tuned"]
    if if_po:
        tuned = {k: get_po_points(v) for k, v in tuned.items()}
    objs_2d = [d, objs["res"], objs["sql"]] + [tuned[k] for k in [-3, -2, 0, 2, 3]]
    X, Y = [o[:, 0] for o in objs_2d], [o[:, 1] for o in objs_2d]
    labels = [f"default", f"res_po", f"sql_po", r"ws,$\mu_i - 3\sigma_i$", r"ws,$\mu_i - 2\sigma_i$",
              r"ws,$\mu_i$", r"ws,$\mu_i + 2\sigma_i$", r"ws,$\mu_i + 3\sigma_i$"]
    #     tuned_colors = sns.color_palette("rocket", 5)
    #     colors = ["black", "green", "blue"] + tuned_colors

    colors = ["black"] + sns.color_palette("bright")
    linestyles = ["--"] * len(X)
    markers = ["o", "o", "o", "v", "^", "X", "<", ">"]

    fig_name = f"q{tid}_po({aqe_sign})" if if_po else f"q{tid}_po({aqe_sign})_raw"
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for x, y, c, ls, m, label in zip(X, Y, colors, linestyles, markers, labels):
        ax.plot(x, y, c=c, ls=ls, marker=m, label=label)
    ax.legend(ncol=2)
    ax.set_xlabel("latency (s)")
    ax.set_ylabel("cost($)")
    ax.set_title(f"PO Frontiers ({aqe_sign.upper()})")
    plt.tight_layout()
    figpath = f"{fig_header}/{fig_name}.pdf"
    fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
    plt.show()
    plt.close()


def plot_q_all(tid, meta):
    default_obj_dict, mp_misc, spark_knob, bm, script_header, out_header, fig_header, pred_header, pred_name = meta
    q_sign = QSIGNS[tid - 1]
    d_off, d_pred = get_obj_and_obj_hat_default(default_objs=default_obj_dict["aqe_off"], q_sign=q_sign, misc=mp_misc)
    d_on, _ = get_obj_and_obj_hat_default(default_objs=default_obj_dict["aqe_on"], q_sign=q_sign, misc=mp_misc)

    obj_off = get_objs_all(tid, meta, "aqe_off")
    obj_on = get_objs_all(tid, meta, "aqe_on")

    # 1. predicted space
    plot_pred(fig_header, tid,
              objs=[d_off, obj_off["res"], obj_off["sql"], obj_off["tuned"][0]],
              objs_pred=[d_pred, obj_off["res_pred"], obj_off["sql_pred"], obj_off["tuned_pred"]])

    # 2. AQE_OFF obj_space
    plot_actual(d_off, obj_off, "aqe_off", fig_header, tid=tid, if_po=False)
    plot_actual(d_off, obj_off, "aqe_off", fig_header, tid=tid, if_po=True)

    # 3. AQE_ON obj_space
    plot_actual(d_on, obj_on, "aqe_on", fig_header, tid=tid, if_po=False)
    plot_actual(d_on, obj_on, "aqe_on", fig_header, tid=tid, if_po=True)


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-t", "--template-list", type=str, default="1,18,2,3,4")

    def parse(self):
        return self.parser.parse_args()

def main():
    default_obj_dict = get_default_objs(out_header="examples/analyze/1.heuristic.vs.q_level_tuning",
                                        file_name="default_objs.pkl")

    bm = "tpch"
    out_header = "examples/analyze/1.tpch_benchmarking"
    script_header = "examples/trace/spark/internal/2.knob_hp_tuning"
    ckp_header = "examples/model/spark/ckp/tpch_100/GTN/latency/on_off_off_on_on_on"
    ckp_sign = "b7698e80492e5d72"
    pred_header = os.path.join(ckp_header, ckp_sign)
    pred_name = "po_points_10000_ws_1000"
    fig_header = os.path.join(ckp_header, ckp_sign, "fig")
    data_header = "examples/data/spark/cache/tpch_100"
    mp_misc = get_mp(data_header, ckp_header, ckp_sign)

    spark_knob = SparkKnobs()
    meta = default_obj_dict, mp_misc, spark_knob, bm, script_header, out_header, fig_header, pred_header, pred_name

    args = Args().parse()
    for tid in args.template_list.split(","):
        plot_q_all(tid=int(tid), meta=meta)


if __name__ == '__main__':
    main()

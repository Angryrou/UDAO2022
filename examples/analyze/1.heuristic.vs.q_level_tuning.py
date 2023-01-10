# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: An initial comparison over tuning
# (1) res knobs only (20 samples from lhs);
# (2) sql knobs only (20 samples from lhs);
# (3) 12 knobs together by using PO solutions based on a Q-level model (WS_10000_samples_5000_weights)
#
# Created at 10/01/2023
import os
import time

import pandas as pd

from utils.common import JsonUtils, PickleUtils, plot
from trace.parser.spark import get_cloud_cost
import numpy as np
from matplotlib import pyplot as plt

from utils.data.configurations import SparkKnobs, KnobUtils
from utils.model.proxy import ModelProxy
from utils.model.utils import expose_data, prepare_data_for_opt, add_pe
from utils.model.parameters import DEFAULT_DEVICE
from utils.optimization.moo_utils import is_pareto_efficient

DATA_COLNS = ["q_sign", "knob_sign", "lat", "cost"]
QSIGNS = [f"q{i}-1" for i in range(1, 23)]

def sqldt_from_appid(url_header, appid, if_full_plan=False):
    appid_str = f"{appid:04}" if appid < 10000 else str(appid)
    url_str = f"{url_header}_{appid_str}"
    try:
        data = JsonUtils.load_json_from_url(url_str)
        sql = JsonUtils.load_json_from_url(f"{url_str}/sql/1")
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
    except:
        print(f"failed to get {url_str}/sql")
        if if_full_plan:
            return "", "", -1, -1, None
        return "", "", -1, -1


def get_default_objs(out_header, file_name):
    try:
        obj_dict = PickleUtils.load(out_header, file_name)
        print(f"found default objs")
    except:
        print(f"not found default objs, generating...")
        start = time.time()
        query_urls = [
            ("aqe_off", "http://10.0.0.1:18088/api/v1/applications/application_1667868712223", 1),
            ("aqe_on", "http://10.0.0.1:18088/api/v1/applications/application_1667868712223", 67),
        ]
        obj_dict = {}
        for aqe_, url_header, url_suffix_start in query_urls:
            url_suffix_end = url_suffix_start + 66 - 1
            ret = [sqldt_from_appid(url_header, appid) for appid in range(url_suffix_start, url_suffix_end + 1)]
            obj_dict[aqe_] = pd.DataFrame(data=ret, columns=DATA_COLNS)
        PickleUtils.save(obj_dict, out_header, file_name)
        print(f"finished generating default objs, cost {time.time() - start}s")
    return obj_dict


def get_heuristic_objs(out_header, file_name):
    try:
        obj_dict = PickleUtils.load(out_header, file_name)
        print(f"found heuristic objs")
    except:
        print(f"not found heuristic objs, generating...")
        start = time.time()
        query_urls = [
            ("q1", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 4),
            ("q2", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478),
            ("q3", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237),
            ("q4", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 2),
            ("q5", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 3),
            ("q6", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 4),
            ("q7", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 5),
            ("q8", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 6),
            ("q9", "http://10.0.0.1:18088/api/v1/applications/application_1666935336888", 20000),
            ("q10", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 7),
            ("q11", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 8),
            ("q12", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 9),
            ("q13", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 10),
            ("q14", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 11),
            ("q15", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 12),
            ("q16", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 13),
            ("q17", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 14),
            ("q18", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 241),
            ("q19", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 15),
            ("q20", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 16),
            ("q21", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 17),
            ("q22", "http://10.0.0.13:18088/api/v1/applications/application_1666993404824", 478 + 237 * 18)
        ]
        ret_res, ret_sql = [], []
        for qid, url_header, url_suffix_start in query_urls:
            start2 = time.time()
            url_suffix_end = url_suffix_start + 119
            ret_res += [sqldt_from_appid(url_header, appid) for appid in range(url_suffix_start, url_suffix_end + 1)]
            url_suffix_start2 = url_suffix_end + 1
            url_suffix_end2 = url_suffix_start2 + 116
            ret_sql += [sqldt_from_appid(url_header, appid) for appid in \
                        [url_suffix_start + 132, url_suffix_start + 133, url_suffix_start + 134] + list(
                            range(url_suffix_start2, url_suffix_end2 + 1))]
            print(f"finished {qid}, cost {time.time() - start2}s")
        obj_dict = {"res": pd.DataFrame(data=ret_res, columns=DATA_COLNS),
                    "sql": pd.DataFrame(data=ret_sql, columns=DATA_COLNS)}
        PickleUtils.save(obj_dict, out_header, file_name)
        print(f"finished generating heuristic objs, cost {time.time() - start}s")
    return obj_dict


def get_tuned_objs(out_header, file_name, if_full_plan=False):
    try:
        obj_dict = PickleUtils.load(out_header, file_name)
        print(f"found tuned objs")
    except:
        print("not found tuned objs, generating...")
        start = time.time()
        query_urls = [
            ("aqe_off", "http://10.0.0.1:18088/api/v1/applications/application_1673285915271", 1),
            ("aqe_on", "http://10.0.0.1:18088/api/v1/applications/application_1673285915271", 358),
        ]
        obj_dict = {}
        for aqe_, url_header, url_suffix_start in query_urls:
            url_suffix_end = url_suffix_start + 357 - 1
            ret = [sqldt_from_appid(url_header, appid, if_full_plan) for appid in range(url_suffix_start, url_suffix_end + 1)]
            obj_dict[aqe_] = {}
            cols =  (DATA_COLNS + ["full_plan"]) if if_full_plan else DATA_COLNS
            df = pd.DataFrame(data=ret, columns=cols)
            obj_dict[aqe_] = df
        PickleUtils.save(obj_dict, out_header, file_name)
        print(f"finished generating tuned objs, cost {time.time() - start}s")
    return obj_dict

def plot_heuristic_obj(dvalue, res_mu, res_std, sql_mu, sql_std, qid, fig_header, fig_name_suffix, ylabel):
    figsize = (4, 2.5)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(y=dvalue, xmin=0, xmax=40, color="red", linestyle="dashed", label=f"{qid}_default")
    ax.errorbar(range(len(res_mu)), res_mu, yerr=res_std, fmt="g-", label=f"{qid}_res")
    ax.errorbar(range(len(sql_mu)), sql_mu, yerr=sql_std, fmt="b-", label=f"{qid}_sql")
    ax.legend()
    ax.set_xlabel("configuration ids")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Resource knobs VS SQL knobs on {fig_name_suffix}")
    plt.tight_layout()
    figpath = f"{fig_header}/{qid}_{fig_name_suffix}.pdf"
    fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
    plt.close()



def analyze_default_objs(out_header, obj_dict):
    print("qid\t aqe_off (s) \t aqe_on (s)")
    df_off, df_on = obj_dict["aqe_off"], obj_dict["aqe_on"]
    lat_mu_off = df_off.groupby(["q_sign", "knob_sign"]).mean()["lat"].loc[QSIGNS].values
    lat_std_off = df_off.groupby(["q_sign", "knob_sign"]).std()["lat"].loc[QSIGNS].values
    lat_mu_on = df_on.groupby(["q_sign", "knob_sign"]).mean()["lat"].loc[QSIGNS].values
    lat_std_on = df_on.groupby(["q_sign", "knob_sign"]).std()["lat"].loc[QSIGNS].values

    for i, (off_mu, off_std, on_mu, on_std) in enumerate(zip(lat_mu_off, lat_std_off, lat_mu_on, lat_std_on)):
        qid = i + 1
        print(f"q{qid}\t {off_mu:.3f} +- {off_std:.3f} \t {on_mu:.3f} +- {on_std:.3f}")

    queries = [f"q{i}" for i in range(1, 1 + 22)]

    x_pos = np.arange(22)
    n = 2
    labels = ["AQE_OFF", "AQE_ON"]
    mu_list = [lat_mu_off, lat_mu_on]
    std_list = [lat_std_off, lat_std_on]
    colors = ["blue", "red"]
    hatches = ["", ""]

    barWidth = 0.4
    r_list = [
        np.arange(len(queries)) - barWidth * ((n - 1) / 2) + barWidth * i
        for i in range(n)
    ]

    figsize = (12, 2.5)
    fig, ax = plt.subplots(figsize=figsize)
    rect_list = [None] * n
    for i in range(n):
        rect_list[i] = ax.bar(r_list[i], mu_list[i], color=colors[i], width=barWidth, yerr=std_list[i],
                              edgecolor='white', label=labels[i], hatch=hatches[i], alpha=.6)
        bar_labels = [f"{e:.0f}" for e in mu_list[i]]
        ax.bar_label(rect_list[i], padding=2, fontsize=8, labels=bar_labels, rotation=0)

    ax.set_xlabel('queries')
    ax.set_xticks(np.arange(len(queries)))
    ax.set_xticklabels(queries)
    ax.set_xlim(
        [-1 * n / 2 * barWidth - (1 - n * barWidth) / 2, len(queries) - 1 + n / 2 * barWidth + (1 - n * barWidth) / 2])
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_ylim([0, max(lat_mu_off + lat_std_off) * 1.2])
    ax.set_ylabel('latency (s)')
    ax.grid(axis="y")
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    plt.tight_layout()
    ax.set_title("TPCH (each query with 3 runs)")
    plt.savefig(f"{out_header}/default_objs_bar.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.show()

def analyze_heuristic_objs(default_obj_dict, heuristic_obj_dict, fig_header):
    os.makedirs(fig_header, exist_ok=True)
    default_objs = default_obj_dict["aqe_on"]
    obj_default_mu = default_objs.groupby(["q_sign", "knob_sign"]).mean().loc[QSIGNS]
    obj_default_std = default_objs.groupby(["q_sign", "knob_sign"]).std().loc[QSIGNS]
    obj_sql_mu = heuristic_obj_dict["sql"].groupby(["q_sign", "knob_sign"]).mean().loc[QSIGNS]
    obj_sql_std = heuristic_obj_dict["sql"].groupby(["q_sign", "knob_sign"]).std().loc[QSIGNS]
    obj_res_mu = heuristic_obj_dict["res"].groupby(["q_sign", "knob_sign"]).mean().loc[QSIGNS]
    obj_res_std = heuristic_obj_dict["res"].groupby(["q_sign", "knob_sign"]).std().loc[QSIGNS]
    for i in range(22):
        q_sign = QSIGNS[i]
        qid = f"q{i + 1}"
        d_mu = obj_default_mu.loc[q_sign].values[0]
        res_mu, res_std = obj_res_mu.loc[q_sign].values, obj_res_std.loc[q_sign].values
        sql_mu, sql_std = obj_sql_mu.loc[q_sign].values, obj_sql_std.loc[q_sign].values

        ids_res = np.argsort(res_mu[:, 0])
        ids_sql = np.argsort(sql_mu[:, 0])

        for i, (fig_name_suffix, ylabel) in enumerate(zip(["lat", "cost"], ["secs", "cost($)"])):
            plot_heuristic_obj(
                d_mu[i],
                res_mu[ids_res][:, i], res_std[ids_res][:, i],
                sql_mu[ids_sql][:, i], sql_std[ids_sql][:, i],
                qid, fig_header, fig_name_suffix, ylabel
            )


def get_po_points(objs):
    mask = is_pareto_efficient(objs)
    po_objs = objs[mask]
    sorted_inds = np.argsort(po_objs[:, 0])  # sorted by the latency
    po_objs = po_objs[sorted_inds]
    return po_objs


def analyze_tuned_objs(default_obj_dict, heuristic_obj_dict, tuned_obj_dict, fig_header, pred_header, pred_name):
    obj_default_mu = default_obj_dict["aqe_on"].groupby(["q_sign", "knob_sign"]).mean().loc[QSIGNS]
    obj_sql_mu = heuristic_obj_dict["sql"].groupby(["q_sign", "knob_sign"]).mean().loc[QSIGNS]
    obj_res_mu = heuristic_obj_dict["res"].groupby(["q_sign", "knob_sign"]).mean().loc[QSIGNS]

    obj_tuned_mu_on = \
    tuned_obj_dict["aqe_on"][["q_sign", "knob_sign", "lat", "cost"]].groupby(["q_sign", "knob_sign"]).mean().loc[QSIGNS]
    obj_tuned_mu_off = \
    tuned_obj_dict["aqe_off"][["q_sign", "knob_sign", "lat", "cost"]].groupby(["q_sign", "knob_sign"]).mean().loc[
        QSIGNS]

    for i in range(22):
        q_sign = QSIGNS[i]
        qid = f"q{i + 1}"

        d = obj_default_mu.loc[q_sign].values
        res, sql = obj_res_mu.loc[q_sign].values, obj_sql_mu.loc[q_sign].values
        tuned_on, tuned_off = obj_tuned_mu_on.loc[q_sign].values, obj_tuned_mu_off.loc[q_sign].values
        res_po, sql_po = get_po_points(res), get_po_points(sql)
        tuned_on_po, tuned_off_po = get_po_points(tuned_on), get_po_points(tuned_off)
        tuned_pred = PickleUtils.load(f"{pred_header}/tpch_100/{i + 1}-1", pred_name)["objs_pred"]

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        plot(
            X=[d[:, 0], res_po[:, 0], sql_po[:, 0], tuned_off_po[:, 0], tuned_on_po[:, 0], tuned_pred[:, 0]],
            Y=[d[:, 1], res_po[:, 1], sql_po[:, 1], tuned_off_po[:, 1], tuned_on_po[:, 1], tuned_pred[:, 1]],
            xlabel="latency (s)", ylabel="cost($)",
            legend=["default", "res_po", "sql_po", "ws_naqe", "ws_aqe", "ws_pred"],
            fmts=["ko", "go--", "bo--", "co--", "mo--", "ro--"],
            axes=ax, figsize=(4.5, 3.5))

        ax.set_title("PO Frontiers")
        plt.tight_layout()
        figpath = f"{fig_header}/{qid}_po.pdf"
        fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
        plt.show()
        plt.close()


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

def analyze_tuned_objs_model_space(
        default_obj_dict, heuristic_obj_dict, fig_header, pred_header, pred_name,
        data_header, ckp_header, ckp_sign,
):
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
    knobs = spark_knobs.knobs
    obj_default_mu = default_obj_dict["aqe_on"].groupby(["q_sign", "knob_sign"]).mean().loc[QSIGNS]
    obj_sql_mu = heuristic_obj_dict["sql"].groupby(["q_sign", "knob_sign"]).mean().loc[QSIGNS]
    obj_res_mu = heuristic_obj_dict["res"].groupby(["q_sign", "knob_sign"]).mean().loc[QSIGNS]

    misc = df, dag_dict, mp, op_groups, col_dict, minmax_dict, spark_knobs
    for i in range(22):
        q_sign = QSIGNS[i]
        qid = f"q{i + 1}"

        d = obj_default_mu.loc[q_sign].values
        res, sql = obj_res_mu.loc[q_sign].values, obj_sql_mu.loc[q_sign].values
        res_po_mask, sql_po_mask = is_pareto_efficient(res), is_pareto_efficient(sql)
        d_signs = obj_default_mu.loc[q_sign].index.tolist()
        res_knob_signs_po = obj_res_mu.loc[q_sign][res_po_mask].index.tolist()
        sql_knob_signs_po = obj_sql_mu.loc[q_sign][sql_po_mask].index.tolist()
        res_knobs_po = [KnobUtils.sign2knobs(s, knobs) for s in res_knob_signs_po]
        sql_knobs_po = [KnobUtils.sign2knobs(s, knobs) for s in sql_knob_signs_po]

        default_pred = get_objs(q_sign, d_signs, misc)
        res_po_pred = get_objs(q_sign, res_knobs_po, misc)
        sql_po_pred = get_objs(q_sign, sql_knobs_po, misc)
        tuned_pred = PickleUtils.load(f"{pred_header}/tpch_100/{i + 1}-1", pred_name)["objs_pred"]
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        plot(
            X=[default_pred[:, 0], res_po_pred[:, 0], sql_po_pred[:, 0], tuned_pred[:, 0]],
            Y=[default_pred[:, 1], res_po_pred[:, 1], sql_po_pred[:, 1], tuned_pred[:, 1]],
            xlabel="latency (s)", ylabel="cost($)",
            legend=["default", "res_po_pred", "sql_po_pred", "ws_pred"],
            fmts=["ko", "go--", "bo--", "ro--"],
            axes=ax, figsize=(4.5, 3.5))

        ax.set_title("PO Frontiers mapping to predicted space")
        plt.tight_layout()
        figpath = f"{fig_header}/{qid}_po_pred.pdf"
        fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
        plt.show()
        plt.close()

def main():
    out_header = "examples/analyze/1.heuristic.vs.q_level_tuning"
    default_obj_dict = get_default_objs(out_header=out_header, file_name="default_objs.pkl")
    heuristic_obj_dict = get_heuristic_objs(out_header=out_header, file_name="res_and_sql_objs.pkl")
    tuned_obj_dict = get_tuned_objs(out_header=out_header, file_name="tuned_objs.pkl", if_full_plan=True)

    analyze_default_objs(out_header, default_obj_dict)

    fig_header = f"{out_header}/fig"
    analyze_heuristic_objs(default_obj_dict, heuristic_obj_dict, fig_header)

    pred_header = "examples/model/spark/out/2.q_level_conf_reco"
    pred_name = "po_points_10000_ws_5000.pkl"
    analyze_tuned_objs(default_obj_dict, heuristic_obj_dict, tuned_obj_dict, fig_header, pred_header, pred_name)

    data_header = "examples/data/spark/cache/tpch_100"
    ckp_header = "examples/model/spark/ckp/tpch_100/GTN/latency/on_off_off_on_on_on"
    ckp_sign = "40a985a643f1d253"
    analyze_tuned_objs_model_space(
        default_obj_dict, heuristic_obj_dict, fig_header, pred_header, pred_name,
        data_header, ckp_header, ckp_sign
    )

if __name__ == '__main__':
    main()

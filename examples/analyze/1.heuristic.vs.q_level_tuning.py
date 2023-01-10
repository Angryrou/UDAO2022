# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: An initial comparison over tuning
# (1) res knobs only (20 samples from lhs);
# (2) sql knobs only (20 samples from lhs);
# (3) 12 knobs together by using PO solutions based on a Q-level model (WS_10000_samples_5000_weights)
#
# Created at 10/01/2023

import time

import pandas as pd

from utils.common import JsonUtils, PickleUtils
from trace.parser.spark import get_cloud_cost
import numpy as np
from matplotlib import pyplot as plt

DATA_COLNS = ["q_sign", "knob_sign", "lat", "cost"]


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
            obj_dict[aqe_] = pd.DataFrame(data=ret, columns=DATA_COLNS + ["full_plan"])
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
            ret = [sqldt_from_appid(url_header, appid) for appid in range(url_suffix_start, url_suffix_end + 1)]
            obj_dict[aqe_] = {}
            df = pd.DataFrame(data=ret, columns=DATA_COLNS)
            obj_dict[aqe_] = df
        PickleUtils.save(obj_dict, out_header, file_name)
        print(f"finished generating tuned objs, cost {time.time() - start}s")
    return obj_dict


def analyze_default_objs(out_header, obj_dict):
    print("qid\t aqe_off (s) \t aqe_on (s)")
    aqe_off_mu, aqe_off_std, aqe_on_mu, aqe_on_std = \
        obj_dict["aqe_off"]["lat_mu"], obj_dict["aqe_off"]["lat_std"], \
        obj_dict["aqe_on"]["lat_mu"], obj_dict["aqe_on"]["lat_std"]

    for i, (off_mu, off_std, on_mu, on_std) in enumerate(zip(aqe_off_mu, aqe_off_std, aqe_on_mu, aqe_on_std)):
        qid = i + 1
        print(f"{qid}\t {off_mu:.3f} +- {off_std:.3f} \t {on_mu:.3f} +- {on_std:.3f}")

    queries = [f"q{i}" for i in range(1, 1 + 22)]

    n = 2
    labels = ["AQE_OFF", "AQE_ON"]
    mu_list = [aqe_off_mu, aqe_on_mu]
    std_list = [aqe_off_std, aqe_on_std]
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
    ax.set_ylim([0, max(aqe_off_mu + aqe_off_std) * 1.2])
    ax.set_ylabel('latency (s)')
    ax.grid(axis="y")
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    plt.tight_layout()
    ax.set_title("TPCH (each query with 3 runs)")
    plt.savefig(f"{out_header}/default_objs_bar.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.show()


if __name__ == '__main__':
    out_header = "examples/analyze/1.heuristic.vs.q_level_tuning"
    default_obj_dict = get_default_objs(out_header=out_header, file_name="default_objs.pkl")
    heuristic_obj_dict = get_heuristic_objs(out_header=out_header, file_name="res_and_sql_objs.pkl")
    tuned_obj_dict = get_tuned_objs(out_header=out_header, file_name="tuned_objs.pkl", if_full_plan=True)

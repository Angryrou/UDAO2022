# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: trigger that after 2.sql_struct_extraction.py
#
# Created at 12/23/22

import argparse, random, os, json, socket, re, ssl
import urllib.request
import itertools
import socks #pip install PySocks

import numpy as np
import pandas as pd
from utils.common import BenchmarkUtils, PickleUtils, JsonUtils, ParquetUtils
from utils.data.dag_sql2stages import get_sub_sqls_using_topdown_tree, get_stage_plans, Node, Edge, QueryPlanTopology
from utils.data.extractor import get_csvs, SqlStruct, get_tr_val_te_masks, get_csvs_stage
from utils.data.feature import CH1_FEATS, CH2_FEATS, CH3_FEATS, CH4_FEATS, OBJS, CH1_FEATS_STAGE, CH2_FEATS_STAGE, \
    CH3_FEATS_STAGE, CH4_FEATS_STAGE, OBJS_STAGE


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--src-path-header", type=str, default="resources/dataset")
        self.parser.add_argument("--cache-header", type=str, default="examples/data/spark/cache")
        self.parser.add_argument("--master-node", type=str, default="node1-opa")
        self.parser.add_argument("--debug", type=int, default=0)
        self.parser.add_argument("--if-plot", type=int, default=1)
        self.parser.add_argument("--seed", type=int, default=42)

    def parse(self):
        return self.parser.parse_args()

def get_dict_sign(d: dict):
    d = dict(sorted(d.items()))
    return JsonUtils.dump2str(d)

def show_qs_operators(id, df):
    d = df.loc[id]
    queryStage_to_nodes = JsonUtils.load_json_from_str(d["queryStage_to_nodes"])
    nodes_map = JsonUtils.load_json_from_str(d["nodes_map"])
    JsonUtils.print_dict({qs_id: [nodes_map[str(nid)] for nid in node_ids]
                         for qs_id, node_ids in queryStage_to_nodes.items()})

def get_data_using_proxy(url, ctx):
    """
    Receive the content of ``url`` using proxy, parse and return as JSON dictionary.
    """
    response = urllib.request.urlopen(url, context=ctx).read().decode("utf-8")
    data = response.read().decode("utf-8")
    return data

def generate_mapping_patch(df_mappings, df_stage, master_node, src_path_header_stage, patch_name):
    # identify the stage missing for all mappings and create a patch accordingly
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    IP_ADDR = 'localhost'
    PORT = 2000
    socks.set_default_proxy(socks.SOCKS5, IP_ADDR, PORT)
    socket.socket = socks.socksocket

    mapping_patch_unfinished = {}
    mapping_patch_unfinished_examples = {}
    pattern1 = re.compile("WholeStageCodegen \([0-9]+\)")
    pattern2 = re.compile("Scan parquet tpch_100\.[a-z]+")

    for id, d in df_mappings.iterrows():
        sid = d["sql_struct_id"]
        mapping_id = d["mapping_sign_id"]
        qs2stage = JsonUtils.load_json_from_str(d["mapping_sign"])
        hit_stages = set(itertools.chain.from_iterable(qs2stage.values()))
        dropped_stages = set([int(s) for s in d["dropped_stages"].split(",") if int(s) >= 0])
        full_stages = set(df_stage.set_index("id").loc[id, "stage_id"].values.tolist())
        missing_stages = full_stages - hit_stages - dropped_stages
        pdict = {**{s: -2 for s in dropped_stages}, **{s: -1 for s in missing_stages}}

        print("-" * 19)
        print(f"sql_struct_id {sid}, mapping_id {mapping_id}, missing stages {missing_stages}")
        qs_dependencies = JsonUtils.load_json_from_str(d["qs_dependencies"])
        queryStage_to_nodes = JsonUtils.load_json_from_str(d["queryStage_to_nodes"])
        print(f"qs_dependencies: {qs_dependencies}")
        nodes_map = JsonUtils.load_json_from_str(d["nodes_map"])

        nodes = {n["nodeId"]: Node(n) for n in JsonUtils.load_json_from_str(d["nodes"])}
        wscg2stages_dict = {int(node.name.split("(")[1].split(")")[0]): node.involved_stages_from_sparkviz
                            for node in nodes.values() if "WholeStageCodegen" in node.name}
        for kwscg, stages in wscg2stages_dict.items():
            assert len(stages) <= 1
        qs2wscg_dict = {qs_id: list(set(nodes[nid].wscg for nid in nids if nodes[nid].wscg >= 0))
                        for qs_id, nids in queryStage_to_nodes.items()}
        wscg2qs_dict = {}
        for qs_id, wscgs in qs2wscg_dict.items():
            for wscg in wscgs:
                if wscg not in wscg2qs_dict:
                    wscg2qs_dict[wscg] = [qs_id]
                else:
                    if qs_id not in wscg2qs_dict[wscg]:
                        wscg2qs_dict[wscg].append(qs_id)

        add_wscg2stages_dict = {}  # to help valid each wscg only maps to one stage
        for s in missing_stages:
            stage_url = f"http://{master_node}:18088/history/{id}/stages/stage/?id={s}&attempt=0"
            web_str = urllib.request.urlopen(stage_url, context=ctx).read().decode("utf-8")
            wscg_list = pattern1.findall(web_str)
            if len(wscg_list) > 0:
                wscg = int(wscg_list[0].split("(")[1].split(")")[0])
                if wscg in add_wscg2stages_dict and add_wscg2stages_dict[wscg] != s:
                    print(f"duplicated mappings from wscg {wscg} to {add_wscg2stages_dict[wscg]} and {s}, "
                          f"manual checking needed...")
                    continue
                add_wscg2stages_dict[wscg] = s
                qs_ids = wscg2qs_dict[wscg]
                if len(qs_ids) == 1:
                    qs_id = qs_ids[0]
                    if len(qs2wscg_dict[qs_id]) == 1:
                        pdict[s] = int(qs_id)
                    else:
                        print(f"multiple mappings from qs {qs_id} to wscg {qs2wscg_dict[qs_id]}, "
                              f"manual checking needed...")
                else:
                    print(f"duplicated mappings from wscg {wscg} to qs {qs_ids}, manual checking needed...")
            elif "Scan parquet tpch_100" in web_str:
                tablescan_list = pattern2.findall(web_str)
                tablescan = tablescan_list[0]
                if_found = False
                for qs_id, node_ids in queryStage_to_nodes.items():
                    for nid in node_ids:
                        if nodes_map[str(nid)] == tablescan:
                            pdict[s] = int(qs_id)
                            if_found = True
                            break
                    if if_found:
                        break
                assert if_found is True
            else:
                raise Exception(f"getting missing stage {s} in {id}")

        if -1 in set(pdict.values()):
            for qs_id, stage_ids in qs2stage.items():
                if len(stage_ids) == 0:
                    nids = ",".join(str(nid) for nid in queryStage_to_nodes[qs_id])
                    operators = ",".join(nodes_map[str(nid)] for nid in queryStage_to_nodes[qs_id])
                    print(f"QueryStage {qs_id}: {nids} + {operators}")

        if sid not in mapping_patch_unfinished:
            mapping_patch_unfinished[sid] = {mapping_id: pdict}
            mapping_patch_unfinished_examples[sid] = {id: pdict}
        else:
            mapping_patch_unfinished[sid][mapping_id] = pdict
            mapping_patch_unfinished_examples[sid][id] = pdict
    JsonUtils.save_json(mapping_patch_unfinished, f"{src_path_header_stage}/unfinished_{patch_name}")
    JsonUtils.save_json(mapping_patch_unfinished_examples, f"{src_path_header_stage}/unfinished_examples_{patch_name}")

def generate_struct_cache(df, cache_header, struct_cache_name):
    struct_dict = df.loc[df.sql_struct_id.drop_duplicates().index].to_dict(orient="index")
    struct_dgl_dict = {d["sql_struct_id"]: SqlStruct(d) for d in struct_dict.values()}
    global_ops = sorted(list(set.union(*[set(v.get_nnames()) for k, v in struct_dgl_dict.items()])))
    dgl_dict = {k: v.get_dgl(global_ops) for k, v in struct_dgl_dict.items()}
    q2struct = {x[0]: x[1] for x in df[["q_sign", "sql_struct_id"]].values}
    struct_cache = {
        "struct_dict": struct_dict,
        "struct_dgl_dict": struct_dgl_dict,
        "global_ops": global_ops,
        "dgl_dict": dgl_dict,
        "q2struct": q2struct
    }
    PickleUtils.save(struct_cache, header=cache_header, file_name=struct_cache_name)
    return struct_cache

def generate_query_cache(df, df_stage, cache_header, query_cache_name, seed):
    head_cols = ["id", "q_sign", "knob_sign", "template", "sampling", "start_timestamp", "cpl_opt_time"]
    ch1_cols, ch2_cols, ch3_cols, ch4_cols = CH1_FEATS, CH2_FEATS, CH3_FEATS, CH4_FEATS
    obj_cols = OBJS

    df[ch4_cols] = df.knob_sign.str.split(",", expand=True)
    df[["k6", "k7", "s4"]] = (df[["k6", "k7", "s4"]] == "True") + 0
    df[ch4_cols] = df[ch4_cols].astype(float)

    df1 = df[["id", "q_sign", "start_timestamp"]]
    df2 = df_stage.merge(df1)
    df3 = df2[df2["first_task_launched_time"] >= df2["start_timestamp"]]
    df4 = df3[["id", "first_task_launched_time", "start_timestamp"]].groupby("id").min()
    df4["cpl_opt_time"] = df4["first_task_launched_time"] - df4["start_timestamp"]
    df = df.merge(df4.reset_index()[["id", "cpl_opt_time"]], on="id", how="left")

    selected_cols = head_cols + ch1_cols + ch2_cols + ch3_cols + ch4_cols + obj_cols
    df4model = df[selected_cols]
    tr_mask, val_mask, te_mask = get_tr_val_te_masks(
        df=df4model, groupby_col1="template", groupby_col2="template",
        frac_val_per_group=0.1, frac_te_per_group=0.1, seed=seed)
    df_tr, df_val, df_te = df4model[tr_mask].dropna(), df4model[val_mask].dropna(), df4model[te_mask].dropna()

    col_dict = {"ch1": ch1_cols, "ch2": ch2_cols, "ch3": ch3_cols, "ch4": ch4_cols, "obj": obj_cols}
    minmax_dict = {}
    for ch in ["ch1", "ch2", "ch3", "ch4", "obj"]:
        df_ = df_tr[col_dict[ch]]
        min_, max_ = df_.min(), df_.max()
        minmax_dict[ch] = {"min": min_, "max": max_}
    query_cache = {
        "full_cols": selected_cols, "col_dict": col_dict, "minmax_dict": minmax_dict,
        "dfs": [df_tr, df_val, df_te]
    }
    PickleUtils.save(query_cache, cache_header, query_cache_name)
    return query_cache

def generate_qs2stage_mapping(df, df_stage, verbose, num_plans, cache_header, df_qs2stage_mapping_name):
    df = df.set_index("id").loc[df_stage.id.unique().tolist()]
    # df["mapping_sign"] = df.apply(get_mapping_sign, axis=1)
    mapping_signs, queryStage_to_nodes_list, qs_dependencies_list, nodes_map_list = \
        [None] * len(df), [None] * len(df), [None] * len(df), [None] * len(df)
    no_overlap_ids = set()
    for i, v in enumerate(df.to_dict(orient="index").values()):
        nodes = [Node(n) for n in JsonUtils.load_json_from_str(v["nodes"])]
        edges = JsonUtils.load_json_from_str(v["edges"])
        full_plan = QueryPlanTopology(nodes, edges)
        queryStage_to_nodes, qs_dependencies = get_sub_sqls_using_topdown_tree(full_plan)
        queryStages = get_stage_plans(full_plan, queryStage_to_nodes)
        nodes_map = {n.nid: n.name for n in nodes}
        edges_map = {}
        for e in edges:
            if e["fromId"] in edges_map:
                edges_map[e["fromId"]].add(e["toId"])
            else:
                edges_map[e["fromId"]] = {e["toId"]}
        qs2stage = {}
        for qs_id, qs in queryStages.items():
            sset = set()
            seq_stage_cands = set()
            for n in qs.nodes:
                if "exchange" == n.name.lower():
                    if len(edges_map[n.nid]) == 1:
                        child = list(edges_map[n.nid])[0]
                        if child in queryStage_to_nodes[qs_id]:
                            sset |= n.get_exchange_read(n.metrics)
                        else:
                            sset |= n.get_exchange_write(n.metrics)
                    else:
                        children = list(edges_map[n.nid])
                        if any(child in queryStage_to_nodes[qs_id] for child in children):
                            seq_stage_cands |= n.get_exchange_read(n.metrics)
                        else:
                            sset |= n.get_exchange_write(n.metrics)
                elif "broadcastexchange" == n.name.lower():
                    if verbose:
                        print("not implemented due multiple children in BroadcastExchange")
                elif "subquery" == n.name.lower():
                    if verbose:
                        print("subquery")
                    assert "stage" not in json.dumps(n.metrics)
                elif "subquerybroadcast" == n.name.lower():
                    if verbose:
                        print("subquerybroadcast")
                    assert "stage" not in json.dumps(n.metrics)
                else:
                    sset |= set(n.involved_stages_from_sparkviz)
            if len(seq_stage_cands) > 0:
                # if v["q_sign"] not in ("q18-2"):
                # assert len(seq_stage_cands & sset) == 1, "0 or 2+ overlaps b/w sset and seq_stage_cands"
                if len(seq_stage_cands & sset) == 0:
                    no_overlap_ids.add(i)
            qs2stage[qs_id] = sset
        qs2stage = {qs: sorted(list(s)) for qs, s in qs2stage.items()}

        mapping_signs[i] = get_dict_sign(qs2stage)
        queryStage_to_nodes_list[i] = get_dict_sign({k: sorted(v) for k, v in queryStage_to_nodes.items()})
        qs_dependencies_list[i] = JsonUtils.dump2str(sorted(qs_dependencies))
        nodes_map_list[i] = get_dict_sign(nodes_map)
        if (i + 1) % (len(df) // 10) == 0:
            print(f"{i + 1}/{len(df)}")

    df["mapping_sign"] = mapping_signs
    df["queryStage_to_nodes"] = queryStage_to_nodes_list
    df["qs_dependencies"] = qs_dependencies_list
    df["nodes_map"] = nodes_map_list

    # verify queryStage_to_nodes, qs2o_dict and qs_dependencies are the same of each sid
    assert df[["sql_struct_id", "queryStage_to_nodes"]].drop_duplicates().shape[0] == num_plans
    assert df[["sql_struct_id", "qs_dependencies"]].drop_duplicates().shape[0] == num_plans
    assert df[["sql_struct_id", "nodes_map"]].drop_duplicates().shape[0] == num_plans

    # verify that queries from the same sid/mapping_id is missing the same stage(s)
    df_stage_short = df_stage[["id", "stage_id", "first_task_launched_time", "stage_latency"]].copy()
    df_stage_short["stage_finish_time"] = df_stage_short["first_task_launched_time"] + df_stage_short["stage_latency"]
    df_stage_short = df_stage_short.merge(df[["start_timestamp"]].reset_index(), how="inner", on=["id"])
    df_stage_filtered = df_stage_short[df_stage_short["stage_finish_time"] <= df_stage_short["start_timestamp"]]
    df_stage_filtered = df_stage_filtered[["id", "stage_id"]].groupby("id").stage_id.apply(
        lambda x: ','.join(map(str, x)))
    df["dropped_stages"] = "-1"
    df.loc[df_stage_filtered.index, "dropped_stages"] = df_stage_filtered.values
    assert df[["sql_struct_id", "dropped_stages"]].drop_duplicates().shape[0] == num_plans
    assert df[["sql_struct_id", "mapping_sign", "dropped_stages"]].drop_duplicates().shape[0] == \
           df[["sql_struct_id", "mapping_sign"]].drop_duplicates().shape[0]
    df_mappings = df[["sql_struct_id", "mapping_sign", "dropped_stages", "queryStage_to_nodes",
                      "qs_dependencies", "nodes_map"]].drop_duplicates().sort_values(["sql_struct_id", "mapping_sign"])
    df_mappings["nodes"] = df.loc[df_mappings.index, "nodes"]
    df_mappings["mapping_sign_id"] = list(range(len(df_mappings)))
    mapping_sign2id = {sign: id for id, sign in enumerate(df_mappings["mapping_sign"])}
    df["mapping_sign_id"] = df.mapping_sign.apply(lambda x: mapping_sign2id[x])
    n1, n2 = df_qs2stage_mapping_name.split(".")
    df_qs2stage_mapping_name_full = f"{n1}_full.{n2}"
    ParquetUtils.parquet_write(df, cache_header, df_qs2stage_mapping_name_full)
    ParquetUtils.parquet_write(df_mappings, cache_header, df_qs2stage_mapping_name)
    print(f"generated {df_qs2stage_mapping_name}")
    return df_mappings, df

def generate_stage_cache(df, df_stage, master_node, verbose, num_plans, src_path_header_stage, query_cache,
                         struct_dgl_dict, cache_header, stage_cache_name):
    # 3.1 generate QueryStage to Stage mappings
    df_qs2stage_mapping_name = "df_qs2stage_mapping.parquet"
    try:
        n1, n2 = df_qs2stage_mapping_name.split(".")
        df_mappings = ParquetUtils.parquet_read(cache_header, df_qs2stage_mapping_name)
        df_qs2stage_mapping_name_full = f"{n1}_full.{n2}"
        df_processed = ParquetUtils.parquet_read(cache_header, df_qs2stage_mapping_name_full)
        print(f"found the df_qs2stage_mapping at {cache_header}/{df_qs2stage_mapping_name}")
    except:
        print(f"cannot find {df_qs2stage_mapping_name}, generating...")
        df_mappings, df_processed = generate_qs2stage_mapping(
            df, df_stage, verbose, num_plans, cache_header, df_qs2stage_mapping_name)
        print(f"generated df_qs2stage_mapping at {cache_header}/{df_qs2stage_mapping_name}")

    # 3.2 patching the QueryStage to Stage mappings
    patch_name = "mapping_patch.json"
    try:
        df_stage_uniq = df_stage[df_stage.id.isin(df_mappings.index)]
        df_q2stages = df_stage_uniq.groupby("id")["stage_id"].apply(set)
        mapping_patch = JsonUtils.load_json(f"{src_path_header_stage}/{patch_name}")
        print(f"found the mapping_patch at {src_path_header_stage}/{patch_name}")
    except:
        print(f"cannot find the target patch, generating the auto-filled version...")
        generate_mapping_patch(df_mappings, df_stage, master_node, src_path_header_stage, patch_name)
        raise Exception(f"additional manual work needed to finalize the patch")

    mapping_patch = {int(k): {int(vk): {int(vvk): vvv for vvk, vvv in vv.items()} for vk, vv in v.items()}
                     for k, v in mapping_patch.items()}
    assert sum(len(v) for v in mapping_patch.values()) == len(df_mappings)
    qs2stage_dict = {}
    for id, d in df_mappings.iterrows():
        sid, mapping_id = d["sql_struct_id"], d["mapping_sign_id"]
        qs2stage = {int(k): v for k, v in JsonUtils.load_json_from_str(d["mapping_sign"]).items()}
        for stage_id, qs_id in mapping_patch[sid][mapping_id].items():
            assert qs_id != -1, "unfinished patching"
            if qs_id >= 0:
                qs2stage[qs_id].append(stage_id)
        # validating: the involved stages is a subset of all stages
        stages_involved = list(itertools.chain.from_iterable(qs2stage.values()))
        assert len(stages_involved) == len(set(stages_involved)), f"duplicated stages are counted in {id}"
        stages_all = df_q2stages.loc[id]
        assert set(stages_involved) <= stages_all, "stages involved are more than total stages"
        qs2stage_dict[mapping_id] = qs2stage
        df_mappings.loc[id, "patched_mapping"] = JsonUtils.dump2str(qs2stage)
    assert len(qs2stage_dict) == len(df_mappings)
    print(f"patched df_qs2stage_mapping and get the mapping at `qs2stage_dict`")
    stage2qs_dict = {mapping_id: {stage: qs for qs, stages in qs2stage.items() for stage in stages}
                     for mapping_id, qs2stage in qs2stage_dict.items()}

    derived_cols = ["id", "q_sign", "template"] + CH1_FEATS + CH4_FEATS
    dfs_stage = []
    all_ids = set(df_stage.id)
    head_cols = ["id", "q_sign", "sampling"]
    ch1_cols, ch2_cols, ch3_cols, ch4_cols = CH1_FEATS_STAGE, CH2_FEATS_STAGE, CH3_FEATS_STAGE, CH4_FEATS_STAGE
    obj_cols = OBJS_STAGE
    selected_cols = head_cols + ch1_cols + ch2_cols + ch3_cols + ch4_cols + obj_cols
    col_dict = {"ch1": ch1_cols, "ch2": ch2_cols, "ch3": ch3_cols, "ch4": ch4_cols, "obj": obj_cols}
    id2mapping_id = {appid: x["mapping_sign_id"] for appid, x in df_processed.iterrows()}

    for split, df in zip(["tr", "val", "te"], query_cache["dfs"]):
        appid_index = list(all_ids & set(df.id))
        df = df[df.id.isin(appid_index)][derived_cols].copy()
        df["mapping_sign_id"] = df.id.apply(lambda x: id2mapping_id[x])
        df["qs_id"] = df.mapping_sign_id.apply(lambda mapping_id: sorted(list(qs2stage_dict[mapping_id].keys())))
        df_stage1 = df.explode("qs_id").reset_index(drop=True)  # set the rows
        df_stage2 = df_stage.set_index("id").loc[appid_index].reset_index()
        df_stage2["mapping_sign_id"] = df_stage2.id.apply(lambda x: id2mapping_id[x])
        df_stage2["qs_id"] = [stage2qs_dict[v["mapping_sign_id"]][v["stage_id"]]
                              if v["stage_id"] in stage2qs_dict[v["mapping_sign_id"]] else -1
                              for k, v in df_stage2.iterrows()]
        df_stage2 = df_stage2[df_stage2["qs_id"] >= 0]
        df_stage2 = df_stage2.groupby(["id", "qs_id"]).agg(
            starting_time=("first_task_launched_time", "min"),
            stage_latency=("stage_latency", "sum"),
            stage_dt=("stage_dt", "sum"),
            task_num=("task_num", "max"),
            input_bytes=("input_bytes", "max"),
            input_records=("input_records", "max"),
            sr_bytes=("sr_bytes", "max"),
            sr_records=("sr_records", "max"),
            output_bytes=("output_bytes", "max"),
            output_records=("output_records", "max"),
            sw_bytes=("sw_bytes", "max"),
            sw_records=("sw_records", "max"),
            timestamp=("timestamp", "first"),
            m1=("m1", "first"),
            m2=("m2", "first"),
            m3=("m3", "first"),
            m4=("m4", "first"),
            m5=("m5", "first"),
            m6=("m6", "first"),
            m7=("m7", "first"),
            m8=("m8", "first"),
            sampling=("sampling", "max"),
            mapping_sign_id=("mapping_sign_id", "max")
        ).reset_index()
        assert len(df_stage1) == len(df_stage2)
        df_stage_ = df_stage1.merge(df_stage2, how="inner", on=["id", "qs_id", "mapping_sign_id"])
        assert len(df_stage_) == len(df_stage1)
        df_stage_[["input_mb", "sr_mb", "output_mb", "sw_mb"]] = \
            df_stage_[["input_bytes", "sr_bytes", "output_bytes", "sw_bytes"]] / 1024 / 1024
        df_stage_[["input_mb_log", "sr_mb_log", "input_records_log", "sr_records_log"]] = \
            np.log(df_stage_[["input_mb", "sr_mb", "input_records", "sr_records"]] + 1)
        df_stage_[["output_mb_log", "sw_mb_log", "output_records_log", "sw_records_log"]] = \
            np.log(df_stage_[["output_mb", "sw_mb", "output_records", "sw_records"]] + 1)
        df_stage_ = df_stage_[selected_cols]
        dfs_stage.append(df_stage_)
        print(f"finish generating for df_stage {split}")

    minmax_dict = {}
    for ch in ["ch2", "ch3", "ch4", "obj"]:
        df_ = dfs_stage[0][col_dict[ch]]
        min_, max_ = df_.min(), df_.max()
        minmax_dict[ch] = {"min": min_, "max": max_}

    qs2o_dict = {}
    qs_dependencies_dict = {}
    for id, d in df_mappings[["sql_struct_id", "queryStage_to_nodes", "qs_dependencies"]].iterrows():
        sid = d["sql_struct_id"]
        queryStage_to_nodes = {int(k): v for k, v in JsonUtils.load_json_from_str(d["queryStage_to_nodes"]).items()}
        qs2o_dict[sid] = {qs: [struct_dgl_dict[sid].p1.old["nids_old2new"][nid] for nid in nodes]
                          for qs, nodes in queryStage_to_nodes.items()}
        qs_dependencies_dict[sid] = JsonUtils.load_json_from_str(d["qs_dependencies"])

    stage_cache = {
        "full_cols": selected_cols,
        "col_dict": col_dict,
        "minmax_dict": minmax_dict,
        "dfs": dfs_stage,
        "qs2stage_dict": qs2stage_dict,
        "stage2qs_dict": stage2qs_dict,
        "id2mapping_id": id2mapping_id,
        "qs2o_dict": qs2o_dict,
        "qs_dependencies_dict": qs_dependencies_dict
    }
    PickleUtils.save(stage_cache, cache_header, stage_cache_name)

    return stage_cache


if __name__ == "__main__":
# def main():
    args = Args().parse()
    bm = args.benchmark.lower()
    sf = args.scale_factor
    src_path_header = args.src_path_header
    cache_header = f"{args.cache_header}/{bm}_{sf}"
    debug = False if args.debug == 0 else True
    if_plot = False if args.if_plot == 0 else True
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)

    templates = [f"q{i}" for i in BenchmarkUtils.get(bm)]
    src_path_header_query = os.path.join(src_path_header, f"{bm}_{sf}_query_traces")
    df = get_csvs(templates, src_path_header_query, cache_header, samplings=["lhs", "bo"])

    src_path_header_stage = os.path.join(src_path_header, f"{bm}_{sf}_stage_traces")
    df_stage = get_csvs_stage(src_path_header_stage, cache_header, samplings=["lhs", "bo"])
    verbose = False

    # 1. get unique query structures with operator types.
    struct_cache_name = "struct_cache.pkl"
    try:
        struct_cache = PickleUtils.load(cache_header, struct_cache_name)
        print(f"found cached structures at {cache_header}/{struct_cache_name}")
    except:
        print(f"cannot find cached structure, start generating...")
        struct_cache = generate_struct_cache(df, cache_header, struct_cache_name)
        print(f"generated cached structure at {cache_header}/{struct_cache_name}")
    struct_dict = struct_cache["struct_dict"]
    struct_dgl_dict = struct_cache["struct_dgl_dict"]
    num_plans = len(struct_dict)

    # 2. generate data for query-level modeling
    query_cache_name = "query_level_cache_data.pkl"
    try:
        query_cache = PickleUtils.load(cache_header, query_cache_name)
        print(f"found cached query_cache at {cache_header}/{query_cache_name}")
    except:
        print(f"cannot find cached query_cache, start generating...")
        query_cache = generate_query_cache(df, df_stage, cache_header, query_cache_name, seed)
        print(f"generated query_cache at {cache_header}/{query_cache_name}")

    # 3. generate data for stage-level modeling
    stage_cache_name = "stage_level_cache_data.pkl"
    try:
        stage_cache = PickleUtils.load(cache_header, stage_cache_name)
        print(f"found cached stage_cache at {cache_header}/{stage_cache_name}")
    except:
        print(f"cannot find cached stage_cache, start generating...")
        stage_cache = generate_stage_cache(df, df_stage, args.master_node, verbose, num_plans, src_path_header_stage,
                                           query_cache, struct_dgl_dict, cache_header, stage_cache_name)
        print(f"generated stage_cache at {cache_header}/{stage_cache_name}")

    # todo: add query compilation and optimization time into query_cache["dfs"]
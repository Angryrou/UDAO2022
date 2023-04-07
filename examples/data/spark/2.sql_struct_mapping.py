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
    # IP_ADDR = 'localhost'
    # PORT = 2000

    # ctx = ssl.create_default_context()
    # ctx.check_hostname = False
    # ctx.verify_mode = ssl.CERT_NONE

    # request = urllib.request.Request(url)
    # socks.set_default_proxy(socks.SOCKS5, IP_ADDR, PORT)
    # socket.socket = socks.socksocket
    response = urllib.request.urlopen(url, context=ctx).read().decode("utf-8")
    data = response.read().decode("utf-8")
    return data

def get_mapping_sign(v):
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
    qs2s = {}
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
        qs2s[qs_id] = sset
    qs2s = {qs: sorted(list(s)) for qs, s in qs2s.items()}
    return get_dict_sign(qs2s)

if __name__ == "__main__":
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
    struct_cache = PickleUtils.load(cache_header, struct_cache_name)
    struct_dict = struct_cache["struct_dict"]
    struct_dgl_dict = struct_cache["struct_dgl_dict"]
    global_ops = struct_cache["global_ops"]
    dgl_dict = struct_cache["dgl_dict"]
    q2struct = struct_cache["q2struct"]
    num_plans = len(struct_dict)
    print(f"find cached structures at {cache_header}/{struct_cache_name}")

    # generate data for query-level modeling
    query_cache_name = "query_level_cache_data.pkl"
    query_cache = PickleUtils.load(cache_header, query_cache_name)

    # 2. generate QueryStage to Stage mappings
    # get the QueryStage topologies
    try:
        df = ParquetUtils.parquet_read(cache_header, "df_qs2stage_mapping.parquet")
        print(f"found processed df for mapping")
    except:
        print("do not find df_mapping.parquet, generating...")
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
            qs2s = {}
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
                qs2s[qs_id] = sset
            qs2s = {qs: sorted(list(s)) for qs, s in qs2s.items()}

            mapping_signs[i] = get_dict_sign(qs2s)
            queryStage_to_nodes_list[i] = get_dict_sign({k: sorted(v) for k, v in queryStage_to_nodes.items()})
            qs_dependencies_list[i] = JsonUtils.dump2str(sorted(qs_dependencies))
            nodes_map_list[i] = get_dict_sign(nodes_map)
            if (i+1) % (len(df) // 10) == 0:
                print(f"{i+1}/{len(df)}")

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
        df_stage_filtered = df_stage_filtered[["id", "stage_id"]].groupby("id").stage_id.apply(lambda x: ','.join(map(str, x)))
        df["dropped_stages"] = "-1"
        df.loc[df_stage_filtered.index, "dropped_stages"] = df_stage_filtered.values
        assert df[["sql_struct_id", "dropped_stages"]].drop_duplicates().shape[0] == num_plans
        assert df[["sql_struct_id", "mapping_sign", "dropped_stages"]].drop_duplicates().shape[0] == df[["sql_struct_id", "mapping_sign"]].drop_duplicates().shape[0]
        ParquetUtils.parquet_write(df, cache_header, "df_qs2stage_mapping.parquet")
        print("generated df_qs2stage_mapping.parquet")

    # create patches
    # identify the stage missing for all mappings and create a patch accordingly
    patch_name = "mapping_patch.json"
    df_mappings = df[["sql_struct_id", "mapping_sign", "dropped_stages", "queryStage_to_nodes",
                      "qs_dependencies", "nodes_map"]].drop_duplicates().sort_values(["sql_struct_id", "mapping_sign"])
    df_mappings["nodes"] = df.loc[df_mappings.index, "nodes"]
    df_mappings["mapping_sign_id"] = list(range(len(df_mappings)))
    print(df_mappings.shape)
    mapping_patch_unfinished = {}
    mapping_patch_unfinished_examples = {}
    pattern1 = re.compile("WholeStageCodegen \([0-9]+\)")
    pattern2 = re.compile("Scan parquet tpch_100\.[a-z]+")


    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    IP_ADDR = 'localhost'
    PORT = 2000
    socks.set_default_proxy(socks.SOCKS5, IP_ADDR, PORT)
    socket.socket = socks.socksocket

    for id, d in df_mappings.iterrows():
        sid = d["sql_struct_id"]
        mapping_id = d["mapping_sign_id"]
        qs2s = JsonUtils.load_json_from_str(d["mapping_sign"])
        hit_stages = set(itertools.chain.from_iterable(qs2s.values()))
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

        add_wscg2stages_dict = {} # to help valid each wscg only maps to one stage
        for s in missing_stages:
            stage_url = f"http://{args.master_node}:18088/history/{id}/stages/stage/?id={s}&attempt=0"
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
            for qs_id, stage_ids in qs2s.items():
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


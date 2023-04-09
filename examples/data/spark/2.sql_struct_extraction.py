# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description:
#
# Created at 12/23/22

import argparse, random, os, json

import numpy as np
import pandas as pd
from utils.common import BenchmarkUtils, PickleUtils, JsonUtils
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
        self.parser.add_argument("--debug", type=int, default=0)
        self.parser.add_argument("--if-plot", type=int, default=1)
        self.parser.add_argument("--seed", type=int, default=42)

    def parse(self):
        return self.parser.parse_args()


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
    try:
        struct_cache = PickleUtils.load(cache_header, struct_cache_name)
        struct_dict = struct_cache["struct_dict"]
        struct_dgl_dict = struct_cache["struct_dgl_dict"]
        global_ops = struct_cache["global_ops"]
        dgl_dict = struct_cache["dgl_dict"]
        q2struct = struct_cache["q2struct"]
        print(f"find cached structures at {cache_header}/{struct_cache_name}")
    except:
        print(f"cannot find cached structure, start generating...")
        struct_dict = df.loc[df.sql_struct_id.drop_duplicates().index].to_dict(orient="index")
        struct_dgl_dict = {d["sql_struct_id"]: SqlStruct(d) for d in struct_dict.values()}
        global_ops = sorted(list(set.union(*[set(v.get_nnames()) for k, v in struct_dgl_dict.items()])))
        dgl_dict = {k: v.get_dgl(global_ops) for k, v in struct_dgl_dict.items()}
        q2struct = {x[0]: x[1] for x in df[["q_sign", "sql_struct_id"]].values}
        PickleUtils.save({
            "struct_dict": struct_dict,
            "struct_dgl_dict": struct_dgl_dict,
            "global_ops": global_ops,
            "dgl_dict": dgl_dict,
            "q2struct": q2struct
        }, header=cache_header, file_name=struct_cache_name)
        print(f"generated cached structure at {cache_header}/{struct_cache_name}")

    # generate data for query-level modeling
    query_cache_name = "query_level_cache_data.bak.pkl"
    try:
        query_cache = PickleUtils.load(cache_header, query_cache_name)
    except:
        head_cols = ["id", "q_sign", "knob_sign", "template", "sampling", "start_timestamp"]
        ch1_cols, ch2_cols, ch3_cols, ch4_cols = CH1_FEATS, CH2_FEATS, CH3_FEATS, CH4_FEATS
        obj_cols = OBJS

        df[ch4_cols] = df.knob_sign.str.split(",", expand=True)
        df[["k6", "k7", "s4"]] = (df[["k6", "k7", "s4"]] == "True") + 0
        df[ch4_cols] = df[ch4_cols].astype(float)

        selected_cols = head_cols + ch1_cols + ch2_cols + ch3_cols + ch4_cols + obj_cols
        df4model = df[selected_cols]
        tr_mask, val_mask, te_mask = get_tr_val_te_masks(
            df=df4model, groupby_col1="template", groupby_col2="template",
            frac_val_per_group=0.1, frac_te_per_group=0.1, seed=seed)
        df_tr, df_val, df_te = df4model[tr_mask], df4model[val_mask], df4model[te_mask]
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

    stage_cache_name = "stage_level_cache_data.bak.pkl"
    try:
        stage_cache = PickleUtils.load(cache_header, stage_cache_name)
    except:
        appids_uniq = set([v["id"] for v in struct_dict.values()])
        df_stage_uniq = df_stage[df_stage.id.isin(appids_uniq)]
        df_q2s = df_stage_uniq.groupby("id")["stage_id"].apply(set)
        sid2appid = {v["sql_struct_id"]: v["id"] for v in struct_dict.values()}
        qs2s_dict = {}
        qs2s_missing_dict = {}
        nodes_map_dict = {}
        queryStage_to_nodes_dict = {}
        qs_dependencies_dict = {}
        for index, v in struct_dict.items():
            print()
            print(f"start working on {index}, {v['name']}")

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
            for qsid, qs in queryStages.items():
                sset = set()
                seq_stage_cands = set()
                for n in qs.nodes:
                    if "exchange" == n.name.lower():
                        if len(edges_map[n.nid]) == 1:
                            child = list(edges_map[n.nid])[0]
                            if child in queryStage_to_nodes[qsid]:
                                sset |= n.get_exchange_read(n.metrics)
                            else:
                                sset |= n.get_exchange_write(n.metrics)
                        else:
                            children = list(edges_map[n.nid])
                            if any(child in queryStage_to_nodes[qsid] for child in children):
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
                    assert len(seq_stage_cands & sset) == 1, "0 or 2+ overlaps b/w sset and seq_stage_cands"
                qs2s[qsid] = sset
            print(qs2s)
            stages_hit = set.union(*qs2s.values())
            assert stages_hit <= df_q2s[v["id"]]
            stages_missing = df_q2s[v["id"]] - stages_hit
            sid = v["sql_struct_id"]
            qs2s_dict[sid] = qs2s
            qs2s_missing_dict[sid] = stages_missing
            print(f"sql_struct_id {sid} ({index}) missing stages {stages_missing}")
            print(f"{qs_dependencies}")
            print(f"following queryStages missing matches:")
            for qsid, sids in qs2s.items():
                if len(sids) == 0:
                    nids = ",".join(str(nid) for nid in queryStage_to_nodes[qsid])
                    operators = ",".join(nodes_map[nid] for nid in queryStage_to_nodes[qsid])
                    print(f"QueryStage {qsid}: {nids} + {operators}")
            nodes_map_dict[sid] = nodes_map
            queryStage_to_nodes_dict[sid] = queryStage_to_nodes
            qs_dependencies_dict[sid] = qs_dependencies
        patch_name = "manual_patch.json"
        try:
            qs2s_patch = JsonUtils.load_json(f"{src_path_header_stage}/{patch_name}")
            qs2s_patch = {int(k): {int(vk): vv for vk, vv in v.items()} for k, v in qs2s_patch.items()}
            for sid in qs2s_dict.keys():
                print(f"--- patching {sid} ---")
                qs2s = qs2s_dict[sid]
                patch = qs2s_patch[sid]
                for pk, pv in patch.items():
                    if pv == -1:
                        continue
                    qs2s[pv].add(pk)
                stages = df_q2s[sid2appid[sid]]
                stages_hit = set.union(*qs2s.values())
                assert stages_hit <= stages

                nodes_map = nodes_map_dict[sid]
                queryStage_to_nodes = queryStage_to_nodes_dict[sid]
                for qsid, sids in qs2s.items():
                    if len(sids) == 0:
                        nids = ",".join(str(nid) for nid in queryStage_to_nodes[qsid])
                        operators = ",".join(nodes_map[nid] for nid in queryStage_to_nodes[qsid])
                        print(f"QueryStage {qsid}: {nids} + {operators}")
                qs2s_dict[sid] = qs2s
        except:
            qs2s_patch_unfinish = {}
            for sid, v in qs2s_missing_dict.items():
                qs2s_patch_unfinish[sid] = {vi: -1 for vi in v}
            JsonUtils.save_json(qs2s_patch_unfinish, f"{src_path_header_stage}/unfinished_{patch_name}")
            raise Exception("Incomplete Mapping from QueryStage to Stage")

        qs2stage_dict = {k: {vk: list(vv) for vk, vv in v.items()} for k, v in qs2s_dict.items()}
        stage2qs_dict = {}
        for qid, qs2stage in qs2stage_dict.items():
            stage2qs_dict[qid] = {}
            for qs, s in qs2stage.items():
                for si in s:
                    stage2qs_dict[qid][si] = qs
        qs2o_dict = {sid: {vk: [struct_dgl_dict[sid].p1.old["nids_old2new"][vvi] for vvi in vv] for vk, vv in v.items()}
                     for sid, v in queryStage_to_nodes_dict.items()}

        derived_cols = ["id", "q_sign", "template"] + CH1_FEATS + CH4_FEATS
        dfs_stage = []
        all_ids = set(df_stage.id)
        head_cols = ["id", "q_sign", "sampling"]
        ch1_cols, ch2_cols, ch3_cols, ch4_cols = CH1_FEATS_STAGE, CH2_FEATS_STAGE, CH3_FEATS_STAGE, CH4_FEATS_STAGE
        obj_cols = OBJS_STAGE
        selected_cols = head_cols + ch1_cols + ch2_cols + ch3_cols + ch4_cols + obj_cols
        col_dict = {"ch1": ch1_cols, "ch2": ch2_cols, "ch3": ch3_cols, "ch4": ch4_cols, "obj": obj_cols}

        for split, df in zip(["tr", "val", "te"], query_cache["dfs"]):
            appid_index = list(all_ids & set(df.id))
            df = df[df.id.isin(appid_index)][derived_cols].copy()
            id2sid = {x["id"]: x["sql_struct_id"] for k, x in df.iterrows()}
            df["qs_id"] = df.sql_struct_id.apply(lambda sid: sorted(list(qs2s_dict[sid].keys())))
            df_stage1 = df.explode("qs_id").reset_index(drop=True) # set the rows
            df_stage2 = df_stage.set_index("id").loc[appid_index].reset_index()
            df_stage2["sid"] = df_stage2.id.apply(lambda x: id2sid[x])
            df_stage2["qs_id"] = [stage2qs_dict[v["sid"]][v["stage_id"]] if v["stage_id"] in stage2qs_dict[v["sid"]] else -1
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
                sid=("sid", "max")
            ).reset_index()
            assert len(df_stage1) == len(df_stage2)
            df_stage_ = df_stage1.merge(df_stage2, how="inner", on=["id", "qs_id"])
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

        stage_cache = {
            "full_cols": selected_cols,
            "col_dict": col_dict,
            "minmax_dict": minmax_dict,
            "dfs": dfs_stage,
            "qs2stage_dict": qs2stage_dict,
            "stage2qs_dict": stage2qs_dict,
            "qs2o_dict": qs2o_dict,
            "qs_dependencies_dict": qs_dependencies_dict
        }
        PickleUtils.save(stage_cache, cache_header, stage_cache_name)


    # 2. get structure mappings
    #   - QueryStage -> [operators]
    #   - SQL -> [QueryStages] + {QueryStageDep - a DGL} / blocked by Exchanges and Subqueries
    #   - QueryStage -> [stages]

    # 3. generate data for queryStage-level modeling

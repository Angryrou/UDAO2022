import argparse, os, json

from utils.common import BenchmarkUtils, JsonUtils
from utils.data.dag_sql2stages import get_sub_sqls_using_topdown_tree, get_stage_plans, Node, Edge, QueryPlanTopology
from utils.data.extractor import get_csvs, get_csvs_stage


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--src-path-header", type=str, default="resources/dataset")
        self.parser.add_argument("--cache-header", type=str, default="examples/data/spark/cache")
        self.parser.add_argument("--debug", type=int, default=0)

    def parse(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = Args().parse()
    bm = args.benchmark.lower()
    sf = args.scale_factor
    src_path_header = args.src_path_header
    cache_header = f"{args.cache_header}/{bm}_{sf}"

    templates = [f"q{i}" for i in BenchmarkUtils.get(bm)]
    src_path_header_query = os.path.join(src_path_header, f"{bm}_{sf}_query_traces")
    df = get_csvs(templates, src_path_header_query, cache_header, samplings=["lhs", "bo"])
    src_path_header_stage = os.path.join(src_path_header, f"{bm}_{sf}_stage_traces")
    df_stage = get_csvs_stage(src_path_header_stage, cache_header, samplings=["lhs", "bo"])

    struct_dict = df.loc[df.sql_struct_id.drop_duplicates().index].to_dict(orient="index")

    appids_uniq = set([v["id"] for v in struct_dict.values()])
    df_stage_uniq = df_stage[df_stage.id.isin(appids_uniq)]
    df_q2s = df_stage_uniq.groupby("id")["stage_id"].apply(set)
    sid2appid = {v["sql_struct_id"]: v["id"] for v in struct_dict.values()}
    qs2s_dict = {}
    qs2s_missing_dict = {}
    nodes_map_dict = {}
    queryStage_to_nodes_dict = {}
    qs_dependencies_dict = {}
    verbose = False

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
            # if len(seq_stage_cands) > 0:
            #     if len(seq_stage_cands & sset) > 1:
            #         print("something")
            #     assert len(seq_stage_cands & sset) == 1, "0 or 2+ overlaps b/w sset and seq_stage_cands"
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
            print(f"--- start patching {sid} ---")
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
            print(f"Done")
    except:
        qs2s_patch_unfinish = {}
        for sid, v in qs2s_missing_dict.items():
            qs2s_patch_unfinish[sid] = {vi: -1 for vi in v}
        JsonUtils.save_json(qs2s_patch_unfinish, f"{src_path_header_stage}/unfinished_{patch_name}")
        raise Exception("Incomplete Mapping from QueryStage to Stage")

# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: extract data to support variable-length of parameters (tempory use based on the old traces)
#
# Created at 15/06/2023
from utils.common import PickleUtils
import dgl
import networkx as nx
import torch as th

from utils.data.extractor import plot_nx_graph


def is_split(name):
    return True if "exchange" in name.lower() or "subquery" in name.lower() else False

def get_unconnected_components(graph):
    num_nodes = graph.number_of_nodes()
    visited = [False] * num_nodes

    components = []

    for node in range(num_nodes):
        if not visited[node]:
            component = []
            stack = [node]

            while stack:
                current_node = stack.pop()
                if not visited[current_node]:
                    visited[current_node] = True
                    component.append(current_node)
                    neighbors = graph.successors(current_node).tolist()
                    stack.extend(neighbors)

            components.append(component)

    return components

header = "examples/data/spark/cache/tpch_100"
tabular_file = "query_level_cache_data.pkl"
struct_file = "struct_cache.pkl"
tabular_data = PickleUtils.load(header, tabular_file)
struct_data = PickleUtils.load(header, struct_file)
struct_dgl_dict = struct_data["struct_dgl_dict"]
global_ops = sorted(list(set.union(*[set(v.get_nnames()) for k, v in struct_dgl_dict.items()])))
q_signs = {kv["sql_struct_id"]: kv["q_sign"] for kv in struct_data["struct_dict"].values()}

stages_op_dgl_dict, query_stages_dgl_dict = {}, {}
stage2op_map_dict, stage_dep_dict = {}, {}
for k, v in struct_dgl_dict.items():
    struct = v.p1
    nids = struct.nids
    node_id2name = struct.node_id2name
    num_nodes = len(nids)
    from_ids_new, to_ids_new = [], []
    stage_node_from, stage_node_to = [], []
    from_exchange = []
    # 1st path, let us push the dependency like op -> exchange to the stage-level
    for from_id, to_id in zip(struct.from_ids, struct.to_ids):
        if is_split(node_id2name[to_id]):
            stage_node_from.append(from_id)
            stage_node_to.append(to_id)
        else:
            from_ids_new.append(from_id)
            to_ids_new.append(to_id)
    # 2nd path, let us folk the dependencies like ex -> op1 and ex -> op2 by duplicate exchange to two separate nodes
    stage_from_to_add, stage_to_to_add = [], []
    for i in range(len(from_ids_new)):
        from_id = from_ids_new[i]
        if is_split(node_id2name[from_id]) and from_id in from_ids_new[:i]:
            from_id_new = num_nodes
            node_id2name[from_id_new] = node_id2name[from_id]
            num_nodes += 1
            from_ids_new[i] = from_id_new
            # print(q_signs[k])
            assert from_id in stage_node_to
            n_node = 0
            for j in range(len(stage_node_from)):
                assert n_node <= 1
                stage_to = stage_node_to[j]
                if stage_to == from_id:
                    stage_from_to_add.append(from_id_new)
                    stage_to_to_add.append(stage_to)

    stage_node_from += stage_from_to_add
    stage_node_to += stage_to_to_add
    assert len(stage_node_from) == len(stage_node_to)

    g = dgl.graph((from_ids_new, to_ids_new))
    g.ndata["op_gid"] = th.LongTensor([global_ops.index(node_id2name[nid]) for nid in range(num_nodes)])
    G = dgl.to_networkx(g)
    plot_nx_graph(G, node_id2name, dir_name="subqueries", title=f"{k}-{q_signs[k]}")

    stages = [list(x) for x in list(nx.connected_components(G.to_undirected()))]
    stage_ids = list(range(len(stages)))
    from_ids_stage, to_ids_stage = [], []
    for stage_from, stage_to in zip(stage_node_from, stage_node_to):
        stage_id_from = -1
        stage_id_to = -1
        for stage_id, stage in enumerate(stages):
            if stage_from in stage:
                assert stage_id_from == -1
                stage_id_from = stage_id
            if stage_to in stage:
                assert stage_id_to == -1
                stage_id_to = stage_id
        assert stage_id_from >= 0 and stage_id_to >= 0
        from_ids_stage.append(stage_id_from)
        to_ids_stage.append(stage_id_to)

    query_stage_g = dgl.graph((from_ids_stage, to_ids_stage))
    query_stage_G = dgl.to_networkx(query_stage_g)
    node_id2stage = {stage_id: f"stage_{stage_id}" for stage_id in stage_ids}
    plot_nx_graph(query_stage_G, node_id2stage, dir_name="subqueries", title=f"{k}-{q_signs[k]}-query-stage")

    stages_op_dgl_dict[k] = [dgl.node_subgraph(g, stage) for stage in stages]
    query_stages_dgl_dict[k] = query_stage_g
    stage2op_map_dict[k] = stages
    stage_dep_dict[k] = (from_ids_stage, to_ids_stage)
    print(q_signs[k], stages)

PickleUtils.save({
    "stages_op_dgl_dict": stages_op_dgl_dict,
    "query_stages_dgl_dict": query_stages_dgl_dict,
    "stage2op_map_dict": stage2op_map_dict,
    "stage_dep_dict": stage_dep_dict
}, header, "struct_cache_extended.pkl", overwrite=True)
# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#            Arnab SINHA <arnab dot sinha at polytechnique dot edu>
# Description: Creates the DAG of the sql query plan and subplans for each stage
#
# Created at 11/9/22
import collections
import os, sys
from urllib.request import Request, urlopen
import json, ssl, socket, re
import socks #pip install PySocks
import networkx as nx # brew install graphviz && pip install pydot==1.4.2
import dgl, torch as th # pip install dgl==0.9.1

global_stage_counter = 0
class Node():
    def __init__(self, node):
        self.nid = node["nodeId"]
        self.name = node["nodeName"],
        metrics = node["metrics"]
        self.involved_stages_from_sparkviz = self._get_stageIds(metrics)
        self.metrics = metrics

    def get_nid(self):
        return self.nid

    def get_name(self):
        return self.name

    def _get_stageIds(self, metrics):
        stage_ids = []
        for metric in metrics:
            res = re.search('\(stage (.*):', metric['value'])
            if res is not None:
                stage_id = int(float(res.group(1)))
                stage_ids.append(stage_id)
        stage_ids_unique = sorted([*set(stage_ids)])
        return stage_ids_unique

        def _set_stageIds(self, stage_ids: list(int)):
            self.involved_stages_from_sparkviz = stage_ids
class Edge():
    def __init__(self, edge_index, edge):
        self.eid = edge_index
        self.from_nid = edge["fromId"]
        self.to_nid = edge["toId"]


class QueryPlanTopology():
    def __init__(self, nodes: list[Node], edges: list[Edge]):
        self.nodes = nodes
        self.edges = edges
        self.node2sparkvizStageIds = {node.nid: node.involved_stages_from_sparkviz for node in nodes}

    def get_node(self, nodeId: int):
        for node in self.nodes:
            if node.nid == nodeId:
                return node

def get_dgl_graph(edges : list[Edge]):
    u_list, v_list = [], []
    for ed in edges:
        fromId = ed['fromId']
        u_list.append(fromId)
        toId = ed['toId']
        v_list.append(toId)
    u, v = th.tensor(u_list), th.tensor(v_list)
    dgl_graph = dgl.graph((u, v))
    print(f'dgl {dgl_graph}')
    return dgl_graph


def get_topdowntree_root_leaves(edges: list[Edge]):
    tree = {}
    nodes_set = set()
    #print(edges)
    for ed in edges:
        childId = ed['fromId']
        parentId = ed['toId']
        tree.setdefault(parentId, []).append(childId)
        nodes_set.add(parentId)
        nodes_set.add(childId)

    root_node = nodes_set.copy()
    child_nodes = nodes_set.copy()

    # Removing child nodes from the set and we are left with the root
    for ed in edges:
        childId = ed['fromId']
        parentId = ed['toId']
        if childId in root_node:
            root_node.remove(childId)
        if parentId in child_nodes:
            child_nodes.remove(parentId)

    if len(root_node) > 1:
        print(f'Tree contains multiple root nodes: {root_node}')
    rootNode = root_node.pop()
    print(f'Tree {tree}')
    print(f'Root node is {rootNode}')
    print(f'Child nodes are {child_nodes}')
    return tree, rootNode, list(child_nodes)


# def get_leaves_rev_edges(edges: list[Edge]):
#     tree = [[] for i in range(1005)]
#     leaves = []
#     def dfs(curr_node, parent):
#         flag = 1
#         # Iterating the children of current curr_node
#         for ir in tree[curr_node]:
#             # There is at least a child
#             # of the current curr_node
#             if (ir != parent):
#                 flag = 0
#                 dfs(ir, curr_node)
#         # Current curr_node is connected to only
#         # its parent i.e. it is a leaf curr_node
#         if (flag == 1):
#             #print(curr_node, end=" ")
#             leaves.append(curr_node)
#
#     def reverse_edges(edges: list[Edge]):
#         reverse_edges = []
#         for ed in edges:
#             frmId = ed['fromId']
#             toId = ed['toId']
#             reverse_edges.append((toId, frmId))
#         return reverse_edges
#
#     rev_edges = reverse_edges(edges)
#     cnt = len(rev_edges)
#     # Number of nodes
#     curr_node = cnt + 1
#     # Create the tree
#     for i in range(cnt):
#         tree[rev_edges[i][0]].append(rev_edges[i][1])
#         tree[rev_edges[i][1]].append(rev_edges[i][0])
#     # Function call
#     dfs(1, 0)
#     return leaves, rev_edges


def get_node_toId(node_fromId, edges: list[tuple]):
    for tup in edges:
        if(tup['fromId']==node_fromId):
            return tup['toId']

def get_node_name(nid:int, nodes:list[Node]):
    name = None
    for node in nodes:
        if(node.nid==nid):
            name = ''.join(node.name)
    return name

def get_node(nid:int, nodes:list[Node]):
    n = None
    for node in nodes:
        if(node.nid==nid):
            n = node
    return n


def dict_unique_values(dup_dict: dict):
    for key, values_list in dup_dict.items():
        dup_dict[key] = sorted([*set(values_list)])
    return dup_dict


def bfs_rec(full_plan: QueryPlanTopology, current_stage: int, stage: dict, stage_dependency: list, topdown_tree: dict, curr_node=None):
    nodes = full_plan.nodes
    edges = full_plan.edges
    global global_stage_counter

    if curr_node == None:
        return
    else:
        stage.setdefault(current_stage, []).append(curr_node)
    exchange_node_flag = False
    if ((curr_node != None) and (('exchange' in (get_node_name(curr_node, nodes)).lower()) or ('subquery' in (get_node_name(curr_node, nodes)).lower()))):# exchange and subquery related operators considered exchange of stages
        exchange_node_flag = True
    if curr_node in topdown_tree.keys():
        for child_node in topdown_tree[curr_node]:
            if exchange_node_flag:
                save_current_stage = current_stage
                current_stage = global_stage_counter = global_stage_counter + 1
                stage.setdefault(current_stage, []).append(curr_node) # the operator demarcating stage change also added to the children nodes belonging in the next stage
                stage_dependency.append((current_stage, save_current_stage))
            bfs_rec(full_plan, current_stage, stage, stage_dependency, topdown_tree, child_node)
    else:#i.e. curr_node is the leaf curr_node
        bfs_rec(full_plan, current_stage, stage, stage_dependency, topdown_tree, None)


def get_sub_sqls_using_topdown_tree(full_plan: QueryPlanTopology):
    nodes = full_plan.nodes
    edges = full_plan.edges
    node2stageIds = full_plan.node2sparkvizStageIds
    # todo: Arnab, return a dict {stageId: [QueryPlanTopology]}, each with a subSQL plan.
    #leaves, rev_edges = get_leaves_rev_edges(edges)
    topdown_tree, root, leaves = get_topdowntree_root_leaves(edges)

    stage = {}
    stage_dependency = []
    global global_stage_counter
    global_stage_counter = 0
    current_stage = global_stage_counter
    bfs_rec(full_plan, current_stage, stage, stage_dependency, topdown_tree, root)

    return stage, stage_dependency


def get_subsql_plans(full_plan: QueryPlanTopology, stage: dict):
    # Creating the subsql_plans{stageId: [SubQueryPlanTopology]}
    subsql_plans = {}
    for stg_id, n_ids in stage.items():
        # stg = stage1[0]
        # n_ids = stage_to_nodeids[stg]
    #for stg, n_ids in stage_to_nodeids.items():
        ndes = list()
        for n_id in n_ids:
            ndes.append(full_plan.get_node(n_id))
        edgs = list()
        for edge in full_plan.edges:
            if edge['fromId'] in n_ids and edge['toId'] in n_ids:
                edgs.append(edge)
        sub_plan_topo = QueryPlanTopology(ndes, edgs)
        subsql_plans[stg_id] = sub_plan_topo
    return subsql_plans


def get_sub_sqls(full_plan: QueryPlanTopology):
    nodes = full_plan.nodes
    edges = full_plan.edges
    node2stageIds = full_plan.node2sparkvizStageIds
    # todo: Arnab, return a dict {stageId: [QueryPlanTopology]}, each with a subSQL plan.
    #leaves, rev_edges = get_leaves_rev_edges(edges)
    tree, root, leaves = get_topdowntree_root_leaves(edges)

    subsql_plans = {}
    stage_to_nodeids = {}

    for leaf in leaves:
        traverse_start_node = leaf
        traverse_end_node = get_node_toId(leaf,edges)
        subplan_start_node = get_node(traverse_start_node, nodes)
        subplan_start_node_stages = subplan_start_node.involved_stages_from_sparkviz
        while (get_node_name(traverse_start_node, nodes)):
            if subplan_start_node_stages:
                for stageId in subplan_start_node_stages:
                         if stageId not in stage_to_nodeids.keys():
                             stage_to_nodeids[stageId] = [traverse_start_node]
                         else:
                             stage_to_nodeids[stageId].append(traverse_start_node)
            traverse_start_node = traverse_end_node
            traverse_end_node = get_node_toId(traverse_start_node,edges)
            if ((traverse_start_node != None) and 'exchange' in (get_node_name(traverse_start_node, nodes)).lower()):
                #Add 'exchange' curr_node to previous stage before traversing to next stage
                if subplan_start_node_stages:
                    for stageId in subplan_start_node_stages:
                        if stageId not in stage_to_nodeids.keys():
                            stage_to_nodeids[stageId] = [traverse_start_node]
                        else:
                            stage_to_nodeids[stageId].append(traverse_start_node)

                subplan_start_node = get_node(traverse_end_node, nodes)
                subplan_start_node_stages = subplan_start_node.involved_stages_from_sparkviz
    stage_to_nodeids = dict_unique_values(stage_to_nodeids)
    # return stage_to_nodeids

    stageids_to_stages = get_queryStages(stage_to_nodeids)

    # Creating the subsql_plans{stageId: [SubQueryPlanTopology]}
    for stg_id, stage in stageids_to_stages.items():
        stg = stage[0]
        n_ids = stage_to_nodeids[stg]
    #for stg, n_ids in stage_to_nodeids.items():
        ndes = list()
        for n_id in n_ids:
            ndes.append(full_plan.get_node(n_id))
        edgs = list()
        for edge in full_plan.edges:
            if edge['fromId'] in n_ids and edge['toId'] in n_ids:
                edgs.append(edge)
        sub_plan_topo = QueryPlanTopology(ndes, edgs)
        subsql_plans[stg_id] = sub_plan_topo

    # # Creating the subsql_plans{stage: [SubQueryPlanTopology]}
    # for stg, n_ids in stage_to_nodeids.items():
    #     ndes = list()
    #     for n_id in n_ids:
    #         ndes.append(full_plan.get_node(n_id))
    #     edgs = list()
    #     for edge in full_plan.edges:
    #         if edge['fromId'] in n_ids and edge['toId'] in n_ids:
    #             edgs.append(edge)
    #     sub_plan_topo = QueryPlanTopology(ndes, edgs)
    #     subsql_plans[stg] = sub_plan_topo
    return stageids_to_stages, stage_to_nodeids, subsql_plans


def get_queryStages(stage_to_nodeids: dict):
    stageids_to_stages = {}
    for index_outer, key_outer in enumerate(stage_to_nodeids):
        stages = list()
        stages.append(key_outer)
        for index_inner, key_inner in enumerate(stage_to_nodeids):
            if index_inner > index_outer:
                #print(index_outer,index_inner, "===", key_outer, key_inner)
                if collections.Counter(stage_to_nodeids[key_outer]) == collections.Counter(stage_to_nodeids[key_inner]):
                    #print(f'Stages {key_outer} and {key_inner} are same')
                    stages.append(key_inner)
        stageids_to_stages[index_outer] = stages
    #print(stageids_to_stages)
    return stageids_to_stages

def format_edges(plan: QueryPlanTopology):
    plan_edges = plan.edges
    plan_nodes = {}
    for n in plan.nodes:
        plan_nodes[n.nid] = n.name[0]
    edges_tuples = []
    for ed in plan_edges:
        frmId = ed['fromId']
        toId = ed['toId']
        fromIdNode = str(frmId) + " - " + plan_nodes[frmId]
        toIdNode = str(toId) + " - " + plan_nodes[toId]
        edges_tuples.append((fromIdNode,toIdNode))
    return edges_tuples

def topo_visualization(QueryPlanTopology, dir_name, title):
    # todo: Arnab, draw the query plan in a pdf, each query operator shows its curr_node.nid and curr_node.name
    list_edge_tuples = format_edges(QueryPlanTopology)
    dependency_visualization(list_edge_tuples, dir_name, title)

def dependency_visualization(dependencies: list, dir_name, title):
    ###
    ## Older version
    # G = nx.DiGraph()
    # G.add_edges_from(list_edge_tuples)
    # pos = nx.spring_layout(G, seed=50, k=0.60, scale=None, iterations=350)
    # nx.draw(G, pos, ax=None, with_labels=True, font_size=4, node_size=900, node_color='lightgreen', edge_color='black')
    # #plt.show()
    # plt.savefig(title + ".pdf", format="PDF", dpi=2000)
    # plt.clf()
    ###
    # brew install graphviz && pip install pydot==1.4.2
    G = nx.DiGraph()
    G.add_edges_from(dependencies)
    p = nx.drawing.nx_pydot.to_pydot(G)
    dir_to_save = 'application_graphs/' + dir_name
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    p.write_png(dir_to_save + '/' + title + '.png')

def get_json_data(url):
    """
    Receive the content of ``url``, parse and return as JSON dictionary.
    """
    response = urlopen(url)
    #data = response.read()
    data = response.read().decode("utf-8")
    return json.loads(data)


def get_json_data_using_proxy(url):
    """
    Receive the content of ``url`` using proxy, parse and return as JSON dictionary.
    """
    IP_ADDR = 'localhost'
    PORT = 2000

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    request = Request(url)
    socks.set_default_proxy(socks.SOCKS5, IP_ADDR, PORT)
    socket.socket = socks.socksocket
    response = urlopen(request, context=ctx)
    data = response.read().decode("utf-8")
    #print(data)
    return json.loads(data)


#print(sys.getrecursionlimit())
sys.setrecursionlimit(5000)
#print(sys.getrecursionlimit())
#Example query:
#url = "http://node13-opa:18088/api/v1/applications/application_1666993404824_4998/sql"
#url = "http://node13-opa:18088/api/v1/applications/application_1666973014171_0012/sql"

urls = {}
# Test url
#urls['application_1666973014171_0015'] = "http://node13-opa:18088/api/v1/applications/application_1666973014171_0015/sql"

# Queries without AQE:
# application_1666973014171_0012 (Q1) - application_1666973014171_0075 (Q22), step by 3
for i in range(12,76,3):
    qrun = str(i).zfill(4)
    #print(qrun)
    app_id = f'application_1666973014171_{qrun}'
    urls[app_id] = f'http://node13-opa:18088/api/v1/applications/{app_id}/sql'

# Queries with AQE (skip query 9)
# application_1666973014171_0078 (Q1) - application_1666973014171_0141 (Q22), step by 3.
for i in range(78,142,3):
    if i in [102,103,104]:#skipping query 9
        continue
    qrun = str(i).zfill(4)
    #print(qrun)
    app_id = f'application_1666973014171_{qrun}'
    urls[app_id] = f'http://node13-opa:18088/api/v1/applications/{app_id}/sql'

for app_id, url in urls.items():
    print(f'Processing {app_id}')
    # sql_id = 0 is doing `use TPCH_100`, skip.
    # sql_id = 1 -> analyses
    data = get_json_data_using_proxy(url) # get a json file as a dictionary from the url

    sql_data = data[1]
    assert sql_data["id"] == 1

    # WARN: use the nodeId and edgeId instead of the index.
    nodes = [Node(n) for n in sql_data["nodes"]]
    edges = sql_data["edges"]
    # topdowntree, root, leaves = get_topdowntree_root_leaves(edges)
    # # leaves = get_leaves_rev_edges(edges)

    full_plan = QueryPlanTopology(nodes, edges)
    stage_to_nodes, stage_dependencies = get_sub_sqls_using_topdown_tree(full_plan)
    print(f"Stage allocation to subqueries : {stage_to_nodes}")
    print(f"Stage dependencies : {stage_dependencies}")
    stage2plan = get_subsql_plans(full_plan, stage_to_nodes)

    # stageids_to_stages, stage_to_nodeids, stage2plan = get_sub_sqls(full_plan)
    # print(f'Stageids to stages => {stageids_to_stages}')
    # print(f'Stage to nodeIds => {stage_to_nodeids}')
    #print(f'Stage to plan => {stage2plan}')
    print(f'Saving visualization for full plan')
    topo_visualization(full_plan, app_id, title="full_plan")
    full_plan_dgl = get_dgl_graph(edges)
    dgl_subplans = {}
    for stageId, subplan in stage2plan.items():
        print(f'Saving subplan visualization for stage {stageId}')
        topo_visualization(subplan, app_id, title=f"stage_{stageId}")
        dgl_subplan = get_dgl_graph(subplan.edges)
        dgl_subplans[stageId] = dgl_subplan

    # todo: [later], how to get the dependency among stages.
    print(f'Saving stage dependency visualization')
    dependency_visualization(stage_dependencies, app_id, title=f"stage_dependencies")

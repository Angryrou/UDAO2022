# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#            Arnab SINHA <arnab dot sinha at polytechnique dot edu>
# Description: Creates the DAG of the sql query plan and subplans for each stage
#
# Created at 11/9/22

from urllib.request import Request, urlopen
import json, ssl, socket, re
import socks #pip install PySocks
import networkx as nx # brew install graphviz && pip install pydot==1.4.2

class Node():
    def __init__(self, node):
        self.nid = node["nodeId"]
        self.name = node["nodeName"],
        metrics = node["metrics"]
        self.involved_stages = self._get_stageIds(metrics)
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
            self.involved_stages = stage_ids
class Edge():
    def __init__(self, edge_index, edge):
        self.eid = edge_index
        self.from_nid = edge["fromId"]
        self.to_nid = edge["toId"]


class QueryPlanTopology():
    def __init__(self, nodes: list[Node], edges: list[Edge]):
        self.nodes = nodes
        self.edges = edges
        self.node2stageIds = {node.nid: node.involved_stages for node in nodes}

    def get_node(self, nodeId: int):
        for node in self.nodes:
            if node.nid == nodeId:
                return node

def get_leaves_rev_edges(edges: list[Edge]):
    tree = [[] for i in range(1005)]
    leaves = []
    def dfs(node, parent):
        flag = 1
        # Iterating the children of current node
        for ir in tree[node]:
            # There is at least a child
            # of the current node
            if (ir != parent):
                flag = 0
                dfs(ir, node)
        # Current node is connected to only
        # its parent i.e. it is a leaf node
        if (flag == 1):
            #print(node, end=" ")
            leaves.append(node)

    def reverse_edges(edges: list[Edge]):
        reverse_edges = []
        for ed in edges:
            frmId = ed['fromId']
            toId = ed['toId']
            reverse_edges.append((toId, frmId))
        return reverse_edges

    rev_edges = reverse_edges(edges)
    cnt = len(rev_edges)
    # Number of nodes
    node = cnt + 1
    # Create the tree
    for i in range(cnt):
        tree[rev_edges[i][0]].append(rev_edges[i][1])
        tree[rev_edges[i][1]].append(rev_edges[i][0])
    # Function call
    dfs(1, 0)
    return leaves, rev_edges


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


def get_sub_sqls(full_plan: QueryPlanTopology):
    nodes = full_plan.nodes
    edges = full_plan.edges
    node2stageIds = full_plan.node2stageIds
    # todo: Arnab, return a dict {stageId: [QueryPlanTopology]}, each with a subSQL plan.
    leaves, rev_edges = get_leaves_rev_edges(edges)

    subsql_plans = {}
    stage_to_nodeids = {}

    for leaf in leaves:
        traverse_start_node = leaf
        traverse_end_node = get_node_toId(leaf,edges)
        subplan_start_node = get_node(traverse_start_node, nodes)
        subplan_start_node_stages = subplan_start_node.involved_stages
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
                #Add 'exchange' node to previous stage before traversing to next stage
                if subplan_start_node_stages:
                    for stageId in subplan_start_node_stages:
                        if stageId not in stage_to_nodeids.keys():
                            stage_to_nodeids[stageId] = [traverse_start_node]
                        else:
                            stage_to_nodeids[stageId].append(traverse_start_node)

                subplan_start_node = get_node(traverse_end_node, nodes)
                subplan_start_node_stages = subplan_start_node.involved_stages
    stage_to_nodeids = dict_unique_values(stage_to_nodeids)
    print(f'Stage to nodeIds => {stage_to_nodeids}')
    # return stage_to_nodeids

    # Creating the subsql_plans{stageId: [SubQueryPlanTopology]}
    for stg, n_ids in stage_to_nodeids.items():
        ndes = list()
        for n_id in n_ids:
            ndes.append(full_plan.get_node(n_id))
        edgs = list()
        for edge in full_plan.edges:
            if edge['fromId'] in n_ids and edge['toId'] in n_ids:
                edgs.append(edge)
        sub_plan_topo = QueryPlanTopology(ndes, edgs)
        subsql_plans[stg] = sub_plan_topo
    return subsql_plans


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

def visualization(QueryPlanTopology, title):
    # todo: Arnab, draw the query plan in a pdf, each query operator shows its node.nid and node.name
    list_edge_tuples = format_edges(QueryPlanTopology)
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
    G.add_edges_from(list_edge_tuples)
    p = nx.drawing.nx_pydot.to_pydot(G)
    p.write_png(title + '.png')
    ...


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
    PORT = 9000

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

#Example query:
url = "http://node13-opa:18088/api/v1/applications/application_1666993404824_4998/sql"

# sql_id = 0 is doing `use TPCH_100`, skip.
# sql_id = 1 -> analyses
data = get_json_data_using_proxy(url) # get a json file as a dictionary from the url

sql_data = data[1]
assert sql_data["id"] == 1

# WARN: use the nodeId and edgeId instead of the index.
nodes = [Node(n) for n in sql_data["nodes"]]
edges = sql_data["edges"]
leaves = get_leaves_rev_edges(edges)

full_plan = QueryPlanTopology(nodes, edges)
stage2plan = get_sub_sqls(full_plan)
print(f'Saving visualization for full plan')
visualization(full_plan, title="full_plan")
for stageId, subplan in stage2plan.items():
    print(f'Saving subplan visualization for stage {stageId}')
    visualization(subplan, title=f"stage_{stageId}")

# todo: [later], how to get the dependency among stages.
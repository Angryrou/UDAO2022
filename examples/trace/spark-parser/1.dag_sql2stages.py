# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 11/9/22

class Node():
    def __init__(self, node):
        self.nid = node["nodeId"]
        self.name = node["nodeName"],
        metrics = node["metrics"]
        self.involved_stages = self._get_stageIds(metrics)
        self.metrics = metrics

    def _get_stageIds(self, metrics):
        # todo: Arnab, return the stagesIds.
        ...

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

def get_sub_sqls(full_plan: QueryPlanTopology):
    nodes = full_plan.nodes
    edges = full_plan.edges
    node2stageIds = full_plan.node2stageIds
    # todo: Arnab, return a dict {stageId: [QueryPlanTopology]}, each with a subSQL plan.
    ...


def visualization(QueryPlanTopology, title):
    # todo: Arnab, draw the query plan in a pdf, each query operator shows its node.nid and node.name
    ...


url = "http://node13-opa:18088/api/v1/applications/application_1666993404824_4998/sql"
# sql_id = 0 is doing `use TPCH_100`, skip.
# sql_id = 1 -> analyses
data = ... # get a json file as a dictionary from the url

sql_data = data[1]
assert sql_data["id"] == 1

# WARN: use the nodeId and edgeId instead of the index.
nodes = [Node(n) for n in sql_data["nodes"]]
edges = sql_data["edges"]

full_plan = QueryPlanTopology(nodes, edges)
stage2plan = get_sub_sqls(full_plan)
visualization(full_plan, title="full_plan")
for stageId, subplan in stage2plan.items():
    visualization(subplan, title=f"stage_{stageId}")

# todo: [later], how to get the dependency among stages.
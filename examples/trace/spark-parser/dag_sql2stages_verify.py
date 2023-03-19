# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#            Arnab SINHA <arnab dot sinha at polytechnique dot edu>
# Description: Constructs the DAG of the sql query plan and its stages
#
# Created at 23/01/23
import collections
import os, sys
from urllib.request import Request, urlopen
import json, ssl, socket, re
import socks #pip install PySocks
import networkx as nx # brew install graphviz && pip install pydot==1.4.2
import dgl, torch as th # pip install dgl==0.9.1

import dag_sql2stages as viz
import math
import pandas as pd
from utils.common import PickleUtils

def read_csv(file: str):
    df = pd.read_csv(file, sep="\u0001", index_col=[0, 1])
    return df
    #return df["id"], df["name"], df["q_sign"], df["knob_sign"], df["nodes"], df["edges"]

def sparkvizStageIds_revDict(sparkvizStageIds : dict):
    rev_dict = {}
    stages_arr = []
    for key, value in sparkvizStageIds.items():
        #print(key, value)
        if len(value):
            for stage in value:
                rev_dict.setdefault(stage, []).append(key)
                stages_arr.append(stage)
                #rev_dict[stage] = key
    stages_arr = list(set(stages_arr))
    return rev_dict, stages_arr


def map_qstage_sstage(qstage_dict: dict, sstage_dict: dict):
    map_qstage_sstage_dict = {}
    for qstage, qnodes in qstage_dict.items():
        for sstage, snodes in sstage_dict.items():
            for qnode in qnodes:
                if qnode in snodes:
                    map_qstage_sstage_dict.setdefault(qstage,[]).append(sstage)
    for qstage, sstages in map_qstage_sstage_dict.items():
        map_qstage_sstage_dict[qstage] = list(set(sstages))
    return map_qstage_sstage_dict


def get_stage_info(filename: str):
    df_stages = pd.read_csv(filename, sep="\u0001", index_col=[0])
    completed_stages_dict = {}
    for index, row in df_stages.iterrows():
        app = row["id"]
        stage = row["stage_id"]
        if math.isnan(row["err"]):
            completed_stages_dict.setdefault(app,[]).append(stage)
    return completed_stages_dict

def check_multiple_operators(QStage_dict: dict, nodes: [viz.Node]):
    multi_ops = False
    for key, value in QStage_dict.items():
        stage_multi_ops_cnt = 0
        for curr_node in value:
            curr_node_name = (viz.get_node_name(curr_node, nodes)).lower()
            if re.search('|'.join(['join']), curr_node_name):
            # if curr_node_name in ['sort', 'hash']:
                stage_multi_ops_cnt = stage_multi_ops_cnt + 1
        if stage_multi_ops_cnt > 1:
            #print(f'#operator {stage_multi_ops_cnt} Multi-op stage {key} : {value}')
            print(f' : stage {key} => {stage_multi_ops_cnt} joins\n {value}')
        multi_ops = multi_ops or (True if stage_multi_ops_cnt>1 else False)
    return multi_ops


if __name__ == "__main__":
    # app_id, name, q_sign, knob_sign, nodes, edges = read_csv("tpcds_repr_split_verify.csv")
    # input from csv files instead of fetching online
    # df = read_csv("dag_sql2stages_verify_input/tpcds_repr_split_verify.csv")
    # df = read_csv("dag_sql2stages_verify_input/tpch_repr_split_verify.csv")
    df = read_csv("dag_sql2stages_verify_input/tpch_repr_split_verify-test.csv")
    # TPCH completed stages info provided in csv for verification.
    completed_stages = get_stage_info("dag_sql2stages_verify_input/tpch_repr_stage_status.csv")

    QStage_dep_pkl = {}

    for index, row in df.iterrows():
        # print(row["Name"], row["Age"])
        app_id = row["id"]
        nodes_json = json.loads(row["nodes"])
        edges = json.loads(row["edges"])

        print(f'Processing {app_id}')
        # # sql_id = 0 is doing `use TPCH_100`, skip.
        # # sql_id = 1 -> analyses
        # data = get_json_data_using_proxy(url) # get a json file as a dictionary from the url
        #
        # sql_data = data[1]
        # assert sql_data["id"] == 1

        # # WARN: use the nodeId and edgeId instead of the index.
        nodes = [viz.Node(n) for n in nodes_json]
        nodes_dict = {n.nid: n.name for n in nodes}
        print(nodes_dict)
        # edges = sql_data["edges"]
        # # topdowntree, root, leaves = get_topdowntree_root_leaves(edges)
        # # # leaves = get_leaves_rev_edges(edges)

        full_plan = viz.QueryPlanTopology(nodes, edges)
        QStage_to_nodes, QStage_dependencies = viz.get_sub_sqls_using_topdown_tree(full_plan)
        print(f"Query Stage allocation to nodes : {QStage_to_nodes}")
        print(f"Query Stage dependencies : {QStage_dependencies}")
        print(f'{app_id} app')
        if check_multiple_operators(QStage_to_nodes, nodes):
            print(f'Note: Atleast one stage has multiple operators in app {app_id}')
        rev_dict_SStageIds, stages_arr = sparkvizStageIds_revDict(full_plan.node2sparkvizStageIds) # WholeStageCodegen (6)
        print(f'Spark Stage allocation in Spark Viz : {rev_dict_SStageIds}')
        print(f'mapping Query Stage => Spark Stage {map_qstage_sstage(QStage_to_nodes,rev_dict_SStageIds)}')
        stage2plan = viz.get_stage_plans(full_plan, QStage_to_nodes)

        ## Verifying if the known query stage ids from the spark api is a subset of the completed query stages
        if (set(stages_arr)).issubset(set(completed_stages[app_id])):
            print(f'Stages completed : {stages_arr}')
        else:
            print(f'Error completing stages {app_id}')

        # stageids_to_stages, stage_to_nodeids, stage2plan = get_sub_sqls(full_plan)
        # print(f'Stageids to stages => {stageids_to_stages}')
        # print(f'Stage to nodeIds => {stage_to_nodeids}')
        #print(f'Stage to plan => {stage2plan}')
        print(f'Saving visualization for full plan')
        viz.topo_visualization(full_plan, app_id, title="full_plan")
        full_plan_dgl = viz.get_dgl_graph(edges)
        dgl_querystages = {}
        for stageId, querystage in stage2plan.items():
            print(f'Saving visualization for Query Stage {stageId}')
            viz.topo_visualization(querystage, app_id, title=f"stage_{stageId}")
            dgl_querystage = viz.get_dgl_graph(querystage.edges)
            dgl_querystages[stageId] = dgl_querystage

        # getting the dependency among stages.
        print(f'Saving query stage dependency visualization')
        viz.dependency_visualization(QStage_dependencies, app_id, title=f"query_stage_dependencies-{app_id}")
        QStage_dep_pkl[app_id] = QStage_dependencies

    PickleUtils.save(QStage_dep_pkl,f"QStage_dependencies_dump",f"QStage_dependencies_tpch_apps.pkl")
    print(f"Pickle file saved")
        #sys.exit(0)
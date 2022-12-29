# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 23/12/2022
import json, os
import numpy as np
import pandas as pd

import dgl

from utils.common import JsonUtils, ParquetUtils


def get_csvs(templates, header, cache_header, samplings=["lhs", "bo"]):
    fname = f"csvs.parquet"
    try:
        df = ParquetUtils.parquet_read(cache_header, fname)
        print(f"found cached {fname} at {cache_header}")
    except:
        df_dict = {}
        for t in templates:
            df_t = []
            for sampling in samplings:
                file_path = f"{header}/{t}_{sampling}.csv"
                assert os.path.exists(file_path)
                df = pd.read_csv(file_path, sep="\u0001")
                df["sampling"] = sampling
                df_t.append(df)
            df = pd.concat(df_t).reset_index()
            df["sql_struct_sign"] = df.apply(lambda row: serialize_sql_structure(row), axis=1)
            df_dict[t] = df
            print(f"template {t} has {df.sql_struct_sign.unique().size} different structures")
        df = pd.concat(df_dict)
        structure_list = df["sql_struct_sign"].unique().tolist()
        df["sql_struct_id"] = df["sql_struct_sign"].apply(lambda x: structure_list.index(x))
        ParquetUtils.parquet_write(df, cache_header, fname)
        print(f"generated cached {fname} at {cache_header}")
    return df

def serialize_sql_structure(row):
    j_nodes, j_edges = json.loads(row["nodes"]), json.loads(row["edges"])
    nodeIds, nodeNames = JsonUtils.extract_json_list(j_nodes, ["nodeId", "nodeName"])
    nodeIds, nodeNames = zip(*sorted(zip(nodeIds, nodeNames)))
    nodeNames = [name.split()[0] for name in nodeNames]
    fromIds, toIds = JsonUtils.extract_json_list(j_edges, ["fromId", "toId"])
    fromIds, toIds = zip(*sorted(zip(fromIds, toIds)))
    sign = ",".join(str(x) for x in nodeIds) + ";" + ",".join(nodeNames) + ";" + \
           ",".join(str(x) for x in fromIds) + ";" + ",".join(str(x) for x in toIds)
    return sign


def get_sql_structure(s: str) -> dgl.graph:
    ...


class SqlStruct():

    def __init__(self, d: dict):
        self.id = d["sql_struct_id"]
        self.struct_sign = d["sql_struct_sign"]
        nodeIds, nodeNames, fromIds, toIds = self.struct_sign.split(";")
        node_id2name = {int(i): n for i, n in zip(nodeIds.split(","), nodeNames.split(","))}
        fromIds = [int(i) for i in fromIds.split(",")]
        toIds = [int(i) for i in toIds.split(",")]

        # we only consider the nodes that are used in the topology (omit nodes like WSCG)
        nids_old = sorted(list(set(fromIds) | set(toIds)))
        nids_new = range(len(nids_old))
        nids_new2old = {n: o for n, o in zip(nids_new, nids_old)}
        nids_old2new = {o: n for n, o in zip(nids_new, nids_old)}
        fromIds_new = [nids_old2new[i] for i in fromIds]
        toIds_new = [nids_old2new[i] for i in toIds]
        node_id2name_new = {nids_old2new[k]: v for k, v in node_id2name.items() if k in nids_old2new}

        self.nids = nids_new
        self.nnames = [node_id2name_new[k] for k, v in node_id2name_new.items()]
        self.from_ids = fromIds_new
        self.to_ids = toIds_new

        self.old = {
            "nids": nids_old,
            "from_ids": fromIds,
            "to_ids": toIds,
            "nids_new2old": nids_new2old
        }


# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 23/12/2022
import json, os, re, time, random

import networkx
import numpy as np
import pandas as pd
import torch as th
import dgl
import networkx as nx  # brew install graphviz && pip install pydot==1.4.2
from networkx.algorithms import isomorphism

from utils.common import JsonUtils, ParquetUtils, PickleUtils

from IPython.display import Image, display
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec


def get_tr_val_te_masks(df, groupby_col1, groupby_col2, frac_val_per_group, frac_te_per_group, seed):
    """return tr_mask, val_mask, te_mask"""
    random.seed(seed)
    np.random.seed(seed)
    n_rows = len(df)
    val_mask, te_mask = np.array([False] * n_rows), np.array([False] * n_rows)
    df = df.reset_index()
    val_index1 = df.groupby(groupby_col1).sample(frac=frac_val_per_group).index
    val_index2 = df.groupby(groupby_col1).sample(frac=frac_val_per_group, random_state=seed).index
    assert (val_index1 == val_index2).all()
    val_index = val_index1
    val_mask[val_index] = True
    frac_te_per_group = frac_te_per_group / (1 - frac_val_per_group)
    te_index = df[~val_mask].groupby(groupby_col2).sample(frac=frac_te_per_group).index
    te_mask[te_index] = True
    tr_mask = ~val_mask & ~te_mask
    return tr_mask, val_mask, te_mask


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
            df["sql_struct_sign"], df["input_mb"], df["input_records"] = \
                zip(*df.apply(lambda row: serialize_sql_structure(row), axis=1))
            df["input_mb_log"], df["input_records_log"] = np.log(df["input_mb"]), np.log(df["input_records"])
            df_dict[t] = df
            print(f"template {t} has {df.sql_struct_sign.unique().size} different structures")
        df = pd.concat(df_dict)
        structure_list = df["sql_struct_sign"].unique().tolist()
        df["sql_struct_id"] = df["sql_struct_sign"].apply(lambda x: structure_list.index(x))
        df["qid"] = df["q_sign"].apply(lambda x: int(x.split("-")[1]))
        for sid in range(len(structure_list)):
            matches = (df["sql_struct_id"] == sid)
            df.loc[matches, "sql_struct_svid"] = np.arange(sum(matches))
        ParquetUtils.parquet_write(df, cache_header, fname)
        print(f"generated cached {fname} at {cache_header}")
    return df


def get_csvs_stage(header, cache_header, samplings=["lhs", "bo"]):
    fname = f"csvs_stage.parquet"
    try:
        df = ParquetUtils.parquet_read(cache_header, fname)
        print(f"found cached {fname} at {cache_header}")
    except:
        df_s = []
        for sampling in samplings:
            file_path = f"{header}/{sampling}.csv"
            assert os.path.exists(file_path)
            df = pd.read_csv(file_path, sep="\u0001")
            df["sampling"] = sampling
            df_s.append(df)
        df = pd.concat(df_s).reset_index()
        ParquetUtils.parquet_write(df, cache_header, fname)
        print(f"generated cached {fname} at {cache_header}")
    return df


def get_csvs_tr_val_te(templates, header, cache_header, seed, samplings=["lhs", "bo"]):
    df = get_csvs(templates, header, cache_header, samplings)
    tr_mask, val_mask, te_mask = get_tr_val_te_masks(df=df, groupby_col1="template", groupby_col2="template",
                                                     frac_val_per_group=0.1, frac_te_per_group=0.1, seed=seed)
    df_tr, df_val, df_te = df[tr_mask], df[val_mask], df[te_mask]
    return df_tr, df_val, df_te


def serialize_sql_structure(row):
    j_nodes, j_edges = json.loads(row["nodes"]), json.loads(row["edges"])
    nodeIds, nodeNames = JsonUtils.extract_json_list(j_nodes, ["nodeId", "nodeName"])
    nodeIds, nodeNames = zip(*sorted(zip(nodeIds, nodeNames)))
    nodeNames = [name.split()[0] for name in nodeNames]
    fromIds, toIds = JsonUtils.extract_json_list(j_edges, ["fromId", "toId"])
    fromIds, toIds = zip(*sorted(zip(fromIds, toIds)))
    sign = ",".join(str(x) for x in nodeIds) + ";" + ",".join(nodeNames) + ";" + \
           ",".join(str(x) for x in fromIds) + ";" + ",".join(str(x) for x in toIds)
    input_mb, input_records = extract_query_input_size(j_nodes)
    return sign, input_mb, input_records


def extract_query_input_size(j_nodes):
    nids, nnames, nmetrics = JsonUtils.extract_json_list(j_nodes, ["nodeId", "nodeName", "metrics"])
    input_mb, input_records = 0, 0
    for nid, nname, nmetric in zip(nids, nnames, nmetrics):
        if nname.split()[0] == "Scan":
            for m in nmetric:
                if m["name"] == "size of files read":
                    input_mb = format_size(m["value"]) / 1024 / 1024
                if m["name"] == "number of output rows":  # for scan, n_input_records = n_output_records
                    input_records += float(m["value"].replace(",", ""))
    return input_mb, input_records


def nodes_old2new(from_ids, to_ids, node_id2name, reverse=False):
    nids_old = sorted(list(set(from_ids) | set(to_ids)), reverse=reverse)
    nids_new = list(range(len(nids_old)))
    nids_new2old = {n: o for n, o in zip(nids_new, nids_old)}
    nids_old2new = {o: n for n, o in zip(nids_new, nids_old)}
    from_ids_new = [nids_old2new[i] for i in from_ids]
    to_ids_new = [nids_old2new[i] for i in to_ids]
    node_id2name_new = {nids_old2new[k]: v for k, v in node_id2name.items() if k in nids_old2new}
    return nids_old, nids_new, nids_new2old, nids_old2new, from_ids_new, to_ids_new, node_id2name_new


def plot_nx_graph(G: networkx.DiGraph, node_id2name: dict, dir_name: str, title: str, prefix: bool = True,
                  color: str = None, fillcolor: str = None, jupyter: bool = False):
    p = nx.drawing.nx_pydot.to_pydot(G)
    for i, node in enumerate(p.get_nodes()):
        if prefix:
            node.set_label(f"{i}-{node_id2name[i]}")
        else:
            node.set_label(node_id2name[i])
        if color is not None:
            node.set("color", color)
            if fillcolor is not None:
                node.set("style", "filled")
                node.set("fillcolor", color)
    dir_to_save = 'application_graphs/' + dir_name
    os.makedirs(dir_to_save, exist_ok=True)
    p.write_png(dir_to_save + '/' + title + '.png')
    if jupyter:
        display(Image(dir_to_save + '/' + title + '.png'))


def plot_nx_graph_augment(G: networkx.DiGraph, node_id2name: dict, dir_name: str, title: str, nodes_desc: dict):
    node_id2name_new = {}
    for k, v in node_id2name.items():
        assert v.split()[0] == v, (k, v)
        if v == "Scan":
            tbl = nodes_desc[k].split("tpch_100.")[1].split()[0]
            v = f"Scan({tbl})"
        node_id2name_new[k] = v
    plot_nx_graph(G, node_id2name_new, dir_name, title)


def plot_dgl_graph(g: dgl.DGLGraph, node_id2name: dict, dir_name: str, title: str, prefix: bool = True,
                   color: str = None, fillcolor: str = None, jupyter: bool = False):
    G = dgl.to_networkx(g)
    plot_nx_graph(G, node_id2name, dir_name, title, prefix, color, fillcolor, jupyter)


def plot_timeline(sid, q_sign, analyze_dt, s_ids, s_starts, s_ends, q_end, save_to=None):
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = sns.color_palette("mako", len(s_ids))
    # sns.set_theme(style="ticks")
    ax.xaxis.grid(True)
    ax.set(xlabel=f"Relative Timestamps {sid}({q_sign})")
    ax.set(ylabel="")
    ax.set_yticks([])
    for i, (stage_id, sstart, send) in enumerate(zip(s_ids, s_starts, s_ends)):
        plt.plot([sstart, send], [i + 2, i + 2], color=colors[i], marker="|", linestyle="-",
                 label=f"stage_{stage_id}[{send-sstart:.1f}s]")
    plt.plot([0, analyze_dt], [1, 1], "b|-", label=f"analyze [{analyze_dt:.1f}s]")
    plt.plot([0, q_end], [0, 0], "r|-", label=f"query [{q_sign}, {q_end:.1f}s]]")
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        figpath = f"{save_to}/timeline_{sid}({q_sign}).pdf"
        fig.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
    plt.show()


def show_q(df_q, df_s, sid, q_sign, save_to=None):
    target_q = df_q[df_q.q_sign == q_sign]
    assert len(target_q) == 1
    target_s = df_s[df_s.id == target_q.index[0]].sort_values(["first_task_launched_time", "stage_id"])
    assert len(target_s) >= 1

    q_start, q_lat = target_q.start_timestamp[0], target_q.latency[0]
    q_end = q_start + q_lat
    s_ids = target_s.stage_id.to_numpy()
    s_starts = target_s.first_task_launched_time.to_numpy()
    s_lats = target_s.stage_latency.to_numpy()
    s_ends = s_starts + s_lats

    offset = q_start
    q_end = q_lat
    print(f"query: 0 - {q_lat:.3f}s")
    s_starts = s_starts - offset
    s_ends = s_ends - offset
    print(f"stage: {s_starts.min():.3f} - {s_ends.max():.3f}s")

    analyze_dt = s_starts[np.where(s_starts > 0)[0]].min()
    print(f"analyze time: {analyze_dt}")
    plot_timeline(sid, q_sign, analyze_dt, s_ids, s_starts, s_ends, q_end, save_to)


def list_strip(inputs):
    return [i.strip() for i in inputs if i.strip() not in ("", "\"", "\\n\\n")]


def get_op_id(row):
    return int(row.split("(")[-1].split(")")[0])


def get_tree_structure_internal(ids, rows):
    from_ids, to_ids = [], []
    pre_id = ids[0]
    ranks = [(len(r) - len(r.lstrip())) / 3 for r in rows]
    ranks[0] = -1
    threads = {}
    for i, (id, row, rank) in enumerate(zip(ids, rows, ranks)):
        if i == 0:
            continue
        if row.lstrip()[0] == ":":
            if rank in threads:
                threads[rank]["ids"].append(id)
                threads[rank]["rows"].append(row.lstrip()[3:])
            else:
                threads[rank] = {
                    "ids": [id],
                    "rows": [row.lstrip()[3:]]
                }
        elif row.lstrip()[0] == "+":
            from_ids.append(id)
            to_ids.append(pre_id)
            pre_id = id
        else:
            raise ValueError(i, (id, row, rank))

    for rank, v in threads.items():
        sub_ids, sub_rows = v["ids"], v["rows"]
        from_ids.append(sub_ids[0])
        to_id_cands = [i for i, r, rk in zip(ids, rows, ranks) if rk == rank - 1 and r.lstrip()[0] != ":"]
        if len(to_id_cands) != 1:
            print(rank, v)
        to_ids.append(to_id_cands[0])
        sub_from_ids, sub_to_ids = get_tree_structure_internal(sub_ids, sub_rows)
        from_ids += sub_from_ids
        to_ids += sub_to_ids
    return from_ids, to_ids


def get_tree_structure(tree_str):
    rows = [s for s in tree_str.split("\\n") if s != ""]
    ids = [get_op_id(r) for r in rows]
    from_ids, to_ids = get_tree_structure_internal(ids, rows)
    return from_ids, to_ids


def get_node_details(details_str):
    node_short_dict = {}
    node_long_dict = {}
    for d in list_strip(details_str.split("\\n\\n")):
        ss = re.split(" |\\\\n", d)
        nid = int(ss[0][1:-1])
        node_short_dict[nid] = ss[1]
        node_long_dict[nid] = " ".join(ss[1:])
    return node_short_dict, node_long_dict


def connect_reused_exchange(from_ids, to_ids, node_id2name, node_details_dict):
    reused_k = []
    for k, v in node_id2name.items():
        if v == "ReusedExchange":
            ref_id = int(node_details_dict[k].split("Reuses operator id: ")[1].split("]")[0])
            from_ids = [x if x != k else ref_id for x in from_ids]
            to_ids = [x if x != k else ref_id for x in to_ids]
            reused_k.append(k)
    for k in reused_k:
        del node_id2name[k], node_details_dict[k]
    return from_ids, to_ids, node_id2name, node_details_dict


def dependency_visualization(dependencies, dir_name, title):
    # from Arnab
    G = nx.DiGraph()
    G.add_edges_from(dependencies)
    p = nx.drawing.nx_pydot.to_pydot(G)
    dir_to_save = 'application_graphs/' + dir_name
    os.makedirs(dir_to_save, exist_ok=True)
    p.write_png(dir_to_save + '/' + title + '.png')


def construct_from_metrics(struct_sign):
    nodeIds, nodeNames, fromIds, toIds = struct_sign.split(";")
    node_id2name = {int(i): n for i, n in zip(nodeIds.split(","), nodeNames.split(","))}
    fromIds = [int(i) for i in fromIds.split(",")]
    toIds = [int(i) for i in toIds.split(",")]
    return construct_internal(fromIds, toIds, node_id2name, reverse=False)


def construct_from_plan(desc):
    plans = list_strip(re.compile("={2,}").split(desc))
    assert plans[0] == "Physical Plan"
    tree_str, details_str = list_strip(plans[1].split("\\n\\n\\n"))

    from_ids_old, to_ids_old = get_tree_structure(tree_str)
    node_id2name, node_details_dict = get_node_details(details_str)
    if len(plans) == 4:
        assert plans[2] == "Subqueries"
        subqueries = list_strip(plans[3].split("Subquery:"))
        subqueries_detail_dict = {}
        for i, subquery in enumerate(subqueries):
            sub_tree_str = list_strip(subquery.split("\\n\\n\\n"))[0]
            sub_tree_strs = sub_tree_str.split("\\n")
            subquery_id = - int(sub_tree_strs[0].split()[0])
            subquery_detail = sub_tree_strs[0].split("Hosting Expression = ")[-1]
            subqueries_detail_dict[subquery_id] = subquery_detail

            if len(list_strip(subquery.split("\\n\\n\\n"))) > 1:
                sub_tree_str, sub_details_str = list_strip(subquery.split("\\n\\n\\n"))
                # add the link from subquery to main
                from_ids_old.append(subquery_id)
                to_ids_old.append(int(sub_tree_strs[0].split("Hosting operator id = ")[1].split()[0]))
                # add the link from the root subquery_id
                from_ids_old.append(get_op_id(sub_tree_strs[1]))
                to_ids_old.append(subquery_id)
                # add the dependencies in the subquery
                sub_tree_str2 = "\\n".join(sub_tree_strs[1:])
                sub_from_ids, sub_to_ids = get_tree_structure(sub_tree_str2)
                sub_node_id2name, sub_node_details_dict = get_node_details(sub_details_str)
                # add node "Subquery" or "SubqueryBroadcast"
                if sub_node_id2name[get_op_id(sub_tree_strs[1])] == "BroadcastExchange":
                    sub_node_id2name[subquery_id] = "SubqueryBroadcast"
                else:
                    sub_node_id2name[subquery_id] = "Subquery"
                sub_node_details_dict[subquery_id] = subquery_detail
                # merge to the main query
                from_ids_old += sub_from_ids
                to_ids_old += sub_to_ids
                node_id2name = {**node_id2name, **sub_node_id2name}
                node_details_dict = {**node_details_dict, **sub_node_details_dict}
            else:
                # todo: extend for TPCDS Q1,2,...
                ...
    from_ids_old, to_ids_old, node_id2name, node_details_dict = \
        connect_reused_exchange(from_ids_old, to_ids_old, node_id2name, node_details_dict)
    struct_data = construct_internal(from_ids_old, to_ids_old, node_id2name, reverse=True)
    node_details_dict = {struct_data.old["nids_old2new"][k]: v for k, v in node_details_dict.items()}
    return struct_data, node_details_dict


def construct_internal(from_ids, to_ids, node_id2name, reverse=False):
    # we only consider the nodes that are used in the topology (omit nodes like WSCG)
    nids_old, nids_new, nids_new2old, nids_old2new, \
    from_ids_new, to_ids_new, node_id2name_new = nodes_old2new(from_ids, to_ids, node_id2name, reverse)
    nnames = [node_id2name_new[nid] for nid in nids_new]
    return SqlStructData(
        nids=nids_new,
        nnames=[node_id2name_new[nid] for nid in nids_new],
        node_id2name={i: n for i, n in zip(nids_new, nnames)},
        from_ids=from_ids_new,
        to_ids=to_ids_new,
        old={
            "nids": nids_old,
            "from_ids": from_ids,
            "to_ids": to_ids,
            "node_id2name": node_id2name,
            "nids_new2old": nids_new2old,
            "nids_old2new": nids_old2new
        }
    )


class SqlStructData():
    def __init__(self, nids, nnames, node_id2name, from_ids, to_ids, old):
        self.nids = nids
        self.nnames = nnames
        self.node_id2name = node_id2name
        self.from_ids = from_ids
        self.to_ids = to_ids
        g = dgl.graph((from_ids, to_ids))
        G = dgl.to_networkx(g)
        nx.set_node_attributes(G, self.node_id2name, name="nname")
        self.g = g
        self.G = G
        self.old = old

    def plot(self, dir_name, title):
        plot_nx_graph(self.G, self.node_id2name, dir_name=dir_name, title=title)

    def graph_match(self, p2):
        G1, G2 = self.G, p2.G
        GM = isomorphism.GraphMatcher(G1, G2, node_match=lambda n1, n2: n1["nname"] == n2["nname"])
        if GM.is_isomorphic():
            return GM.mapping
        else:
            return None


class SqlStructBefore():
    def __init__(self, desc):
        self.struct, self.nodes_desc = construct_from_plan(desc)

    def get_op_feats(self, id_order=None):
        if id_order is None:
            return [self.nodes_desc[nid] for nid in self.struct.nids]
        else:
            return [self.nodes_desc[nid] for nid in id_order]


class SqlStructAfter():
    def __init__(self, d):
        self.struct = construct_from_metrics(d["sql_struct_sign"])
        self.nodes_metric = self.get_nodes_metric(d["nodes"])

    def get_nodes_metric(self, nodes_str):
        j_nodes = json.loads(nodes_str)
        nid_old_to_metric = {
            j_node["nodeId"]: j_node["metrics"]
            for j_node in j_nodes if j_node["nodeId"] in self.struct.old["nids"]
        }
        nid_to_metric = {self.struct.old["nids_old2new"][k]: v for k, v in nid_old_to_metric.items()}
        return nid_to_metric


class SqlStruct():

    def __init__(self, d: dict):
        self.d = d
        self.id = d["sql_struct_id"]
        self.struct_before = SqlStructBefore(d["planDescription"])
        self.struct_after = SqlStructAfter(d)
        self.p1 = self.struct_after.struct  # metric
        self.p2 = self.struct_before.struct  # planDesc

        # mapping from metrics (p1) to the plan (p2)
        mapping = self.p1.graph_match(self.p2)
        if mapping is None:
            raise Exception(f"{self.id} failed to pass the isomorphic test")
        else:
            self.mapping = dict(sorted(mapping.items()))  # sort the mapping in key

    def get_nnames(self):
        return self.p1.nnames

    def get_dgl(self, global_ops) -> dgl.DGLGraph:
        """return a DGLGraph with id of global operators"""
        g = self.p1.g
        g.ndata["op_gid"] = th.LongTensor([global_ops.index(name) for name in self.p1.nnames])
        return g


def extract_ofeats(lp, all=False):
    nops = len(lp)
    nids = np.arange(nops)
    from_ids, to_ids = get_tree_structure_internal(nids, lp)
    from2to, to2from = {}, {}
    for f, t in zip(from_ids, to_ids):
        if f in from2to:
            from2to[f].append(t)
        else:
            from2to[f] = [t]
        if t in to2from:
            to2from[t].append(f)
        else:
            to2from[t] = [f]

    nnames, sizes, nrows = [None] * nops, [None] * nops, [None] * nops
    for nid in reversed(nids):
        l = lp[nid]
        a, b, c = re.split(r"([a-z])", l, 1, flags=re.I)
        name = (b + c).split(" ")[0]
        if name == "Relation":
            name = (b + c).split("[")[0]
        nnames[nid] = name
        sizes[nid] = l.split("sizeInBytes=")[1].split("B")[0] + "B"
        if "rowCount=" in l:
            nrows[nid] = l.split("rowCount=")[1].split(")")[0]
        else:
            assert nid in to2from
            if len(to2from[nid]) == 1:
                pid = to2from[nid][0]
                assert nrows[pid] is not None
                nrows[nid] = nrows[pid]
            elif len(to2from[nid]) == 2:
                pid1, pid2 = to2from[nid]
                assert nrows[pid1] is not None and nrows[pid2] is not None
                assert float(nrows[pid1]) * float(nrows[pid2]) == 0
                nrows[nid] = "0"
            else:
                raise NotImplementedError
    if all:
        return sizes, nrows, nnames, from_ids, to_ids
    return sizes, nrows


def format_size(x):
    n, unit = x.replace(",", "").split()
    n = float(n)
    if unit == "B":
        return n
    elif unit == "KiB":
        return n * 1024
    elif unit == "MiB":
        return n * 1024 * 1024
    elif unit == "GiB":
        return n * 1024 * 1024 * 1024
    elif unit == "TiB":
        return n * 1024 * 1024 * 1024 * 1024
    else:
        raise Exception(f"unseen {unit} in {x}")


def format_time(x):
    n, unit = x.replace(",", "").split()
    n = float(n)
    if unit == "ms":
        return n / 1000
    elif unit == "s":
        return n
    elif unit == "m":
        return n * 60
    else:
        raise Exception(f"unseen {unit} in {x}")

class LogicalStruct():

    def __init__(self, lp):
        self.lp = lp
        nops = len(lp)
        nids = np.arange(nops)
        sizes, nrows, nnames, from_ids, to_ids = extract_ofeats(lp, all=True)
        self.struct = SqlStructData(
            nids=nids,
            nnames=nnames,
            node_id2name={i: n for i, n in zip(nids, nnames)},
            from_ids=from_ids,
            to_ids=to_ids,
            old=None
        )
        self.nids = nids
        self.nnames = nnames
        self.sizes = sizes
        self.nrows = nrows


def replace_symbols(s):
    return s.replace(" >= ", " GE ") \
        .replace(" <= ", " LE ") \
        .replace(" == ", " EQ") \
        .replace(" = ", " EQ ") \
        .replace(" > ", " GT ") \
        .replace(" < ", " LT ") \
        .replace(" != ", " NEQ ") \
        .replace(" + ", " rADD ") \
        .replace(" - ", " rMINUS ") \
        .replace(" / ", " rDIV ") \
        .replace(" * ", " rMUL ")


def remove_hash_suffix(s):
    return re.sub("#[0-9]+[L]*[,]*", " ", s)


def brief_clean(s):
    return re.sub("[^A-Za-z\'_]+", " ", s).lower()


def df_convert_query2op(df):
    return df.planDescription.apply(lambda x: SqlStructBefore(x).get_op_feats()).explode()

class Broadcast():
    def __init__(self, m):
        self.broadcast_s = None
        self.build_s = None
        self.collect_s = None
        self.rows = None
        self.size_mb = None

        ks, vs = JsonUtils.extract_json_list(m, ["name", "value"])
        for k, v in zip(ks, vs):
            if k == "time to broadcast":
                self.broadcast_s = format_time(v)
            elif k == "time to build":
                self.build_s = format_time(v)
            elif k == "time to collect":
                self.collect_s = format_time(v)
            elif k == "number of output rows":
                self.rows = float(v.replace(",", ""))
            elif k == "data size":
                self.size_mb = format_size(v) / 1024 / 1024
            else:
                raise Exception(f"unexpected {k} in {m}")
        for v in [self.broadcast_s, self.build_s, self.collect_s, self.rows, self.size_mb]:
            assert v is not None

    def tolist(self):
        return [self.broadcast_s, self.build_s, self.collect_s, self.rows, self.size_mb]

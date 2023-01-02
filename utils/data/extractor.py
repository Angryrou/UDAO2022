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

from utils.common import JsonUtils, ParquetUtils

from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import RegexpTokenizer

def get_tr_val_te_masks(df, groupby_col1, groupby_col2, frac_val_per_group, frac_te_per_group, seed):
    """return tr_mask, val_mask, te_mask"""
    random.seed(seed)
    np.random.seed(seed)
    n_rows = len(df)
    val_mask, te_mask = np.array([False] * n_rows), np.array([False] * n_rows)
    df = df.reset_index()
    val_index = df.groupby(groupby_col1).sample(frac=frac_val_per_group).index
    val_mask[val_index] = True
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


def nodes_old2new(from_ids, to_ids, node_id2name, reverse=False):
    nids_old = sorted(list(set(from_ids) | set(to_ids)), reverse=reverse)
    nids_new = list(range(len(nids_old)))
    nids_new2old = {n: o for n, o in zip(nids_new, nids_old)}
    nids_old2new = {o: n for n, o in zip(nids_new, nids_old)}
    from_ids_new = [nids_old2new[i] for i in from_ids]
    to_ids_new = [nids_old2new[i] for i in to_ids]
    node_id2name_new = {nids_old2new[k]: v for k, v in node_id2name.items() if k in nids_old2new}
    return nids_old, nids_new, nids_new2old, nids_old2new, from_ids_new, to_ids_new, node_id2name_new


def plot_nx_graph(G: networkx.DiGraph, node_id2name: dict, dir_name: str, title: str):
    p = nx.drawing.nx_pydot.to_pydot(G)
    for i, node in enumerate(p.get_nodes()):
        node.set_label(f"{i}-{node_id2name[i]}")
    dir_to_save = 'application_graphs/' + dir_name
    os.makedirs(dir_to_save, exist_ok=True)
    p.write_png(dir_to_save + '/' + title + '.png')


def plot_dgl_graph(g: dgl.DGLGraph, node_id2name: dict, dir_name: str, title: str):
    G = dgl.to_networkx(g)
    plot_nx_graph(G, node_id2name, dir_name, title)


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

class SqlStuctBefore():
    def __init__(self, desc):
        self.struct, self.nodes_desc = construct_from_plan(desc)

    def get_op_feats(self, id_order=None):
        if id_order is None:
            return [self.nodes_desc[nid] for nid in self.struct.nids]
        else:
            return [self.nodes_desc[nid] for nid in id_order]

class SqlStuctAfter():
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
        self.struct_before = SqlStuctBefore(d["planDescription"])
        self.struct_after = SqlStuctAfter(d)
        self.p1 = self.struct_after.struct # metric
        self.p2 = self.struct_before.struct # planDesc

        # mapping from metrics (p1) to the plan (p2)
        mapping = self.p1.graph_match(self.p2)
        if mapping is None:
            raise Exception(f"{self.id} failed to pass the isomorphic test")
        else:
            self.mapping = dict(sorted(mapping.items())) # sort the mapping in key

    def get_nnames(self):
        return self.p1.nnames

    def get_dgl(self, global_ops) -> dgl.DGLGraph:
        """return a DGLGraph with id of global operators"""
        g = self.p1.g
        g.ndata["op_gid"] = th.LongTensor([global_ops.index(name) for name in self.p1.nnames])
        return g


def replace_symbols(s):
    return s.replace(" >= ", " GE ")\
        .replace(" <= ", " LE ")\
        .replace(" == ", " EQ")\
        .replace(" = ", " EQ ")\
        .replace(" > ", " GT ")\
        .replace(" < ", " LT ")\
        .replace(" != ", " NEQ ")\
        .replace(" + ", " rADD ")\
        .replace(" - ", " rMINUS ")\
        .replace(" / ", " rDIV ")\
        .replace(" * ", " rMUL ")


def self_eval_ori(model, train_corpus):
    match_cnt = 0
    start = time.time()
    for i in range(len(train_corpus)):
        doc_id = train_corpus[i].tags[0]
        assert doc_id == i
        inferred_vector = model.infer_vector(train_corpus[i].words)
        if model.dv.most_similar([inferred_vector], topn=1)[0][0] == doc_id:
            match_cnt += 1
    dt = time.time() - start
    rate = match_cnt/len(train_corpus)
    print(f"matched: {match_cnt}/{len(train_corpus)} = {rate:.3f}, "
          f"cost {dt:.3f}s for evaluation")
    return rate

def self_evals(model, train_corpus, sample=1000):
    match_cnt = 0
    start = time.time()
    tc = random.sample(train_corpus, sample)
    for i in range(len(tc)):
        doc_id = tc[i].tags[0]
        inferred_vector = model.infer_vector(tc[i].words)
        if train_corpus[model.dv.most_similar([inferred_vector], topn=1)[0][0]].tags[0] == doc_id:
            match_cnt += 1
    dt = time.time() - start
    rate = match_cnt/len(train_corpus)
    print(f"matched: {match_cnt}/{len(train_corpus)} = {rate:.3f}, "
          f"cost {dt:.3f}s for evaluation")
    return rate

def infer_evals(model, corpus):
    return [model.infer_vector(c.words) for c in corpus]


def evals(model, corpus, real_cat, mapping_to_cat):
    pred_cat = -1 * np.ones(len(real_cat))
    for doc_id in range(len(corpus)):
        inferred_vector = model.infer_vector(corpus[doc_id].words)
        match = model.dv.most_similar([inferred_vector], topn=1)[0][0]
        pred_cat[doc_id] = mapping_to_cat[match]
    match_rate = (real_cat == real_cat).sum() / len(real_cat)
    return match_rate


def df_convert_query2op(df):
    return df.planDescription.apply(lambda x: SqlStuctBefore(x).get_op_feats()).explode()

def tokenize_op_descs(df):
    # tokenize reference:
    # https://towardsdatascience.com/dynamic-word-tokenization-with-regex-tokenizer-801ae839d1cd
    # https://regex101.com/
    tokenizer = RegexpTokenizer(
        "(?:(?<=\s)|(?<=^)|(?<=[\[\"()<>,=]))[a-z0-9'_#.+\-*\/:=]*[a-z0-9](?:(?=\s)|(?=$)|(?=[\]():,=<>]))")
    tokens_list = list(map(lambda x: tokenizer.tokenize(x.lower()), df.values))
    return [TaggedDocument(d, [i]) for i, d in enumerate(tokens_list)]

def get_operator_descs(input_df):
    """
    return all_operators and all_operators_cat with cat1=(operator_type), cat2=(struct_id, operator_type)
    """
    # operator's desc will be cleaned for "\n" and "\n\n" at `SqlStuctBefore`.
    all_operators = df_convert_query2op(input_df)
    all_operators = all_operators.apply(replace_symbols)
    all_operators_cat = all_operators.apply(lambda x: x.split()[0]).reset_index()
    all_operators_cat.columns = ["struct_id", "id", "cat1"]
    all_operators_cat["cat2"] = all_operators_cat.apply(lambda r: f"{r['struct_id']}-{r['cat1']}", axis=1)
    cat1 = sorted(all_operators_cat.cat1.unique())
    cat2 = sorted(all_operators_cat.cat2.unique())
    all_operators_cat["cat1_index"] = all_operators_cat["cat1"].replace(cat1, list(range(len(cat1))))
    all_operators_cat["cat2_index"] = all_operators_cat["cat2"].replace(cat2, list(range(len(cat2))))
    return all_operators, all_operators_cat

def prepare_operator_tokens(all_operators, all_operators_cat, debug, seed):
    train_mask, eval1_mask, eval2_mask = get_tr_val_te_masks(
        all_operators_cat, "cat1_index", "cat2_index", 0.2 if debug else 0.1, 0.2 if debug else 0.1, seed)
    all_operators_tr, all_operators_eval1, all_operators_eval2 = \
        all_operators[train_mask], all_operators[eval1_mask], all_operators[eval2_mask]
    print(f"get {len(all_operators_tr)} TR, {len(all_operators_eval1)} EVAL1, "
          f"and {len(all_operators_eval2)} EVAL2 rows")
    train_corpus, eval1_corpus, eval2_corpus = [tokenize_op_descs(all_operators_) for all_operators_ in
        [all_operators_tr, all_operators_eval1, all_operators_eval2]]
    return train_corpus, eval1_corpus, eval2_corpus, train_mask, eval1_mask, eval2_mask

def get_d2v_model(cache_header, n_samples, input_df, workers, seed, debug, vec_size=20, epochs=400, alpha=0.025):
    model_path = f"{cache_header}/d2v_{n_samples}_tr_samples.model"
    try:
        model = Doc2Vec.load(model_path)
        print(f"found d2v_{n_samples}_tr_samples model at {model_path}")
    except:
        print(f"cannot find d2v_{n_samples}_tr_samples model at {model_path}, start generating...")
        all_operators, all_operators_cat = get_operator_descs(input_df)
        train_corpus, eval1_corpus, eval2_corpus, train_mask, eval1_mask, eval2_mask = prepare_operator_tokens(
            all_operators, all_operators_cat, debug, seed)
        eval1_cat1, eval2_cat2 = all_operators_cat.cat1[eval1_mask], all_operators_cat.cat2[eval2_mask]

        model = Doc2Vec(vector_size=vec_size, alpha=alpha, workers=workers, min_count=1, dm=1)
        model.build_vocab(train_corpus)
        start = time.time()
        model.train(train_corpus, total_examples=model.corpus_count, epochs=epochs)
        print(f"d2v model training cost {time.time() - start:.3f}s")
        tr_samples = min(len(train_corpus), max(1000, n_samples / 10))
        rate_tr = self_evals(model, train_corpus, sample=tr_samples)
        rate1 = evals(model, eval1_corpus, real_cat=eval1_cat1, mapping_to_cat=all_operators_cat.cat1_index)
        rate2 = evals(model, eval2_corpus, real_cat=eval2_cat2, mapping_to_cat=all_operators_cat.cat2_index)
        print(f"match_rate: TR_over_{tr_samples}/{len(train_corpus)}_samples={rate_tr * 100:.1f}%, "
              f"EVAL1={rate1 * 100:.1f}%, EVAL2={rate2 * 100:.1f}%")
        if not debug:
            model.save(f"{cache_header}/d2v_{n_samples}_tr_samples.model")
    return model
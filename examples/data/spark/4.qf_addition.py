# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: generate additional necessary information for running QueryFormer
#
# Created at 12/02/2023
from utils.common import PickleUtils
import argparse
import torch as th
import dgl

from utils.data.extractor import plot_dgl_graph
from utils.model.utils import get_tensor


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--src-path-header", type=str, default="resources/dataset/tpch_100_query_traces")
        self.parser.add_argument("--cache-header", type=str, default="examples/data/spark/cache")
        self.parser.add_argument("--debug", type=int, default=1)
        self.parser.add_argument("--seed", type=int, default=42)

    def parse(self):
        return self.parser.parse_args()


args = Args().parse()
print(args)
bm = args.benchmark.lower()
sf = args.scale_factor
cache_header = f"{args.cache_header}/{bm}_{sf}"
struct_cache = PickleUtils.load(cache_header, "struct_cache.pkl")
cache_name = "qf_dgl.pkl"


struct_dgl_dict, dgl_dict = struct_cache["struct_dgl_dict"], struct_cache["dgl_dict"]
dst2src_dep_dict = {} # src2dst
d_dict = {} # d
qf_dgl_dict = {}
for i, g in dgl_dict.items():
    dep, d = {}, {}
    srcs, dsts = g.edges()
    for src, dst in zip(srcs.numpy(), dsts.numpy()):
        if dst not in dep:
            dep[dst] = [src]
        else:
            dep[dst].append(src)
        d[(src, dst)] = [1]
    dst2src_dep_dict[i] = dep

    g_src_nids = th.where(g.in_degrees() == 0)[0]
    for g_src_nid in g_src_nids: # get entry ids
        for rank in dgl.bfs_nodes_generator(g, g_src_nid): # bfs ranks
            for dst_nid in rank.numpy():
                if dst_nid not in dep:
                    continue
                src_nids_cur = dep[dst_nid]
                d_ = 1
                while len(src_nids_cur) > 0:
                    src_nids_next = []
                    for src_nid in src_nids_cur:
                        if d_ == 1:
                            assert 1 in d[(src_nid, dst_nid)]
                        if src_nid in dep:
                            src_nids_next += dep[src_nid]
                        if (src_nid, dst_nid) in d and d_ not in d[(src_nid, dst_nid)]:
                            d[(src_nid, dst_nid)].append(d_)
                        else:
                            d[(src_nid, dst_nid)] = [d_]
                    d_ += 1
                    src_nids_cur = src_nids_next
    d_dict[i] = d

new_dgl_dict = {}
q_signs = {kv["sql_struct_id"]: kv["q_sign"] for kv in struct_cache["struct_dict"].values()}
for i, d in d_dict.items():
    sids, dids, ds = [], [], []
    for sd, d_list in d.items():
        for d_i in d_list:
            sids.append(sd[0])
            dids.append(sd[1])
            ds.append(d_i)
    new_g = dgl.graph((sids, dids))
    new_g.edata["dist"] = get_tensor(ds, dtype=th.int)
    for k in dgl_dict[i].ndata.keys():
        new_g.ndata[k] = dgl_dict[i].ndata[k]
    new_dgl_dict[i] = new_g

    id = struct_dgl_dict[i].id
    q_sign = q_signs[i]
    # plot_dgl_graph(
    #     new_g,
    #     struct_dgl_dict[i].p1.node_id2name,
    #     "tpch_100_from_metric_qf",
    #     f"{id}-{q_sign}"
    # )

PickleUtils.save(new_dgl_dict, cache_header, cache_name, overwrite=True)
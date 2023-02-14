# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: generate additional necessary information for running RAAL
#
# Created at 12/02/2023

from utils.common import PickleUtils
import argparse


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
cache_name = "raal_dgl.pkl"

struct_dgl_dict, dgl_dict = struct_cache["struct_dgl_dict"], struct_cache["dgl_dict"]
non_siblings_map = {}
for i, g in dgl_dict.items():
    srcs, dsts, eids = g.edges(form="all", order='srcdst')
    child_dep = {}
    for src, dst in zip(srcs.numpy(), dsts.numpy()):
        if dst in child_dep:
            child_dep[dst].append(src)
        else:
            child_dep[dst] = [src]

    total_nids = set(range(g.num_nodes()))
    non_sibling = {}
    for src, dst, eid in zip(srcs.numpy(), dsts.numpy(), eids.numpy()):
        non_sibling[eid] = total_nids.difference(set(child_dep[dst]))
    non_siblings_map[i] = {k: list(v) for k, v in non_sibling.items()}

PickleUtils.save(non_siblings_map, cache_header, cache_name, overwrite=True)
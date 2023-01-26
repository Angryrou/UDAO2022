# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: pick one query from each template such that the physical query plan in the traces has the largest number
#              of BroadcastHashJoin among those in the same template. That is to say, the physical query structure in
#              the trace should be the same as when s3=320M (the largest value in our experiment)
#              Since I have already run many manually tuning over *-1, I will pick *-1 as necessary as possible.
# Created at 25/01/2023

from utils.common import PickleUtils, BenchmarkUtils
from utils.data.extractor import get_csvs
from utils.data.extractor import SqlStructAfter

data_header = "examples/data/spark/cache/tpch_100"
struct_cache_name = "struct_cache.pkl"
struct_cache = PickleUtils.load(data_header, struct_cache_name)
df = get_csvs(
    templates=[f"q{i}" for i in BenchmarkUtils.get("tpch")],
    header="resources/dataset/tpch_100_query_traces",
    cache_header="examples/data/spark/cache/tpch_100",
    samplings=["lhs", "bo"]
)

struct_dgl_dict = struct_cache["struct_dgl_dict"]
struct_dict = struct_cache["struct_dict"]
# {tid: [n_bhj, s3, sid]}
min_s3_dict = {f"q{i}": [-1, -1, -1, ""] for i in range(1, 23)}
for k, v in struct_dict.items():
    tid = k[0]
    name = v["name"]
    s3 = name.split(",")[-2]
    struct_dgl = struct_dgl_dict[v["sql_struct_id"]]
    n_bhj = len([o for o in struct_dgl.get_nnames() if o == "BroadcastHashJoin"])
    if n_bhj > min_s3_dict[tid][0]:
        min_s3_dict[tid] = [n_bhj, s3, v["sql_struct_id"], v["q_sign"]]

target_qsigns = []
for tid in range(1, 23):
    record = df[df["q_sign"] == f"q{tid}-1"]
    sid, svid = record.sql_struct_id[0], record.sql_struct_svid[0]
    assert int(svid) == 0
    n_bhj_max, s3, sid_max, q_sign_cand = min_s3_dict[f"q{tid}"]
    d = record.iloc[0].to_dict()
    sql_sa = SqlStructAfter(d)
    node_names = sql_sa.struct.nnames
    n_bhj = len([o for o in node_names if o == "BroadcastHashJoin"])

    if sid == sid_max:
        assert n_bhj_max == n_bhj
        target_qsigns.append(f"q{tid}-1")
    elif sid < sid_max:
        assert n_bhj_max > n_bhj
        target_qsigns.append(q_sign_cand)

print(",".join(target_qsigns))

# s3_to_nbhj_map = {f"q{i}": {} for i in range(1, 23)}
# for r in df.iterrows():
#     k = r[0][0]
#     s3 = r[1]["knob_sign"].split(",")[-2]
#     n_bhj = len([o for o in SqlStructAfter(r[1].to_dict()).struct.nnames if o == "BroadcastHashJoin"])
#     if s3 in s3_to_nbhj_map[k]:
#         s3_to_nbhj_map[k][s3].add(n_bhj)
#     else:
#         s3_to_nbhj_map[k][s3] = set([n_bhj])
#
# s3_to_nbhj_map2 = {k: {kk: min(vv) for kk, vv in v.items()} for k, v in s3_to_nbhj_map.items()}
# s3_adjustable = {}
# for k, v in s3_to_nbhj_map2.items():
#     s3_6 = v["6"]
#     rlist = [i for i in range(6) if v[str(i)] == s3_6]
#     s3_adjustable[k] = rlist
#
# ------------------------------------------------------------------------------------------------------------
# get:
# s3_adjustable = {
#     'q1': [0, 1, 2, 3, 4, 5],
#     'q2': [4, 5],
#     'q3': [4, 5],
#     'q4': [0, 1, 2, 3, 4, 5],
#     'q5': [3, 4, 5],
#     'q6': [0, 1, 2, 3, 4, 5],
#     'q7': [5],
#     'q8': [],
#     'q9': [],
#     'q10': [5],
#     'q11': [0, 1, 2, 3, 4, 5],
#     'q12': [5],
#     'q13': [0, 1, 2, 3, 4, 5],
#     'q14': [],
#     'q15': [0, 1, 2, 3, 4, 5],
#     'q16': [],
#     'q17': [1, 2, 3, 4, 5],
#     'q18': [0, 1, 2, 3, 4, 5],
#     'q19': [3, 4, 5],
#     'q20': [],
#     'q21': [4, 5],
#     'q22': [0, 1, 2, 3, 4, 5]}

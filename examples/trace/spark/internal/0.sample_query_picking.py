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

# print("q\tn_bhj\ts3\tsid")
# for k, (v1, v2, v3) in min_s3_dict.items():
#     print(f"{k}\t{v1}\t{v2}\t{v3}")







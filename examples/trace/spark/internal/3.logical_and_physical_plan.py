# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 12/01/2023

from utils.common import PickleUtils

data_header = "examples/data/spark/cache/tpch_100"
struct_file="struct_cache.pkl"
struct_data = PickleUtils.load(data_header, struct_file)

for l in struct_data["struct_dict"].values():
    q_sign = l["q_sign"]
    knob_sign = l["knob_sign"]
    path = f"~/chenghao/UDAO2022/resources/scripts/tpch-{l['sampling']}/{q_sign.split('-')[0][1:]}/{q_sign}_{knob_sign}.sh"

    print(q_sign, path)




# verify q_signs with the same knob_sign will have the same topology structure.

# to verify if q_signs with different knob_signs would have same Logical Plan.

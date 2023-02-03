# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: e2e training via GTN
#
# Created at 03/01/2023
import os

from utils.model.args import ArgsGTN
from utils.model.parameters import set_params
from utils.model.utils import expose_data, pipeline

args = ArgsGTN().parse()
print(args)
pj = f"{args.benchmark.lower()}_{args.scale_factor}"
debug = False if args.debug == 0 else True
data_header = f"{args.data_header}/{args.benchmark.lower()}_{args.scale_factor}"
assert os.path.exists(data_header), f"data not exists at {data_header}"
assert args.granularity in ("Q", "QS")

data_params, learning_params, net_params = set_params(args)
obj, model_name = data_params["obj"], data_params["model_name"]
ckp_header = f"examples/model/spark/ckp/{pj}/{model_name}/{obj}/" \
             f"{'_'.join([data_params[f] for f in ['ch1_type', 'ch1_cbo', 'ch1_enc', 'ch2', 'ch3', 'ch4']])}"

ds_dict, col_dict, minmax_dict, dag_dict, n_op_types, op_feats_data = expose_data(
    header=data_header,
    tabular_file=f"{'query_level' if args.granularity == 'Q' else 'stage_level'}_cache_data.pkl",
    struct_file="struct_cache.pkl",
    op_feats_file="logical_plans.parquet",
    debug=debug
)

data_meta = [ds_dict, op_feats_data, col_dict, minmax_dict, dag_dict, n_op_types]
model, results = pipeline(data_meta, data_params, learning_params, net_params, ckp_header)

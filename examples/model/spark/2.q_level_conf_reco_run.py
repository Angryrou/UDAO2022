# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: used weighted sum to recommend a Pareto set.
#
# Created at 08/01/2023

import os
import time

from trace.parser.spark import get_cloud_cost
from utils.common import BenchmarkUtils, PickleUtils, TimeUtils, FileUtils
from utils.data.collect import run_q_confs
from utils.data.configurations import SparkKnobs
from utils.model.args import ArgsRecoQRun
from utils.model.parameters import set_data_params, get_gpus
from utils.model.proxy import ModelProxy, ws_return
from utils.model.utils import expose_data, analyze_cols, add_pe, prepare_data_for_opt, \
    get_sample_spark_knobs
from utils.optimization.moo_utils import is_pareto_efficient

import numpy as np
import pandas as pd

args = ArgsRecoQRun().parse()
print(args)

bm, sf, pj = args.benchmark.lower(), args.scale_factor, f"{args.benchmark.lower()}_{args.scale_factor}"
debug = False if args.debug == 0 else True
seed = args.seed
data_header = f"{args.data_header}/{pj}"
query_header = args.query_header
assert os.path.exists(data_header), f"data not exists at {data_header}"
assert args.model_name == "GTN"
model_name = args.model_name
obj = args.obj
ckp_sign = args.ckp_sign
n_samples = args.n_samples
gpus, device = get_gpus(args.gpu)
q_signs = BenchmarkUtils.get_sampled_q_signs(bm) if args.q_signs is None else \
    [BenchmarkUtils.extract_sampled_q_sign(bm, sign) for sign in args.q_signs.split(",")]
assert len(q_signs) == 1, "only support run queries one at a time"
q_sign = q_signs[0]

data_params = set_data_params(args)
ckp_header = f"examples/model/spark/ckp/{pj}/{model_name}/{obj}/" \
             f"{'_'.join([data_params[f] for f in ['ch1_type', 'ch1_cbo', 'ch1_enc', 'ch2', 'ch3', 'ch4']])}"
ckp_path = os.path.join(ckp_header, ckp_sign)
out_header = f"{ckp_path}/{q_sign}"


assert args.algo in ["robust", "vc"] and args.moo in ["bf", "ws"] and args.alpha in (-3, -2, 0, 2, 3)
algo, moo, alpha = args.algo, args.moo, args.alpha
cache_prefix = f"rs({n_samples}x100)"
moo_sign = moo if moo == "bf" else f"{moo}({args.n_weights})"
cache_conf_name = f"{cache_prefix}_po_{algo}_{moo_sign}_alpha({alpha:.0f}).pkl"
if_aqe = False if args.if_aqe == 0 else True
conf_cache = PickleUtils.load(out_header, cache_conf_name)


conf_df_pareto = conf_cache["conf_df"]
print(f"prepared to run {len(conf_df_pareto)} recommended PO configurations")
aqe_sign = "aqe_on" if if_aqe else "aqe_off"
script_header = f"examples/trace/spark/internal/2.knob_hp_tuning/{bm.lower()}_{aqe_sign}/{q_sign}"
spark_knobs = SparkKnobs(meta_file="resources/knob-meta/spark.json")
start = time.time()
run_q_confs(
    bm=bm, sf=sf, spark_knobs=spark_knobs, query_header=query_header,
    out_header=script_header, seed=seed, workers=BenchmarkUtils.get_workers(args.worker),
    n_trials=3, debug=debug, q_sign=q_sign, conf_df=conf_df_pareto, if_aqe=if_aqe)
cache_res_name = f"{cache_prefix}_po_{algo}_{moo_sign}_alpha({alpha:.0f})_dt({'aqe_on' if if_aqe else 'aqe_off'})"
FileUtils.write_str(f"{out_header}/{cache_res_name}.{TimeUtils.get_current_iso()}", f"{time.time() - start}s")
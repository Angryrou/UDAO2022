# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description:
#
# Created at 12/23/22

import argparse, random
import numpy as np
from utils.common import BenchmarkUtils
from utils.data.extractor import get_csvs, SqlStruct, SqlStuctBefore, replace_symbols, evals_self, evals, infer_evals, \
    get_tr_val_te_masks, get_d2v_model, df_convert_query2op, tokenize_op_descs


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--src-path-header", type=str, default="resources/dataset/tpch_100_query_traces")
        self.parser.add_argument("--cache-header", type=str, default="examples/data/spark/cache")
        self.parser.add_argument("--workers", type=int, default=1)
        self.parser.add_argument("--debug", type=int, default=1)
        self.parser.add_argument("--seed", type=int, default=42)
        self.parser.add_argument("--mode", type=str, default="d2v")
        self.parser.add_argument("--n-samples-for-tr", type=int, default=10000)
        self.parser.add_argument("--vec-size", type=int, default=32)
        self.parser.add_argument("--alpha", type=float, default=0.025)
        self.parser.add_argument("--epochs", type=int, default=200)

    def parse(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = Args().parse()
    print(args)
    bm = args.benchmark.lower()
    sf = args.scale_factor
    src_path_header = args.src_path_header
    cache_header = f"{args.cache_header}/{bm}_{sf}"
    workers = args.workers
    debug = False if args.debug == 0 else True
    seed = args.seed
    mode = args.mode
    n_samples = args.n_samples_for_tr
    assert mode in ("d2v", "w2v"), ValueError(mode)

    random.seed(seed)
    np.random.seed(seed)

    templates = [f"q{i}" for i in BenchmarkUtils.get(bm)]
    df = get_csvs(templates, src_path_header, cache_header, samplings=["lhs", "bo"])
    tr_mask, val_mask, te_mask = get_tr_val_te_masks(df=df, groupby_col1="template", groupby_col2="template",
        frac_val_per_group=0.1, frac_te_per_group=0.1, seed=seed)
    df_tr, df_val, df_te = df[tr_mask], df[val_mask], df[te_mask]

    input_df = df_tr.loc[df_tr.sql_struct_id.drop_duplicates().index] if debug else df_tr.sample(n_samples)
    input_df = input_df.reset_index().rename(columns={"level_0": "template", "level_1": "vid"}) \
        .set_index(["sql_struct_id", "id"])
    print(f"get {len(input_df)} queries")

    if mode == "d2v":
        for vec_size in [20, 50, 100, 200]:
            for epochs in [5, 10, 20]:
                model = get_d2v_model(cache_header, n_samples, input_df, workers, seed, debug,
                                      vec_size=vec_size, epochs=epochs, alpha=args.alpha)

        # todo: cache operator features to d2v_features.parquet for each struct_id

        # corpus_tr, corpus_val, corpus_te = [tokenize_op_descs(df_convert_query2op(df_))
        #                                     for df_ in [df_tr, df_val, df_te]]
        # vecs_tr, vecs_val, vecs_te = [infer_evals(model, corpus_) for corpus_ in [corpus_tr, corpus_val, corpus_te]]
        #
        # d2v_cols = [f"d2v_{i}" for i in range(20)]
        # all_operators_feat = pd.DataFrame(data=[], index=all_operators.index, columns=d2v_cols)
        # all_operators_feat[train_mask] = vecs_tr
        # all_operators_feat[eval1_mask] = vecs_ev1
        # all_operators_feat[eval2_mask] = vecs_ev2

    elif mode == "w2v":
        ...
    else:
        raise ValueError(mode)

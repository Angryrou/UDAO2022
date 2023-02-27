# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: e2e training via GTN
#
# Created at 03/01/2023

if __name__ == "__main__":

    import os
    from utils.data.feature import L2P_MAP
    from utils.model.args import ArgsTrainLatBuck
    from utils.model.parameters import set_params
    from utils.model.utils import expose_data, pipeline

    args = ArgsTrainLatBuck().parse()
    print(args)
    debug = False if args.debug == 0 else True
    data_header = f"{args.data_header}/{args.benchmark.lower()}_{args.scale_factor}"
    assert os.path.exists(data_header), f"data not exists at {data_header}"
    assert args.granularity in ("Q", "QS")

    bid = args.bid
    bsize = args.bsize
    finetune_header = args.finetune_header

    if args.benchmark.lower() == "tpch" and args.scale_factor == 100:
        if bsize == 20:
            assert bid in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23], \
                ValueError(bid)
        else:
            raise ValueError(bsize)

    data_params, learning_params, net_params = set_params(args)
    obj, model_name = data_params["obj"], data_params["model_name"]
    if finetune_header is None:
        pj = f"{args.benchmark.lower()}_{args.scale_factor}_l{bid}"
        ckp_header = f"examples/model/spark/ckp/{pj}/{model_name}/{obj}/" \
                     f"{'_'.join([data_params[f] for f in ['ch1_type', 'ch1_cbo', 'ch1_enc', 'ch2', 'ch3', 'ch4']])}"
    else:
        ckp_header = f"{finetune_header}/finetune_b{bsize}x{bid}"

    op_feats_file = {}
    if data_params["ch1_cbo"] == "on":
        op_feats_file["cbo"] = "cbo_cache.pkl"
    if data_params["ch1_enc"] != "off":
        ch1_enc = data_params["ch1_enc"]
        op_feats_file["enc"] = f"enc_cache_{ch1_enc}.pkl"

    ds_dict_all, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, op_feats_data, clf_feat = expose_data(
        header=data_header,
        tabular_file=f"{'query_level' if args.granularity == 'Q' else 'stage_level'}_cache_data.pkl",
        struct_file="struct_cache.pkl",
        op_feats_file=op_feats_file,
        debug=debug,
        model_name=model_name
    )

    if data_params["ch1_cbo"] == "on":
        op_feats_data["cbo"]["l2p"] = L2P_MAP[args.benchmark.lower()]

    ds_dict = ds_dict_all.filter(lambda e: e["latency"] // bsize == bid)
    data_meta = [ds_dict, op_feats_data, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, clf_feat]
    pipeline(data_meta, data_params, learning_params, net_params, ckp_header, finetune_header)

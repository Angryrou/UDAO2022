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
    from utils.common import PickleUtils
    import numpy as np

    args = ArgsTrainLatBuck().parse()
    print(args)
    debug = False if args.debug == 0 else True
    data_header = f"{args.data_header}/{args.benchmark.lower()}_{args.scale_factor}"
    assert os.path.exists(data_header), f"data not exists at {data_header}"
    assert args.granularity in ("Q", "QS")
    assert args.finetune_header is not None

    bid = args.bid
    bsize = args.bsize
    finetune_header = args.finetune_header

    if args.benchmark.lower() == "tpch" and args.scale_factor == 100:
        if bsize == "20":
            assert bid in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23], \
                ValueError(bid)
        elif bsize in ("3w", "3h", "3c"):
            assert bid in [0, 1, 2], ValueError(bid)
        else:
            raise ValueError(bsize)

    data_params, learning_params, net_params = set_params(args)
    obj, model_name = data_params["obj"], data_params["model_name"]
    assert obj == "latency"
    ckp_header = f"{finetune_header}/finetune_b{bsize}x{bid}_hat"

    op_feats_file = {}
    if data_params["ch1_cbo"] == "on":
        op_feats_file["cbo"] = "cbo_cache.pkl"
    elif data_params["ch1_cbo"] == "on2":
        op_feats_file["cbo"] = "cbo_cache_recollect.pkl"

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

    if data_params["ch1_cbo"] in ("on", "on2"):
        op_feats_data["cbo"]["l2p"] = L2P_MAP[args.benchmark.lower()]
    lat_hat_dict = PickleUtils.load(finetune_header, "lat_hat_all.pkl")
    for split, lat_hat in lat_hat_dict.items():
        ds_dict_all[split] = ds_dict_all[split].add_column("latency_hat", lat_hat.squeeze().tolist())

    if bsize == "20":
        bsize = int(bsize)
        if bid >= 15:
            bid = 15
            ds_dict = ds_dict_all.filter(lambda e: e["latency_hat"] // bsize >= bid)
        else:
            ds_dict = ds_dict_all.filter(lambda e: e["latency_hat"] // bsize == bid)
    else:
        if bsize == "3c":
            lat_splits = [10, 100]
        else:
            def get_3bid(l, ls):
                if l < ls[0]:
                    return 0
                if l < ls[1]:
                    return 1
                return 2
            lats = np.hstack([v["latency_hat"] for v in ds_dict_all.values()])
            if bsize == "3h": # equal-height split
                lat_splits = np.percentile(lats, [33, 67])
            else: # equal-width split
                lmin, lmax = min(lats), max(lats)
                gap = lmax - lmin
                lat_splits = [lmin + gap / 3, lmin + gap / 3 * 2]
        ds_dict = ds_dict_all.filter(lambda e: get_3bid(e["latency_hat"], lat_splits) == bid)

    data_meta = [ds_dict, op_feats_data, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, clf_feat]
    pipeline(data_meta, data_params, learning_params, net_params, ckp_header, finetune_header)

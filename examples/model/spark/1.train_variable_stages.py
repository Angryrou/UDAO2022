# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: temporary implementation for the global model with the configuration including
# 4 runtime parameters per stage as input
#
# Created at 03/01/2023

if __name__ == "__main__":

    import os
    from utils.model.args import ArgsTrain
    from utils.model.parameters import set_params
    from utils.model.utils import expose_data, pipeline

    args = ArgsTrain().parse()
    print(args)
    pj = f"{args.benchmark.lower()}_{args.scale_factor}"
    debug = False if args.debug == 0 else True
    data_header = f"{args.data_header}/{args.benchmark.lower()}_{args.scale_factor}"
    assert os.path.exists(data_header), f"data not exists at {data_header}"
    assert args.granularity == "Q"

    data_params, learning_params, net_params = set_params(args)
    obj, model_name = data_params["obj"], data_params["model_name"]
    assert obj == "latency" and model_name == "AVGMLP_GLB"
    assert data_params["ch1_type"] == "on" and data_params["ch1_cbo"] == "off" and data_params["ch1_enc"] == "off"
    ckp_header = f"examples/model/spark/ckp/{pj}/{model_name}/{obj}/" \
                 f"{'_'.join([data_params[f] for f in ['ch1_type', 'ch1_cbo', 'ch1_enc', 'ch2', 'ch3', 'ch4']])}"

    op_feats_file = {}
    ds_dict, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, op_feats_data, clf_feat = expose_data(
        header=data_header,
        tabular_file=f"{'query_level' if args.granularity == 'Q' else 'stage_level'}_cache_data.pkl",
        struct_file="struct_cache.pkl",
        op_feats_file=op_feats_file,
        debug=debug,
        model_name=model_name,
        clf_feat_file=data_params["clf_feat"]
    )
    # if data_params["ch1_cbo"] in ("on", "on2"):
    #     op_feats_data["cbo"]["l2p"] = L2P_MAP[args.benchmark.lower()]

    data_meta = [ds_dict, op_feats_data, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, clf_feat]
    model, results, hp_params, hp_prefix_sign = pipeline(data_meta, data_params, learning_params, net_params, ckp_header)

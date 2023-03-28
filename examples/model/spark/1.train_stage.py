# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: e2e training for stage latency via GTN
#
# Created at 26/03/2023

if __name__ == "__main__":

    import os
    from utils.data.feature import L2P_MAP
    from utils.model.args import ArgsTrain
    from utils.model.parameters import set_params
    from utils.model.utils import expose_data_stage, pipeline_stage

    args = ArgsTrain().parse()
    print(args)
    pj = f"{args.benchmark.lower()}_{args.scale_factor}"
    debug = False if args.debug == 0 else True
    data_header = f"{args.data_header}/{args.benchmark.lower()}_{args.scale_factor}"
    assert os.path.exists(data_header), f"data not exists at {data_header}"
    assert args.granularity == "QS"


    data_params, learning_params, net_params = set_params(args)
    obj, model_name = data_params["obj"], data_params["model_name"]
    ckp_header = f"examples/model/spark/ckp/{pj}/{model_name}/{obj}/" \
                 f"{'_'.join([data_params[f] for f in ['ch1_type', 'ch1_cbo', 'ch1_enc', 'ch2', 'ch3', 'ch4']])}"

    ds_dict, dag_misc, stage_feat, col_dict, minmax_dict, n_op_types = expose_data_stage(
        header=data_header,
        tabular_file="stage_level_cache_data.pkl",
        struct_file="struct_cache.pkl",
        data_params=data_params,
        benchmark=args.benchmark.lower(),
        model_name=model_name,
        debug=debug
    )

    data_meta = [ds_dict, dag_misc, stage_feat, col_dict, minmax_dict, n_op_types]
    model, results, hp_params, hp_prefix_sign = \
        pipeline_stage(data_meta, data_params, learning_params, net_params, ckp_header)

# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: pretrain a classifier model
#
# Created at 03/01/2023
from utils.common import PickleUtils

if __name__ == "__main__":

    import os
    from utils.data.feature import L2P_MAP
    from utils.model.args import ArgsTrain
    from utils.model.parameters import set_params
    from utils.model.utils import expose_data, pipeline_classifier, augment_net_params, setup_data, collate_clf, \
    expose_clf_feats

    args = ArgsTrain().parse()
    print(args)
    pj = f"{args.benchmark.lower()}_{args.scale_factor}"
    debug = False if args.debug == 0 else True
    data_header = f"{args.data_header}/{args.benchmark.lower()}_{args.scale_factor}"
    assert os.path.exists(data_header), f"data not exists at {data_header}"
    assert args.granularity in ("Q", "QS")

    data_params, learning_params, net_params = set_params(args)
    obj, model_name = data_params["obj"], data_params["model_name"]
    assert model_name == "AVGMLP"

    if obj == "tid":
        cats = [f"q{i}" for i in range(1, 23)]
        ncats = len(cats)
    elif obj == "latbuck20":
        cats = [f"b20_{i}" for i in range(18)] + ["others"] # 19 choices
        ncats = len(cats)
    else:
        raise ValueError(obj)

    ckp_header = f"examples/model/spark/ckp/{pj}/{model_name}/{obj}/" \
                 f"{'_'.join([data_params[f] for f in ['ch1_type', 'ch1_cbo', 'ch1_enc', 'ch2', 'ch3', 'ch4']])}"

    op_feats_file = {}
    if data_params["ch1_cbo"] == "on":
        op_feats_file["cbo"] = "cbo_cache.pkl"
    elif data_params["ch1_cbo"] == "on2":
        op_feats_file["cbo"] = "cbo_cache_recollect.pkl"

    if data_params["ch1_enc"] != "off":
        ch1_enc = data_params["ch1_enc"]
        op_feats_file["enc"] = f"enc_cache_{ch1_enc}.pkl"

    ds_dict, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, op_feats_data, _ = expose_data(
        header=data_header,
        tabular_file=f"{'query_level' if args.granularity == 'Q' else 'stage_level'}_cache_data.pkl",
        struct_file="struct_cache.pkl",
        op_feats_file=op_feats_file,
        debug=debug,
        model_name=model_name,
        obj=obj
    )
    if data_params["ch1_cbo"] in ("on", "on2"):
        op_feats_data["cbo"]["l2p"] = L2P_MAP[args.benchmark.lower()]

    data_meta = [ds_dict, op_feats_data, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, ncats]
    model, results, hp_params, hp_prefix_sign = pipeline_classifier(
        data_meta, data_params, learning_params, net_params, ckp_header)
    ckp_path = os.path.join(ckp_header, hp_prefix_sign)

    if os.path.exists(f"{ckp_path}/clf_feats.pkl"):
        print("clf_feats.pkl existed.")
    else:
        print("start generating clf_feats.pkl...")
        net_params, op_groups, picked_groups, picked_cols = augment_net_params(data_params, net_params, data_meta)
        dataset, in_feat_minmax, obj_minmax, tr_loader, val_loader, te_loader = setup_data(
            ds_dict, picked_cols, op_feats_data, col_dict, picked_groups, op_groups,
            dag_dict, struct2template, learning_params, net_params, minmax_dict, coll=collate_clf, train_shuffle=False)
        device = learning_params["device"]
        feat_dict = {}
        for mode, loader in zip(["tr", "val", "te"], [tr_loader, val_loader, te_loader]):
            print(f"start generating feats for split: {mode}")
            feat_dict[mode] = expose_clf_feats(model, loader, device, in_feat_minmax, obj)
        PickleUtils.save(feat_dict, ckp_path, "clf_feats.pkl")
        print(f"{ckp_path}/clf_feats.pkl generated ")



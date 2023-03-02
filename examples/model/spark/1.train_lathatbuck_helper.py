# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: e2e training via GTN
#
# Created at 03/01/2023
if __name__ == "__main__":

    import os
    from model.architecture.avg_mlp import AVGMLP
    from utils.common import JsonUtils, PickleUtils
    from utils.data.feature import L2P_MAP
    from utils.model.args import ArgsTrainLatBuck
    from utils.model.parameters import set_params
    from utils.model.utils import expose_data, analyze_cols, setup_data, collate, evaluate_model

    import torch as th


    args = ArgsTrainLatBuck().parse()
    print(args)
    debug = False if args.debug == 0 else True
    data_header = f"{args.data_header}/{args.benchmark.lower()}_{args.scale_factor}"
    assert os.path.exists(data_header), f"data not exists at {data_header}"
    assert args.granularity in ("Q", "QS")
    assert args.finetune_header is not None
    assert args.bsize == 20

    bsize = args.bsize
    data_params, learning_params, _ = set_params(args)
    obj, model_name = data_params["obj"], data_params["model_name"]
    op_feats_file = {}
    if data_params["ch1_cbo"] == "on":
        op_feats_file["cbo"] = "cbo_cache.pkl"
    elif data_params["ch1_cbo"] == "on2":
        op_feats_file["cbo"] = "cbo_cache_recollect.pkl"

    if data_params["ch1_enc"] != "off":
        ch1_enc = data_params["ch1_enc"]
        op_feats_file["enc"] = f"enc_cache_{ch1_enc}.pkl"
    ds_dict, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, op_feats_data, clf_feat = expose_data(
        header=data_header,
        tabular_file=f"{'query_level' if args.granularity == 'Q' else 'stage_level'}_cache_data.pkl",
        struct_file="struct_cache.pkl",
        op_feats_file=op_feats_file,
        debug=debug,
        model_name=model_name,
        clf_feat_file=data_params["clf_feat"]
    )
    if data_params["ch1_cbo"] in ("on", "on2"):
        op_feats_data["cbo"]["l2p"] = L2P_MAP[args.benchmark.lower()]
    print("data prepared")

    op_groups, picked_groups, picked_cols = analyze_cols(data_params, col_dict)
    ckp_path = args.finetune_header
    device = learning_params["device"]
    hp_params = JsonUtils.load_json(f"{ckp_path}/hp_prefix.json")
    hp_params["op_groups"] = op_groups
    hp_params["n_op_types"] = n_op_types
    hp_params["name"] = model_name
    assert not (data_params["ch1_type"] == "off" and data_params["ch1_cbo"] == "off" and data_params["ch1_enc"] == "off")
    if model_name == "AVGMLP":
        hp_params["ped"] = 8
        model = AVGMLP(hp_params).to(device)
        trained_weights = th.load(f"{ckp_path}/best_weight.pth", map_location=device)["model"]
        model.load_state_dict(trained_weights)
    else:
        raise NotImplementedError
    print("model prepared")

    # data_meta = [ds_dict, op_feats_data, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, clf_feat]
    dataset, in_feat_minmax, obj_minmax, tr_loader, val_loader, te_loader = setup_data(
        ds_dict, picked_cols, op_feats_data, col_dict, picked_groups, op_groups,
        dag_dict, struct2template, learning_params, hp_params, minmax_dict, coll=collate, train_shuffle=False)

    lat_hat_dict = {}
    for mode, loader in zip(["tr", "val", "te"], [tr_loader, val_loader, te_loader]):
        print(f"start generating for {mode}")
        loss, mdict, y, y_hat = evaluate_model(
            model, loader, device, in_feat_minmax, obj_minmax,
            loss_type="wmape", obj=obj, loss_ws=learning_params["loss_ws"], if_y=True)
        lat_hat_dict[mode] = y_hat

    PickleUtils.save(lat_hat_dict, ckp_path, "lat_hat_all.pkl")
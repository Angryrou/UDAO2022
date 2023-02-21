# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 21/02/2023

import os
import numpy as np
import torch as th
from model.architecture.avg_mlp import AVGMLP
from utils.model.utils import expose_data, collate, view_model_param, get_tensor, evaluate_model, MyDSBase
from utils.data.feature import L2P_MAP
from torch.utils.data import DataLoader

cdevice = th.device("cpu")
device = th.device("cuda:0")

# prepare model placeholder

op_groups = ['ch1_type', 'ch1_cbo', 'ch1_enc']
picked_groups = ['ch1', 'ch2', 'ch3', 'ch4', 'obj']
picked_cols = ['sql_struct_id', 'sql_struct_svid', 'qid', 'input_mb', 'input_records', 'input_mb_log',
               'input_records_log', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'k1', 'k2', 'k3',
               'k4', 'k5', 'k6', 'k7', 'k8', 's1', 's2', 's3', 's4', 'latency']
n_op_types = 13
model_name = "AVGMLP"

ori_ckp_header = "examples/model/spark/ckp/tpch_100/AVGMLP/latency/on_on_w2v_on_on_on/4ab746c8c1ddda18"
ori_results = th.load(f"{ori_ckp_header}/results.pth", map_location=cdevice)
ori_weights = th.load(f"{ori_ckp_header}/best_weight.pth", map_location=device)
hp_params = {**ori_results["hp_params"], **{
    "op_groups": op_groups,
    "n_op_types": n_op_types,
    "name": model_name
}}
model = AVGMLP(hp_params).to(device=device)
view_model_param(model_name, model)

# prepare data
op_feats_file = {
    "cbo": "cbo_cache.pkl",
    "enc": "enc_cache_w2v.pkl",
}
data_header="examples/data/spark/cache/tpch_100"

ds_dict_all, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, op_feats_data, clf_feat = expose_data(
    header=data_header,
    tabular_file=f"query_level_cache_data.pkl",
    struct_file="struct_cache.pkl",
    op_feats_file=op_feats_file,
    debug=False,
    model_name=model_name
)
assert clf_feat is None
op_feats_data["cbo"]["l2p"] = L2P_MAP["tpch"]
ds_te_all = ds_dict_all["te"]
ds_te_all.set_format(type="torch", columns=picked_cols)
picked_groups_in_feat = [ch for ch in picked_groups if ch not in ("ch1", "obj")]
in_feat_minmax = {
    "min": th.cat([get_tensor(minmax_dict[ch]["min"].values, device=device) for ch in picked_groups_in_feat]),
    "max": th.cat([get_tensor(minmax_dict[ch]["max"].values, device=device) for ch in picked_groups_in_feat])
}
obj_minmax = {"min": get_tensor(minmax_dict["obj"]["min"].values, device=device),
              "max": get_tensor(minmax_dict["obj"]["max"].values, device=device)}


def pick_weight_path(header):
    signs = os.listdir(header)
    best_sign, best_wmape = None, 1e10
    for sign in signs:
        try:
            wmape = th.load(f"{header}/{sign}/results.pth", map_location=cdevice)["metric_val"]["latency"]["wmape"]
            if wmape < best_wmape:
                best_sign = sign
                best_wmape = wmape
        except:
            print(f"{header}/{sign} is not finished.")
    return f"{header}/{best_sign}/best_weight.pth"


obj = "latency"
loss_type = "wmape"
pre_ckp_header = "examples/model/spark/ckp/tpch_100/AVGMLP/latbuck20/on_on_w2v_on_on_on"

wmape_dict = {}
for pre_sign in ["1fa3dce7e1362141", "c196a193bcd6b8f7", "86d65aceb5cdc4c5", "f2e5be2eec5a5922",
                 "e587ffd7eeb62906", "2980ba32fce02d30", "8426e35c4e1a8836", "164b59ecbb4bda0e",
                 "d0dc97724d2831b1", "eb31ac69f68c4c9f", "67540debaff6aa6c", "75273ab3374d147c",
                 "3e193c2865cecee8", "01cb88866de39fd6", "8ce8a52c7bb254ee", "9d74537f8be6fbbb",
                 "fd3e1e2eca1b8250", "f11a78d35c60f6f6", "16774857e047b9f7", "9ae8ad1415e18bdf",
                 "926c97ca98d9c6fc", "022d2de389d9dddd", "6967b926a203864c", "c992cb3342e58101"]:
    pre_results = th.load(f"{pre_ckp_header}/{pre_sign}/results.pth", map_location=cdevice)
    choice_hat = pre_results["y_te_hat"]
    mask_dict = {i: th.where(choice_hat == i)[0].numpy() for i in range(19)}
    if pre_sign in wmape_dict:
        print(f"{pre_sign} is existed in wmape_dict, with wmape={wmape_dict[pre_sign]:.2f}")
        continue

    print(f"start working on {pre_sign}, with match rate = {pre_results['rate_te']:.2f}")
    lat_list, lat_hat_list = [], []
    for i, mask in mask_dict.items():
        if len(mask) <= 1:
            print(f"buck20x{i if i < 18 else '18+'}, got {len(mask)} data")
            continue
        ds = ds_te_all.select(mask)
        dataset = MyDSBase(ds, op_feats_data, col_dict, picked_groups, op_groups, dag_dict, struct2template, ped=8)
        loader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=collate, num_workers=10)

        # update model weights
        if i < 18:
            weight_path = pick_weight_path(header=f"{ori_ckp_header}/finetune_b20x{i}")
        elif i == 18:
            weight_path = f"{ori_ckp_header}/best_weight.pth"
        else:
            raise ValueError(i)
        model.load_state_dict(th.load(weight_path, map_location=device)["model"])

        loss, m_dict, lat, lat_hat = evaluate_model(
            model, loader, device, in_feat_minmax, obj_minmax, loss_type, obj, None, if_y=True)
        lat_list.append(lat)
        lat_hat_list.append(lat_hat)
        print(f"buck20x{i if i < 18 else '18+'}, got {len(mask)} data, with wmape={m_dict['latency']['wmape']}")

    lat_all, lat_hat_all = np.vstack(lat_list), np.vstack(lat_hat_list)
    wmape = np.abs(lat_all - lat_hat_all).sum() / np.sum(lat_all)
    wmape_dict[pre_sign] = wmape
    print(f"--- WMAPE = {wmape}")
    print()


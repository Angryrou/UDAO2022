# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: support for training.
#
# Created at 03/01/2023
import hashlib
import json
import os
import random
import time

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy import sparse as sp

from datasets import Dataset, DatasetDict
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import pytorch_warmup as warmup
import dgl

from model.architecture.graph_transformer_net import GraphTransformerNet
from model.architecture.mlp_readout_layer import PureMLP
from model.metrics import get_loss
from utils.common import PickleUtils, plot
from utils.model.parameters import OBJ_MAP, ALL_OP_FEATS, DEFAULT_DTYPE, DEFAULT_DEVICE


def get_laplacian_pe(g, pos_enc_dim=16):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = g.adjacency_matrix(transpose=True, scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    return th.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()


def resize_pe(g, pos_enc_dim):
    cur_enc_dim = g.ndata['lap_pe'].shape[1]
    if cur_enc_dim < pos_enc_dim:
        g.ndata['lap_pe'] = F.pad(g.ndata['lap_pe'], pad=(0, pos_enc_dim - cur_enc_dim), mode='constant', value=0)
    else:
        g.ndata['lap_pe'] = g.ndata['lap_pe'][:, :pos_enc_dim]
    return g


def get_random_flips(lap_pe, device):
    sign_flip = th.rand(lap_pe.size(1)).to(device)
    sign_flip[sign_flip >= 0.5] = 1.0
    sign_flip[sign_flip < 0.5] = -1.0
    lap_pe *= sign_flip.unsqueeze(0)
    return lap_pe


def add_pe(model_name, dag_dict):
    for k, dag in dag_dict.items():
        if model_name == "GTN":
            dag.ndata["lap_pe"] = get_laplacian_pe(dgl.to_bidirected(dag), dag.num_nodes() - 2)
        elif model_name == "RAAL":
            raise NotImplementedError
        elif model_name == "QF":
            raise NotImplementedError
        elif model_name == "TL":
            break
        else:
            ValueError(model_name)


def expose_data(header, tabular_file, struct_file, op_feats_file, debug):
    tabular_data = PickleUtils.load(header, tabular_file)
    struct_data = PickleUtils.load(header, struct_file)
    op_feats_data = ...
    col_dict, minmax_dict, dfs = tabular_data["col_dict"], tabular_data["minmax_dict"], tabular_data["dfs"]
    col_groups = ["ch1", "ch2", "ch3", "ch4", "obj"]
    assert set(col_groups) == set(minmax_dict.keys())
    ds_dict = DatasetDict({split: Dataset.from_pandas(df.sample(frac=0.01) if debug else df)
                           for split, df in zip(["tr", "val", "te"], dfs)})
    n_op_types = len(struct_data["global_ops"])
    dag_dict = struct_data["dgl_dict"]

    return ds_dict, col_dict, minmax_dict, dag_dict, n_op_types, op_feats_data


def analyze_cols(data_params, col_dict):
    picked_groups, picked_cols = [], []
    if data_params["ch1_type"] != "off" or data_params["ch1_cbo"] != "off" or data_params["ch1_enc"] != "off":
        picked_groups.append("ch1")
        picked_cols += col_dict["ch1"]
    for ch in ["ch2", "ch3", "ch4", "obj"]:
        if data_params[ch] != "off":
            picked_groups.append(ch)
            picked_cols += col_dict[ch]
    assert "obj" in picked_groups
    op_groups = [ch1_ for ch1_ in ALL_OP_FEATS if data_params[ch1_] != "off"]
    return op_groups, picked_groups, picked_cols


def get_str_hash(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def get_tensor(x, dtype=None, device=None, requires_grad=False, if_sparse=False):
    dtype = DEFAULT_DTYPE if dtype is None else dtype
    device = DEFAULT_DEVICE if device is None else device
    if if_sparse:
        return th.sparse_coo_tensor(x.coords, x.data, x.shape, device=device, dtype=dtype, requires_grad=requires_grad)
    else:
        return th.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def get_hp(data_params, learning_params, net_params, case=""):
    learning_params_list = ["batch_size", "seed", "init_lr", "min_lr", "weight_decay", "epochs", "loss_type"]
    if len(OBJ_MAP[data_params["obj"]]) > 1:
        learning_params_list.append("loss_ws")

    if case == "GTN":
        net_params_list = ["ped", "in_feat_size_op", "in_feat_size_inst", "out_feat_size",
                           "L_gtn", "L_mlp", "n_heads", "hidden_dim", "out_dim", "dropout",
                           "residual", "readout", "batch_norm", "layer_norm"]
    elif case == "MLP":
        net_params_list = ["in_feat_size_inst", "out_feat_size", "L_mlp", "hidden_dim"]
    elif case == "TL":
        net_params_list = ["in_feat_size_op", "in_feat_size_inst", "out_feat_size",
                           "L_mlp", "hidden_dim", "out_dim", "dropout", "readout"]
        raise NotImplementedError(case)
    elif case == "RAAL":
        raise NotImplementedError(case)
    elif case == "QF":
        raise NotImplementedError(case)
    else:
        raise Exception(f"unsupported case {case}")

    for ch1_ in net_params["op_groups"]:
        net_params_list.append(f"{ch1_}_dim")

    param_dict = OrderedDict({**{p: learning_params[p] for p in learning_params_list},
                              **{p: net_params[p] for p in net_params_list}})
    hp_prefix_sign = get_str_hash(json.dumps(param_dict, indent=2))
    if data_params["debug"]:
        hp_prefix_sign += "_debug"
    return param_dict, hp_prefix_sign


def if_pth_existed(pth):
    if isinstance(pth, str):
        return os.path.exists(pth)
    elif isinstance(pth, list):
        return all([if_pth_existed(p) for p in pth])
    else:
        raise Exception(f'pth type {type(pth)} not supported.')


def view_model_param(MODEL_NAME, model):
    total_param = 0
    print("MODEL DETAILS:\n")
    print(model)
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return


def view_data(dataset):
    print("DATA/Train Points: {}, Val Points: {}, Test Points {}"
          .format(dataset["tr"].num_rows, dataset["val"].num_rows, dataset["te"].num_rows))


def collate(samples):
    stages, insts, ys = map(list, zip(*samples))
    insts = th.FloatTensor(insts)
    ys = th.FloatTensor(ys)
    if ys.dim() == 1:
        ys = ys.unsqueeze(1)
    batched_stage = dgl.batch(stages) if stages[0] is not None else None
    return batched_stage, insts, ys


def norm_in_feat_inst(x, minmax):
    return (x - minmax["min"]) / (minmax["max"] - minmax["min"])

def denorm_obj(o, minmax):
    return o * (minmax["max"] - minmax["min"]) + minmax["min"]

class MyDSBase(Dataset):
    def __init__(self, dataset, col_dict, picked_groups, op_groups, dag_dict, ped=1):
        super(MyDSBase, self).__init__(
            arrow_table=dataset._data,
            info=dataset._info,
            split=dataset._split,
            fingerprint=dataset._fingerprint,
            indices_table=dataset._indices
        )
        self._format_type = dataset._format_type
        self._format_kwargs = dataset._format_kwargs
        self._format_columns = dataset._format_columns
        self._output_all_columns = dataset._output_all_columns
        self.dataset = dataset
        self.col_dict = col_dict
        self.picked_groups = picked_groups
        self.op_groups = op_groups
        self.dag_dict = dag_dict
        self.ped = ped

    def __getitem__(self, item):
        x = super(MyDSBase, self).__getitem__(item)
        if "ch1" in self.picked_groups:
            sid = x[self.col_dict["ch1"][0]].item()
            svid = x[self.col_dict["ch1"][1]].item()
            g = self.dag_dict[sid].clone()
            resize_pe(g, self.ped)

            if "ch1_cbo" in self.op_groups:
                # todo: add cbo feats from a data source, and normalize it
                g.nodes["cbo"] = ...
            if "ch1_enc" in self.op_groups:
                # add enc feats from a data source (already normalized)
                g.nodes["enc"] = ...
        else:
            g = None
        inst_feat = []
        for ch in self.picked_groups:
            if ch in ("ch1", "obj"):
                continue
            inst_feat += [x[c] for c in self.col_dict[ch]]
        assert "obj" in self.picked_groups
        y = [x[c] for c in self.col_dict["obj"]]
        return g, inst_feat, y


class TrainStatsTrace():
    def __init__(self, weights_pth_signature):
        self.weights_pth_signature = weights_pth_signature
        self.best_epoch = -1
        self.best_batch = -1
        self.best_loss = float("inf")

    def update(self, model, cur_epoch, cur_batch, cur_loss):
        if cur_loss < self.best_loss:
            self.best_loss = cur_loss
            self.best_epoch = cur_epoch
            self.best_batch = cur_batch
            th.save({
                "model": model.state_dict(),
                "best_epoch": self.best_epoch,
                "best_batch": self.best_batch,
            }, self.weights_pth_signature)

    def pop_model_dict(self, device):
        try:
            ckp_model = th.load(self.weights_pth_signature, map_location=device)
            return ckp_model["model"]
        except:
            raise Exception(f"{self.weights_pth_signature} has not got desire ckp.")


def loss_compute(y, y_hat, loss_type, obj, loss_ws):
    loss_dict = {
        m: get_loss(y[:, i], y_hat[:, i], loss_type)
        for i, m in enumerate(OBJ_MAP[obj])
    }
    if len(loss_dict) == 1:
        loss = loss_dict[OBJ_MAP[obj][0]]
    else:
        loss = sum([loss_ws[k] * l for k, l in loss_dict.items()])
    return loss, loss_dict


def model_out(model, x, in_feat_minmax, obj_minmax, device, mode="train"):
    stage_graph, inst_feat, y = x
    if stage_graph is None:
        batch_insts = inst_feat.to(device)
        batch_y = y.to(device)
        batch_insts = norm_in_feat_inst(batch_insts, in_feat_minmax)
        batch_y_hat = model.forward(batch_insts)
    else:
        batch_stages = stage_graph.to(device)
        batch_insts = inst_feat.to(device)
        batch_y = y.to(device)
        batch_insts = norm_in_feat_inst(batch_insts, in_feat_minmax)
        if model.name == "GTN":
            batch_lap_pos_enc = batch_stages.ndata['lap_pe'].to(device)
            if mode == "train":
                batch_lap_pos_enc = get_random_flips(batch_lap_pos_enc, device)
            batch_y_hat = model.forward(batch_stages, batch_lap_pos_enc, batch_insts)
        elif model.name == "RAAL":
            raise NotImplementedError
        elif model.name == "QF":
            raise NotImplementedError
        elif model.name == "TL":
            raise NotImplementedError
            batch_y_hat = model.forward(batch_stages, device, batch_insts)
        else:
            raise Exception(f"unsupported model_name {model.name}")
    batch_y_hat = denorm_obj(batch_y_hat, obj_minmax)
    return batch_y, batch_y_hat


def get_eval_metrics(y_list, y_hat_list, loss_type, obj, loss_ws, if_y):
    y = th.vstack(y_list)
    y_hat = th.vstack(y_hat_list)
    loss_total, loss_dict = loss_compute(y, y_hat, loss_type, obj, loss_ws)
    metrics_dict = {}
    for i, m in enumerate(OBJ_MAP[obj]):
        loss = loss_dict[m]
        y_i, y_hat_i = y[:, i].detach().cpu().numpy(), y_hat[:, i].detach().cpu().numpy()
        y_err = np.abs(y_i - y_hat_i)
        wmape = (y_err.sum() / y_i.sum()).item()
        y_err_rate = y_err / (y_i + np.finfo(np.float32).eps)
        mape = y_err_rate.mean()
        err_50, err_90, err_95, err_99 = np.percentile(y_err_rate, [50, 90, 95, 99])
        glb_err = np.abs(y_i.sum() - y_hat_i.sum()) / y_hat_i.sum()
        corr, _ = pearsonr(y_i, y_hat_i)
        q_errs = np.maximum(y_i / y_hat_i, y_hat_i / y_i)
        q_err_mean = np.mean(q_errs)
        q_err_50, q_err_90, q_err_95, q_err_99 = np.percentile(q_errs, [50, 90, 95, 99])
        metrics_dict[m] = {
            "loss": loss, "wmape": wmape,
            "mape": mape,
            "err_50": err_50, "err_90": err_90, "err_95": err_95, "err_99": err_99,
            "q_err_mean": q_err_mean,
            "q_err_50": q_err_50, "q_err_90": q_err_90, "q_err_95": q_err_95, "q_err_99": q_err_99,
            "glb_err": glb_err, "corr": corr
        }
    if if_y:
        return loss_total, metrics_dict, y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()
    else:
        return loss_total, metrics_dict


def evaluate_model(model, loader, device, in_feat_minmax, obj_minmax, loss_type, obj, loss_ws, if_y=False):
    assert obj in OBJ_MAP
    model.eval()
    y_all_list = []
    y_hat_all_list = []
    with th.no_grad():
        for batch_idx, x in enumerate(loader):
            batch_y, batch_y_hat = model_out(model, x, in_feat_minmax, obj_minmax, device, mode="eval")
            y_all_list.append(batch_y)
            y_hat_all_list.append(batch_y_hat)
        return get_eval_metrics(y_all_list, y_hat_all_list, loss_type, obj, loss_ws, if_y)


def plot_error_rate(y, y_hat, ckp_path):
    sorted_index = np.argsort(y[:, 0])
    y, y_hat = y[:, 0][sorted_index], y_hat[:, 0][sorted_index]

    x_ids = list(range(len(y)))
    Y_list = [y_hat, y]
    legend_list = ["predicted", "actual"]
    fmts_list = ["g-", "r-"]
    plot(X=x_ids, Y=Y_list, xlabel="insts", ylabel="secs",
         legend=legend_list, yscale="log", figsize=(4.5, 3), fmts=fmts_list)
    plt.tight_layout()
    plt.savefig(f"{ckp_path}/lat.pdf")

    error_rate = np.abs(y - y_hat) / (y + 1e-3) * 100
    plot(X=x_ids, Y=error_rate, xlabel="insts", ylabel="%", figsize=(4.5, 3),
         ylim=[-10, 500], legend=["err_rate"])

    plt.tight_layout()
    plt.savefig(f"{ckp_path}/lat_err.pdf")


def pipeline(data_meta, data_params, learning_params, net_params, ckp_header):
    model_name, obj = data_params["model_name"], data_params["obj"]
    device, loss_type = learning_params["device"], learning_params["loss_type"]
    ds_dict, op_feats_data, col_dict, minmax_dict, dag_dict, n_op_types = data_meta
    op_groups, picked_groups, picked_cols = analyze_cols(data_params, col_dict)
    for ch1 in ALL_OP_FEATS[1:]:
        if ch1 in op_groups:
            assert net_params[f"{ch1}_dim"] == op_feats_data[ch1].shape[1]
    net_params["in_feat_size_inst"] = sum([len(col_dict[ch]) for ch in picked_groups if ch in ["ch2", "ch3", "ch4"]])
    net_params["in_feat_size_op"] = sum([net_params[f"{ch1_}_dim"] for ch1_ in op_groups])
    net_params["out_feat_size"] = len(OBJ_MAP[obj])
    net_params["op_groups"] = op_groups
    net_params["n_op_types"] = n_op_types
    add_pe(model_name, dag_dict)

    if data_params["ch1_type"] == "off" and data_params["ch1_cbo"] == "off" and data_params["ch1_enc"] == "off":
        model = PureMLP(net_params).to(device=device)
        hp_params, hp_prefix_sign = get_hp(data_params, learning_params, net_params, "MLP")
    elif model_name == "GTN":
        model = GraphTransformerNet(net_params).to(device=device)
        hp_params, hp_prefix_sign = get_hp(data_params, learning_params, net_params, "GTN")
    elif model_name == "TL":
        raise NotImplementedError()
    elif model_name == "QF":
        raise NotImplementedError()
    elif model_name == "RAAL":
        raise NotImplementedError()
    else:
        raise ValueError(model_name)

    ckp_path = os.path.join(ckp_header, hp_prefix_sign)
    os.makedirs(ckp_path, exist_ok=True)
    weights_pth_sign = f"{ckp_path}/best_weight.pth"
    results_pth_sign = f"{ckp_path}/results.pth"

    if if_pth_existed(results_pth_sign):
        results = th.load(results_pth_sign, map_location=device)
        print(f"found hps at {results_pth_sign}!")
        print(json.dumps(str({k: v for k, v in results.items() if k != "analyses"}), indent=2))

    print(f"cannot found trained results, start training...")
    print("start preparing data...")

    ds_dict.set_format(type="torch", columns=picked_cols)
    dataset = {split: MyDSBase(ds_dict[split], col_dict, picked_groups, op_groups, dag_dict, net_params["ped"])
               for split in ["tr", "val", "te"]}
    picked_groups_in_feat = [ch for ch in picked_groups if ch not in ("ch1", "obj")]
    in_feat_minmax = {
        "min": th.cat([get_tensor(minmax_dict[ch]["min"].values, device=device) for ch in picked_groups_in_feat]),
        "max": th.cat([get_tensor(minmax_dict[ch]["max"].values, device=device) for ch in picked_groups_in_feat])
    }
    obj_minmax = {"min": get_tensor(minmax_dict["obj"]["min"].values, device=device),
                  "max": get_tensor(minmax_dict["obj"]["max"].values, device=device)}

    view_model_param(model_name, model)
    view_data(dataset)

    tr_loader = DataLoader(dataset["tr"], batch_size=learning_params['batch_size'], shuffle=True,
                           collate_fn=collate, num_workers=learning_params['num_workers'])
    val_loader = DataLoader(dataset["val"], batch_size=learning_params['batch_size'], shuffle=False,
                            collate_fn=collate, num_workers=learning_params['num_workers'])
    te_loader = DataLoader(dataset["te"], batch_size=learning_params['batch_size'], shuffle=False,
                           collate_fn=collate, num_workers=learning_params['num_workers'])

    optimizer = optim.AdamW(model.parameters(), lr=learning_params['init_lr'],
                            weight_decay=learning_params['weight_decay'])
    nbatches = len(tr_loader)
    num_steps = nbatches * learning_params["epochs"]
    lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps,
                                                        eta_min=learning_params["min_lr"])
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    tst = TrainStatsTrace(weights_pth_sign)
    ts = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    log_dir = f"{ckp_header}/{hp_prefix_sign}/{ts}_log"
    writer = SummaryWriter(log_dir=log_dir)
    with open(f"{ckp_header}/{hp_prefix_sign}/hp_prefix.json", "w+") as f:
        f.write(json.dumps(hp_params, indent=2))

    random.seed(learning_params['seed'])
    np.random.seed(learning_params['seed'])
    th.manual_seed(learning_params['seed'])
    if device.type == 'cuda':
        th.cuda.manual_seed(learning_params['seed'])

    loss_ws = learning_params["loss_ws"]
    t0 = time.time()
    try:
        model.train()
        if_break = False
        ckp_start = time.time()
        for epoch in range(learning_params["epochs"]):
            epoch_start_time = time.time()
            for batch_idx, x in enumerate(tr_loader):
                optimizer.zero_grad()
                batch_y, batch_y_hat = model_out(model, x, in_feat_minmax, obj_minmax, device, mode="train")
                loss, loss_dict = loss_compute(batch_y, batch_y_hat, loss_type, obj, loss_ws)
                loss.backward()
                th.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                with warmup_scheduler.dampening():
                    lr_scheduler.step()

                if th.isnan(loss):
                    print("get a nan loss in train")
                    if_break = True
                elif th.isinf(loss):
                    print("get a inf loss in train")
                    if_break = True

                if batch_idx == (nbatches - 1):
                    with th.no_grad():
                        wmape_tr_dict = {
                            m: (th.abs(batch_y[:, i] - batch_y_hat[:, i]).sum() / batch_y[:, i].sum()).detach().item()
                            for i, m in enumerate(OBJ_MAP[obj])
                        }
                        batch_time_tr = (time.time() - ckp_start) / nbatches

                    t1 = time.time()
                    loss_val, m_dict_val = evaluate_model(model, val_loader, device, in_feat_minmax, obj_minmax,
                                                          loss_type, obj, loss_ws, if_y=False)
                    eval_time = time.time() - t1

                    tst.update(model, epoch, batch_idx, loss_val)
                    cur_lr = optimizer.param_groups[0]["lr"]
                    writer.add_scalar("train/_loss", loss.detach().item(), epoch * nbatches + batch_idx)
                    writer.add_scalar("val/_loss", loss_val, epoch * nbatches + batch_idx)
                    for m in OBJ_MAP[obj]:
                        writer.add_scalar(f"train/_wmape_{m}", wmape_tr_dict[m], epoch * nbatches + batch_idx)
                        writer.add_scalar(f"val/_wmape_{m}", m_dict_val[m]["wmape"], epoch * nbatches + batch_idx)
                    writer.add_scalar("learning_rate", cur_lr, epoch * nbatches + batch_idx)
                    writer.add_scalar("batch_time", batch_time_tr, epoch * nbatches + batch_idx)

                    print("Epoch {:03d} | Batch {:06d} | LR: {:.8f} | TR Loss {:.6f} | VAL Loss {:.6f} | "
                          "s/ba {:.3f} | s/eval {:.3f} ".format(
                        epoch, batch_idx, cur_lr, loss.detach().item(), loss_val, batch_time_tr, eval_time))
                    print(" \n ".join([
                        "[{}] TR WMAPE {:.6f} | VAL WAMPE {:.6f} | VAL QErrMean {:.6f} | CORR {:.6f}".format(
                            m, wmape_tr_dict[m], m_dict_val[m]["wmape"], m_dict_val[m]["q_err_mean"],
                            m_dict_val[m]["corr"]
                        ) for m in OBJ_MAP[obj]
                    ]))

                    epoch_time = time.time() - epoch_start_time
                    print(f"Epoch {epoch} cost {epoch_time} s.")
                    print("-" * 89)
                    ckp_start = time.time()
                    model.train()

                if if_break:
                    break

            if if_break:
                break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    total_time = time.time() - t0
    model.load_state_dict(tst.pop_model_dict(device))
    loss_val, m_dict_val = evaluate_model(model, val_loader, device, in_feat_minmax, obj_minmax,
                                          loss_type, obj, loss_ws)
    loss_te, m_dict_te, y_te, y_hat_te = evaluate_model(
        model, te_loader, device, in_feat_minmax, obj_minmax, loss_type, obj, loss_ws, if_y=True)

    results = {
        "hp_params": hp_params,
        "Epoch": tst.best_epoch,
        "Batch": tst.best_batch,
        "Total_time": total_time,
        "timestamp": ts,
        "loss_val": loss_val,
        "loss_te": loss_te,
        "metric_val": m_dict_val,
        "metric_te": m_dict_te,
        "y_te": y_te,
        "y_te_hat": y_hat_te
    }
    th.save(results, results_pth_sign)
    plot_error_rate(y_te, y_hat_te, ckp_path)
    print(json.dumps(str(results), indent=2))

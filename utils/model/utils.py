# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: support for training.
#
# Created at 03/01/2023
import hashlib
import itertools
import json
import os
import random
import time

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

from model.architecture.avg_mlp import AVGMLP
from model.architecture.graph_attention_net import GATv2
from model.architecture.graph_conv_net import GCN
from model.architecture.graph_isomorphism_net import GIN
from model.architecture.graph_transformer_net import GraphTransformerNet
from model.architecture.mlp_readout_layer import PureMLP
from model.metrics import get_loss
from utils.common import PickleUtils, plot
from utils.data.configurations import SparkKnobs, KnobUtils
from utils.data.feature import L2P_MAP
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
    if "lap_pe" not in g.ndata:
        return g
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
        if model_name in ("GTN", "RAAL", "QF", "AVGMLP"):
            dag.ndata["lap_pe"] = get_laplacian_pe(dgl.to_bidirected(dag), dag.num_nodes() - 2)
        elif model_name == "TL":
            break
        else:
            ValueError(model_name)


def expose_data_stage(header, tabular_file, struct_file, data_params, benchmark, debug):
    obj = data_params["obj"]
    tabular_data = PickleUtils.load(header, tabular_file)
    col_dict, minmax_dict = tabular_data["col_dict"], tabular_data["minmax_dict"]
    assert all(o in col_dict["obj"] for o in OBJ_MAP[obj])
    col_dict["obj"] = OBJ_MAP[obj]
    qs2o_dict, qs_dependencies_dict = tabular_data["qs2o_dict"], tabular_data["qs_dependencies_dict"]

    dfs_stage = tabular_data["dfs"]
    ch1_dict = {split: df[col_dict["ch1"][:3]].drop_duplicates().reset_index(drop=True)
                for split, df in zip(["tr", "val", "te"], dfs_stage)}
    df_stage = pd.concat(dfs_stage) \
        .sort_values(["sql_struct_id", "sql_struct_svid", "qs_id"]) \
        .set_index("sql_struct_id")
    sids = df_stage.index.unique().to_list()
    stage_feat = {}
    for sid in sids:
        df_stage_ = df_stage.loc[sid].copy()
        q_num = len(df_stage_.q_sign.unique())
        qs_num = len(df_stage_.qs_id.unique())
        # add placeholder for failed svid
        svids = df_stage_.sql_struct_svid.astype(int).unique()
        missing_svids = [svid_ for svid_ in range(max(svids)) if svid_ not in svids]
        for missing_svid in missing_svids:
            add_df_dict = {c: [None] * qs_num for c in df_stage_.columns}
            add_df_dict["sql_struct_svid"] = [missing_svid] * qs_num
            add_df_dict["qs_id"] = list(range(qs_num))
            df_stage_ = pd.concat([df_stage_, pd.DataFrame(add_df_dict, index=[sid] * qs_num)])
            q_num += 1
        df_stage_ = df_stage_.sort_values(["sql_struct_svid", "qs_id"])
        stage_feat[sid] = {ch_: df_stage_[col_dict[ch_]].values.reshape(q_num, qs_num, -1)
                           for ch_ in ["ch2", "ch3", "ch4", "obj"]}
    ds_dict = DatasetDict({k: Dataset.from_pandas(v.sample(frac=0.01) if debug else v) for k, v in ch1_dict.items()})

    struct_data = PickleUtils.load(header, struct_file)
    struct2template = {v["sql_struct_id"]: v["template"] for v in struct_data["struct_dict"].values()}
    dag_dict = struct_data["dgl_dict"]
    op_feats_file = {}
    if data_params["ch1_cbo"] == "on":
        op_feats_file["cbo"] = "cbo_cache.pkl"
    elif data_params["ch1_cbo"] == "on2":
        op_feats_file["cbo"] = "cbo_cache_recollect.pkl"
    if data_params["ch1_enc"] != "off":
        ch1_enc = data_params["ch1_enc"]
        op_feats_file["enc"] = f"enc_cache_{ch1_enc}.pkl"
    op_feats_data = {k: PickleUtils.load(header, v) for k, v in op_feats_file.items()}
    if data_params["ch1_cbo"] in ("on", "on2"):
        op_feats_data["cbo"]["l2p"] = L2P_MAP[benchmark.lower()]

    dag_misc = [dag_dict, qs2o_dict, op_feats_data, struct2template, qs_dependencies_dict]
    n_op_types = len(struct_data["global_ops"])
    return ds_dict, dag_misc, stage_feat, col_dict, minmax_dict, n_op_types


def expose_data(header, tabular_file, struct_file, op_feats_file, debug, ori=False,
                model_name="GTN", obj="latency", clf_feat_file=None):
    tabular_data = PickleUtils.load(header, tabular_file)
    struct_data = PickleUtils.load(header, struct_file)
    op_feats_data = {k: PickleUtils.load(header, v) for k, v in op_feats_file.items()}
    col_dict, minmax_dict, dfs = tabular_data["col_dict"], tabular_data["minmax_dict"], tabular_data["dfs"]
    col_dict["obj"] = OBJ_MAP[obj]
    col_groups = ["ch1", "ch2", "ch3", "ch4", "obj"]
    assert set(col_groups) == set(minmax_dict.keys())
    df_dict = {}
    if obj in ("latency"):
        df_dict = {split: (df.sample(frac=0.01) if debug else df) for split, df in zip(["tr", "val", "te"], dfs)}
    elif obj == "latbuck20":
        for split, df in zip(["tr", "val", "te"], dfs):
            df_ = df.sample(frac=0.01) if debug else df
            df_[obj] = df_["latency"] // 20
            df_.loc[df_[obj] >= 18, obj] = 18
            df_dict[split] = df_
    elif obj == "tid":
        for split, df in zip(["tr", "val", "te"], dfs):
            df_ = df.sample(frac=0.01) if debug else df
            df_[obj] = df_["template"].apply(lambda a: int(a[1:]) - 1)
            df_dict[split] = df_
    else:
        raise ValueError(obj)
    ds_dict = DatasetDict({k: Dataset.from_pandas(v) for k, v in df_dict.items()})

    if model_name in ("GTN", "AVGMLP"):
        dag_dict = struct_data["dgl_dict"]
        add_pe(model_name, dag_dict)
    elif model_name == "RAAL":
        dag_dict = struct_data["dgl_dict"]
        add_pe(model_name, dag_dict)
        non_siblings_map = PickleUtils.load(header, "raal_dgl.pkl")
        dag_dict["non_siblings_map"] = non_siblings_map
    elif model_name == "QF":
        dag_dict_ori = struct_data["dgl_dict"]
        add_pe(model_name, dag_dict_ori)
        dag_dict = PickleUtils.load(header, "qf_dgl.pkl")
        for i, g in dag_dict.items():
            dag_dict[i].ndata["lap_pe"] = dag_dict_ori[i].ndata["lap_pe"]
        max_dist = max([v.edata["dist"].max().item() for v in dag_dict.values()])
        dag_dict["max_dist"] = max_dist
    elif model_name in ("GCN", "GATv2", "GIN"):
        dag_dict = struct_data["dgl_dict"]
        dag_dict = {k: dgl.add_self_loop(dag) for k, dag in dag_dict.items()}
    else:
        raise ValueError(model_name)
    n_op_types = len(struct_data["global_ops"])
    struct2template = {v["sql_struct_id"]: v["template"] for v in struct_data["struct_dict"].values()}

    clf_feat = None
    if clf_feat_file is not None:
        clf_feat = PickleUtils.load_file(clf_feat_file)
        clf_feat_mask = clf_feat["tr"].std(0) > 0
        clf_feat = {k: v[:, clf_feat_mask] for k, v in clf_feat.items()}
    if ori:
        return dfs, ds_dict, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, op_feats_data, clf_feat
    else:
        return ds_dict, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, op_feats_data, clf_feat


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


def get_tensor(x: object, dtype: object = None, device: object = None, requires_grad: object = False,
               if_sparse: object = False) -> object:
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

    if case in ("GTN", "RAAL", "QF"):
        net_params_list = ["ped", "in_feat_size_op", "in_feat_size_inst", "out_feat_size", "L_gtn", "L_mlp",
                           "n_heads", "hidden_dim", "out_dim", "mlp_dim", "dropout", "dropout2",
                           "residual", "readout", "batch_norm", "layer_norm"]
        if "out_norm" in net_params and net_params["out_norm"] is not None:
            net_params_list.append("out_norm")
    elif case == "MLP":
        net_params_list = ["in_feat_size_inst", "out_feat_size", "L_mlp", "hidden_dim", "mlp_dim", "dropout2"]
    elif case == "AVGMLP":
        net_params_list = ["in_feat_size_op", "in_feat_size_inst", "out_feat_size", "L_mlp", "out_dim",
                           "mlp_dim", "dropout2"]
        if "out_norm" in net_params and net_params["out_norm"] is not None:
            net_params_list.append("out_norm")
        if "agg_dim" in net_params and net_params["agg_dim"] is not None:
            net_params_list.append("agg_dim")
    elif case in ["GCN", "GIN"]:
        net_params_list = ["in_feat_size_op", "in_feat_size_inst", "out_feat_size", "L_gtn", "L_mlp",
                           "hidden_dim", "out_dim", "mlp_dim", "dropout2", "readout"]
        if "agg_dim" in net_params and net_params["agg_dim"] is not None:
            net_params_list.append("agg_dim")
    elif case == "GATv2":
        net_params_list = ["in_feat_size_op", "in_feat_size_inst", "out_feat_size", "L_gtn", "L_mlp", "n_heads",
                           "hidden_dim", "out_dim", "mlp_dim", "dropout", "dropout2", "residual", "readout",
                           "layer_norm"]
        if "agg_dim" in net_params and net_params["agg_dim"] is not None:
            net_params_list.append("agg_dim")
    elif case == "TL":
        net_params_list = ["in_feat_size_op", "in_feat_size_inst", "out_feat_size",
                           "L_mlp", "hidden_dim", "out_dim", "mlp_dim", "dropout", "dropout2", "readout"]
        raise NotImplementedError(case)
    else:
        raise Exception(f"unsupported case {case}")
    if case == "QF":
        net_params_list += ["max_dist"]

    for ch1_ in net_params["op_groups"]:
        net_params_list.append(f"{ch1_}_dim")

    data_params_list = ["clf_feat"] if data_params["clf_feat"] is not None else []
    param_dict = OrderedDict({**{p: data_params[p] for p in data_params_list},
                              **{p: learning_params[p] for p in learning_params_list},
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


def collate_clf(samples):
    stages, insts, ys = map(list, zip(*samples))
    insts = th.FloatTensor(insts)
    ys = th.FloatTensor(ys).to(th.long)
    if ys.dim() == 1:
        ys = ys.unsqueeze(1)
    batched_stage = dgl.batch(stages) if stages[0] is not None else None
    return batched_stage, insts, ys


def collate_stage(samples):
    stages, insts, ys = map(list, zip(*samples))
    stages = list(itertools.chain.from_iterable(stages))
    insts = np.vstack(insts).astype(float)
    ys = np.vstack(ys).astype(float)
    if np.isnan(ys).any():
        raise Exception(f"errors in manipulate data")
    insts = th.FloatTensor(insts)
    ys = th.FloatTensor(ys)
    assert len(stages) == len(insts) and len(insts) == len(ys)
    batched_stage = dgl.batch(stages) if stages[0] is not None else None
    return batched_stage, insts, ys


def norm_in_feat_inst(x, minmax):
    return (x - minmax["min"]) / (minmax["max"] - minmax["min"])


def denorm_obj(o, minmax):
    return o * (minmax["max"] - minmax["min"]) + minmax["min"]


def form_graph(dag_dict, sid, svid, qid, ped, op_groups, op_feats, struct2template):  # to add the enc source
    g = dag_dict[sid].clone()
    resize_pe(g, ped)
    g.ndata["sid"] = get_tensor([sid] * g.num_nodes(), dtype=th.int)
    if "ch1_cbo" in op_groups:
        # todo: add cbo feats from a data source, and normalize it
        l2p = op_feats["cbo"]["l2p"][sid]
        # qid index from 1
        lp_feat = op_feats["cbo"]["ofeat_dict"][struct2template[sid]][qid - 1]
        pp_feat = lp_feat[l2p]
        minmax = op_feats["cbo"]["minmax"]
        pp_feat_norm = norm_in_feat_inst(pp_feat, minmax)
        g.ndata["cbo"] = get_tensor(pp_feat_norm)
    if "ch1_enc" in op_groups:
        # add enc feats from a data source (already normalized)
        enc = op_feats["enc"]["op_encs"][op_feats["enc"]["oid_dict"][sid][int(svid)]]
        g.ndata["enc"] = get_tensor(enc)
    return g


def prepare_data_for_opt(df, q_sign, dag_dict, ped, op_groups, op_feats, struct2template,
                         model_proxy, col_dict, minmax_dict):
    record = df[df["q_sign"] == q_sign]
    sid, svid, qid = record.sql_struct_id[0], record.sql_struct_svid[0], record.qid[0]
    g = form_graph(dag_dict, sid, svid, qid, ped, op_groups, op_feats, struct2template)
    stage_emb = model_proxy.get_stage_emb(g, fmt="numpy")
    ch2_norm = norm_in_feat_inst(record[col_dict["ch2"]], minmax_dict["ch2"]).values
    ch3_norm = np.zeros((1, len(col_dict["ch3"])))  # as like in the idle env
    return stage_emb, ch2_norm, ch3_norm


def get_sample_spark_knobs(knobs, n_samples, bm, q_sign, seed):
    np.random.seed(seed)
    samples = np.random.rand(n_samples, len(knobs))
    if bm == "tpch" and q_sign.split("-")[0] == "q14":
        samples[:, -2] = 0
    else:
        samples[:, -2] = 1  # always set s3 (autoBroadcastJoinThreshold) as the largest (320M in our case)
    knob_df = KnobUtils.knob_denormalize(samples, knobs)
    knob_df = knob_df.drop_duplicates()
    knob_df.index = knob_df.apply(lambda x: KnobUtils.knobs2sign(x, knobs), axis=1)
    conf_norm = KnobUtils.knob_normalize(knob_df, knobs)
    return knob_df, conf_norm


class MyDSBase(Dataset):
    def __init__(self, dataset, op_feats, col_dict, picked_groups, op_groups, dag_dict, struct2template,
                 ped=1, clf_feat=None):
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
        self.op_feats = op_feats
        self.col_dict = col_dict
        self.picked_groups = picked_groups
        self.op_groups = op_groups
        self.dag_dict = dag_dict
        self.struct2template = struct2template
        self.ped = ped
        self.clf_feat = None if clf_feat is None else clf_feat.tolist()

    def __getitem__(self, item):
        x = super(MyDSBase, self).__getitem__(item)
        if "ch1" in self.picked_groups:
            sid = x[self.col_dict["ch1"][0]].item()
            svid = x[self.col_dict["ch1"][1]].item()
            qid = x[self.col_dict["ch1"][2]].item()
            g = form_graph(self.dag_dict, sid, svid, qid, self.ped, self.op_groups, self.op_feats, self.struct2template)
        else:
            g = None
        inst_feat = []
        for ch in self.picked_groups:
            if ch in ("ch1", "obj"):
                continue
            inst_feat += [x[c] for c in self.col_dict[ch]]
        if self.clf_feat is not None:
            inst_feat += self.clf_feat[item]

        assert "obj" in self.picked_groups
        y = [x[c] for c in self.col_dict["obj"]]
        return g, inst_feat, y


class MyStageDSBase(Dataset):
    def __init__(self, dataset, dag_misc, stage_feat, col_dict, picked_groups, op_groups, model_name, ped=1):
        super(MyStageDSBase, self).__init__(
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
        dag_dict, qs2o_dict, op_feats, struct2template, _ = dag_misc

        self.dataset = dataset
        self.dag_dict = dag_dict
        self.qs2o_dict = qs2o_dict
        self.op_feats = op_feats
        self.struct2template = struct2template

        self.stage_feat = stage_feat
        self.col_dict = col_dict
        self.picked_groups = picked_groups
        self.op_groups = op_groups
        self.model_name = model_name
        self.ped = ped

    def __getitem__(self, item):
        x = super(MyStageDSBase, self).__getitem__(item)
        sid = x[self.col_dict["ch1"][0]].item()
        svid = int(x[self.col_dict["ch1"][1]].item())
        qid = x[self.col_dict["ch1"][2]].item()

        if "ch1" in self.picked_groups:
            g = form_graph(self.dag_dict, sid, svid, qid, self.ped, self.op_groups, self.op_feats, self.struct2template)
            g_stages = []
            for qs_id in range(len(self.qs2o_dict[sid])):
                g_stage = g.subgraph(self.qs2o_dict[sid][qs_id])
                if self.model_name in ("GTN"):
                    g_stage.ndata["lap_pe"] = get_laplacian_pe(dgl.to_bidirected(g_stages), g_stage.num_nodes() - 2)
                    resize_pe(g_stage, self.ped)
                elif self.model_name in ("GCN", "GATv2", "GIN"):
                    g_stage = dgl.add_self_loop(g_stage)
                else:
                    raise ValueError(self.model_name)
                g_stages.append(g_stage)
        else:
            g_stages = [None]

        inst_feat = []
        for ch in self.picked_groups:
            if ch in ("ch1", "obj"):
                continue
            inst_feat.append(self.stage_feat[sid][ch][svid])
        inst_feat = np.hstack(inst_feat)
        y = self.stage_feat[sid]["obj"][svid]
        return g_stages, inst_feat, y


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
        if len(in_feat_minmax) > 0:
            batch_insts = norm_in_feat_inst(batch_insts, in_feat_minmax)
        if model.name in ("GTN", "RAAL", "QF"):
            batch_lap_pos_enc = batch_stages.ndata['lap_pe'].to(device)
            if mode == "train":
                batch_lap_pos_enc = get_random_flips(batch_lap_pos_enc, device)
            batch_y_hat = model.forward(batch_stages, batch_lap_pos_enc, batch_insts)
        elif model.name in ("AVGMLP", "GCN", "GATv2", "GIN"):
            batch_y_hat = model.forward(batch_stages, batch_insts)
        elif model.name == "TL":
            raise NotImplementedError
            batch_y_hat = model.forward(batch_stages, device, batch_insts)
        else:
            raise Exception(f"unsupported model_name {model.name}")
    batch_y_hat = denorm_obj(batch_y_hat, obj_minmax)
    return batch_y, batch_y_hat


def model_out_clf(model, x, in_feat_minmax, device):
    stage_graph, inst_feat, y = x
    assert stage_graph is not None
    batch_stages = stage_graph.to(device)
    batch_insts = inst_feat.to(device)
    batch_y = y.to(device)
    batch_insts = norm_in_feat_inst(batch_insts, in_feat_minmax)
    assert model.name == "AVGMLP"
    batch_y_hat = model.forward(batch_stages, batch_insts, mode="clf")
    return batch_y, batch_y_hat


def model_out_clf_feat(model, x, in_feat_minmax, device):
    stage_graph, inst_feat, y = x
    assert stage_graph is not None
    batch_stages = stage_graph.to(device)
    batch_insts = inst_feat.to(device)
    batch_insts = norm_in_feat_inst(batch_insts, in_feat_minmax)
    assert model.name == "AVGMLP"
    clf_feat = model.forward(batch_stages, batch_insts, mode="clf", out="-2")
    return clf_feat


def get_eval_metrics(y_list, y_hat_list, loss_type, obj, loss_ws, if_y):
    y = th.vstack(y_list)
    y_hat = th.vstack(y_hat_list)
    loss_total, loss_dict = loss_compute(y, y_hat, loss_type, obj, loss_ws)
    metrics_dict = {}
    for i, m in enumerate(OBJ_MAP[obj]):
        loss = loss_dict[m].item()
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


def evaluate_model_clf(model, loader, device, in_feat_minmax, obj, if_y=False):
    assert obj in OBJ_MAP
    model.eval()
    y_all_list = []
    y_hat_all_list = []
    with th.no_grad():
        for batch_idx, x in enumerate(loader):
            batch_y, batch_y_hat = model_out_clf(model, x, in_feat_minmax, device)
            y_all_list.append(batch_y)
            y_hat_all_list.append(batch_y_hat)
        y = th.vstack(y_all_list)
        y_hat_flat = th.vstack(y_hat_all_list)
        loss = get_loss(y.squeeze(), y_hat_flat, loss_type="nll")

        y_hat = y_hat_flat.max(1, keepdim=True)[1]
        rate = y_hat.eq(y.view_as(y_hat)).sum().item() / len(y)
        if if_y:
            return loss, rate, y, y_hat, y_hat_flat
        return loss, rate


def expose_clf_feats(model, loader, device, in_feat_minmax, obj):
    assert obj in OBJ_MAP
    model.eval()
    feat_list = []
    with th.no_grad():
        for batch_idx, x in enumerate(loader):
            batch_feat = model_out_clf_feat(model, x, in_feat_minmax, device).cpu().numpy()
            feat_list.append(batch_feat)
            if (batch_idx + 1) % 10 == 0:
                print(f"{batch_idx + 1}/{len(loader)} batches finished.")
        feats = np.vstack(feat_list)
        return feats


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


def show_results(results, obj):
    if obj in ("latency"):
        print(json.dumps(str(results), indent=2))
        print("\n".join([
            "[{}-{}] WMAPE {:.4f} | MAPE {:.4f}| ERR-50,90,95,99 {:.4f},{:.4f},{:.4f},{:.4f} | "
            "QErr-Mean {:.4f} | QERR-50,90,95,99 ,{:.4f},{:.4f},{:.4f},{:.4f} | CORR {:.4f}".format(
                m, split, results[f"metric_{split}"][m]["wmape"],
                *[results[f"metric_{split}"][m][t] for t in ["mape", "err_50", "err_90", "err_95", "err_99"]],
                *[results[f"metric_{split}"][m][t] for t in
                  ["q_err_mean", "q_err_50", "q_err_90", "q_err_95", "q_err_99"]],
                results[f"metric_{split}"][m]["corr"]
            ) for m in OBJ_MAP[obj] for split in ["val", "te"]
        ]))
    elif obj in ("latbuck20", "tid"):
        print(json.dumps(str(results), indent=2))
        print("[VAL] | Loss: {:.4f} | Rate: {:.4f}".format(results["loss_val"], results["rate_val"]))
        print("[TE ] | Loss: {:.4f} | Rate: {:.4f}".format(results["loss_te"], results["rate_te"]))
    else:
        ValueError(obj)


def augment_net_params(data_params, net_params, col_dict, n_op_types):
    op_groups, picked_groups, picked_cols = analyze_cols(data_params, col_dict)
    net_params["in_feat_size_inst"] = sum([len(col_dict[ch]) for ch in picked_groups if ch in ["ch2", "ch3", "ch4"]])
    net_params["in_feat_size_op"] = sum([net_params[f"{ch1_}_dim"] for ch1_ in op_groups])
    net_params["out_feat_size"] = len(OBJ_MAP[data_params["obj"]])
    net_params["op_groups"] = op_groups
    net_params["n_op_types"] = n_op_types
    return net_params, op_groups, picked_groups, picked_cols


def setup_model_and_hp(data_params, learning_params, net_params, ckp_header, dag_dict, finetune_header):
    device = learning_params["device"]
    model_name, obj = data_params["model_name"], data_params["obj"]

    if data_params["ch1_type"] == "off" and data_params["ch1_cbo"] == "off" and data_params["ch1_enc"] == "off":
        model = PureMLP(net_params).to(device=device)
        hp_params, hp_prefix_sign = get_hp(data_params, learning_params, net_params, "MLP")
    elif model_name == "GTN":
        net_params["name"] = model_name
        model = GraphTransformerNet(net_params).to(device=device)
        hp_params, hp_prefix_sign = get_hp(data_params, learning_params, net_params, model_name)
    elif model_name == "RAAL":
        assert "non_siblings_map" in dag_dict
        net_params["name"] = model_name
        net_params["non_siblings_map"] = dag_dict["non_siblings_map"]
        model = GraphTransformerNet(net_params).to(device=device)
        hp_params, hp_prefix_sign = get_hp(data_params, learning_params, net_params, model_name)
    elif model_name == "QF":
        assert "max_dist" in dag_dict
        net_params["name"] = model_name
        net_params["max_dist"] = dag_dict["max_dist"]
        model = GraphTransformerNet(net_params).to(device=device)
        hp_params, hp_prefix_sign = get_hp(data_params, learning_params, net_params, model_name)
    elif model_name == "AVGMLP":
        model = AVGMLP(net_params).to(device=device)
        hp_params, hp_prefix_sign = get_hp(data_params, learning_params, net_params, "AVGMLP")
    elif model_name == "GCN":
        net_params["name"] = model_name
        model = GCN(net_params).to(device=device)
        hp_params, hp_prefix_sign = get_hp(data_params, learning_params, net_params, "GCN")
    elif model_name == "GIN":
        net_params["name"] = model_name
        model = GIN(net_params).to(device=device)
        hp_params, hp_prefix_sign = get_hp(data_params, learning_params, net_params, "GIN")
    elif model_name == "GATv2":
        net_params["name"] = model_name
        model = GATv2(net_params).to(device=device)
        hp_params, hp_prefix_sign = get_hp(data_params, learning_params, net_params, "GATv2")
    elif model_name == "TL":
        raise NotImplementedError()
    else:
        raise ValueError(model_name)

    view_model_param(model_name, model)
    ckp_path = os.path.join(ckp_header, hp_prefix_sign)
    os.makedirs(ckp_path, exist_ok=True)
    weights_pth_sign = f"{ckp_path}/best_weight.pth"
    results_pth_sign = f"{ckp_path}/results.pth"

    if (not data_params["debug"]) and if_pth_existed(results_pth_sign) and if_pth_existed(weights_pth_sign):
        results = th.load(results_pth_sign, map_location=device)
        print(f"found hps at {results_pth_sign}!")
        show_results(results, obj)
        model.load_state_dict(th.load(weights_pth_sign, map_location=device)["model"])
        return True, (model, results, hp_params, hp_prefix_sign)

    if finetune_header is not None:
        trained_weights = th.load(f"{finetune_header}/best_weight.pth", map_location=device)["model"]
        model.load_state_dict(trained_weights)

    print(f"cannot found trained results, start training...")
    return False, (model, hp_params, hp_prefix_sign, ckp_path, model_name, obj, weights_pth_sign, results_pth_sign)


def setup_data(ds_dict, picked_cols, op_feats_data, col_dict, picked_groups, op_groups, dag_dict, struct2template,
               learning_params, net_params, minmax_dict, coll, train_shuffle=True, clf_feat=None):
    device = learning_params["device"]
    print("start preparing data...")
    ds_dict.set_format(type="torch", columns=picked_cols)
    dataset = {split: MyDSBase(ds_dict[split], op_feats_data, col_dict, picked_groups, op_groups,
                               dag_dict, struct2template, net_params["ped"],
                               clf_feat=clf_feat[split] if clf_feat is not None else None)
               for split in ["tr", "val", "te"]}
    picked_groups_in_feat = [ch for ch in picked_groups if ch not in ("ch1", "obj")]
    if len(picked_groups_in_feat) == 0:
        in_feat_minmax = {}
    else:
        in_feat_minmax = {
            "min": th.cat([get_tensor(minmax_dict[ch]["min"].values, device=device) for ch in picked_groups_in_feat]),
            "max": th.cat([get_tensor(minmax_dict[ch]["max"].values, device=device) for ch in picked_groups_in_feat])
        }
    if clf_feat is not None:
        clf_feat_minmax = {"min": clf_feat["tr"].min(0), "max": clf_feat["tr"].max(0)}
        for mm in ["min", "max"]:
            in_feat_minmax[mm] = th.cat([in_feat_minmax[mm], get_tensor(clf_feat_minmax[mm], device=device)])

    obj_minmax = {"min": get_tensor(minmax_dict["obj"]["min"].values, device=device),
                  "max": get_tensor(minmax_dict["obj"]["max"].values, device=device)}

    tr_loader = DataLoader(dataset["tr"], batch_size=learning_params['batch_size'], shuffle=train_shuffle,
                           collate_fn=coll, num_workers=learning_params['num_workers'])
    val_loader = DataLoader(dataset["val"], batch_size=learning_params['batch_size'], shuffle=False,
                            collate_fn=coll, num_workers=learning_params['num_workers'])
    te_loader = DataLoader(dataset["te"], batch_size=learning_params['batch_size'], shuffle=False,
                           collate_fn=coll, num_workers=learning_params['num_workers'])

    return dataset, in_feat_minmax, obj_minmax, tr_loader, val_loader, te_loader


def setup_data_stage(ds_dict, dag_misc, stage_feat, col_dict, picked_groups, op_groups, learning_params, net_params,
                     minmax_dict, model_name, train_shuffle=True):
    device = learning_params["device"]
    print("start preparing data...")
    ds_dict.set_format(type="torch")  # ['sql_struct_id', 'sql_struct_svid', 'qid']
    dataset = {split: MyStageDSBase(ds_dict[split], dag_misc, stage_feat, col_dict, picked_groups, op_groups,
                                    model_name, net_params["ped"])
               for split in ["tr", "val", "te"]}

    picked_groups_in_feat = [ch for ch in picked_groups if ch not in ("ch1", "obj")]
    if len(picked_groups_in_feat) == 0:
        in_feat_minmax = {}
    else:
        in_feat_minmax = {
            "min": th.cat([get_tensor(minmax_dict[ch]["min"].values, device=device) for ch in picked_groups_in_feat]),
            "max": th.cat([get_tensor(minmax_dict[ch]["max"].values, device=device) for ch in picked_groups_in_feat])
        }
    obj_minmax = {"min": get_tensor(minmax_dict["obj"]["min"].values, device=device),
                  "max": get_tensor(minmax_dict["obj"]["max"].values, device=device)}

    tr_loader = DataLoader(dataset["tr"], batch_size=learning_params['batch_size'], shuffle=train_shuffle,
                           collate_fn=collate_stage, num_workers=learning_params['num_workers'])
    val_loader = DataLoader(dataset["val"], batch_size=learning_params['batch_size'], shuffle=False,
                            collate_fn=collate_stage, num_workers=learning_params['num_workers'])
    te_loader = DataLoader(dataset["te"], batch_size=learning_params['batch_size'], shuffle=False,
                           collate_fn=collate_stage, num_workers=learning_params['num_workers'])

    return dataset, in_feat_minmax, obj_minmax, tr_loader, val_loader, te_loader


def setup_train(weights_pth_sign, model, learning_params, nbatches, ckp_header, hp_prefix_sign, hp_params):
    tst = TrainStatsTrace(weights_pth_sign)
    optimizer = optim.AdamW(model.parameters(), lr=learning_params['init_lr'],
                            weight_decay=learning_params['weight_decay'])
    num_steps = nbatches * learning_params["epochs"]
    lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps,
                                                           eta_min=learning_params["min_lr"])
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    ts = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    log_dir = f"{ckp_header}/{hp_prefix_sign}/{ts}_log"
    writer = SummaryWriter(log_dir=log_dir)
    with open(f"{ckp_header}/{hp_prefix_sign}/hp_prefix.json", "w+") as f:
        f.write(json.dumps(hp_params, indent=2))

    random.seed(learning_params['seed'])
    np.random.seed(learning_params['seed'])
    th.manual_seed(learning_params['seed'])
    if learning_params["device"].type == 'cuda':
        th.cuda.manual_seed(learning_params['seed'])

    loss_ws = learning_params["loss_ws"]
    return ts, tst, optimizer, lr_scheduler, warmup_scheduler, writer, loss_ws


def pipeline(data_meta, data_params, learning_params, net_params, ckp_header, finetune_header=None):
    ds_dict, op_feats_data, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, clf_feat = data_meta
    device, loss_type = learning_params["device"], learning_params["loss_type"]
    net_params, op_groups, picked_groups, picked_cols = augment_net_params(data_params, net_params, col_dict,
                                                                           n_op_types)
    if clf_feat is not None:
        assert "tr" in clf_feat
        clf_feat_dim = clf_feat["tr"].shape[1]
        net_params["in_feat_size_inst"] += clf_feat_dim

    # model setup
    exist, ret = setup_model_and_hp(data_params, learning_params, net_params, ckp_header, dag_dict, finetune_header)
    if exist:
        return ret
    model, hp_params, hp_prefix_sign, ckp_path, model_name, obj, weights_pth_sign, results_pth_sign = ret

    # data setup
    dataset, in_feat_minmax, obj_minmax, tr_loader, val_loader, te_loader = setup_data(
        ds_dict, picked_cols, op_feats_data, col_dict, picked_groups, op_groups,
        dag_dict, struct2template, learning_params, net_params, minmax_dict, coll=collate, clf_feat=clf_feat)

    view_model_param(model_name, model)
    view_data(dataset)

    # training setup
    nbatches = len(tr_loader)
    ts, tst, optimizer, lr_scheduler, warmup_scheduler, writer, loss_ws = \
        setup_train(weights_pth_sign, model, learning_params, nbatches, ckp_header, hp_prefix_sign, hp_params)

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
    show_results(results, obj)
    return model, results, hp_params, hp_prefix_sign


def pipeline_stage(data_meta, data_params, learning_params, net_params, ckp_header):
    ds_dict, dag_misc, stage_feat, col_dict, minmax_dict, n_op_types = data_meta
    dag_dict = dag_misc[0]
    device, loss_type = learning_params["device"], learning_params["loss_type"]
    net_params, op_groups, picked_groups, _ = augment_net_params(data_params, net_params, col_dict, n_op_types)

    # model setup
    exist, ret = setup_model_and_hp(data_params, learning_params, net_params, ckp_header, dag_dict, None)
    if exist:
        return ret
    model, hp_params, hp_prefix_sign, ckp_path, model_name, obj, weights_pth_sign, results_pth_sign = ret

    # data setup
    dataset, in_feat_minmax, obj_minmax, tr_loader, val_loader, te_loader = setup_data_stage(
        ds_dict, dag_misc, stage_feat, col_dict, picked_groups, op_groups, learning_params,
        net_params, minmax_dict, model_name)

    view_model_param(model_name, model)
    view_data(dataset)

    # training setup
    nbatches = len(tr_loader)
    ts, tst, optimizer, lr_scheduler, warmup_scheduler, writer, loss_ws = \
        setup_train(weights_pth_sign, model, learning_params, nbatches, ckp_header, hp_prefix_sign, hp_params)

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
    show_results(results, obj)
    return model, results, hp_params, hp_prefix_sign


def pipeline_classifier(data_meta, data_params, learning_params, net_params, ckp_header):
    ds_dict, op_feats_data, col_dict, minmax_dict, dag_dict, n_op_types, struct2template, ncats = data_meta
    device, loss_type = learning_params["device"], learning_params["loss_type"]
    net_params, op_groups, picked_groups, picked_cols = augment_net_params(
        data_params, net_params, col_dict, n_op_types)
    net_params["out_feat_size"] = ncats

    # model setup
    exist, ret = setup_model_and_hp(data_params, learning_params, net_params, ckp_header, dag_dict, None)
    if exist:
        return ret
    model, hp_params, hp_prefix_sign, ckp_path, model_name, obj, weights_pth_sign, results_pth_sign = ret

    # data setup
    dataset, in_feat_minmax, obj_minmax, tr_loader, val_loader, te_loader = setup_data(
        ds_dict, picked_cols, op_feats_data, col_dict, picked_groups, op_groups,
        dag_dict, struct2template, learning_params, net_params, minmax_dict, coll=collate_clf)

    # view_model_param(model_name, model)
    view_data(dataset)

    # training setup
    nbatches = len(tr_loader)
    ts, tst, optimizer, lr_scheduler, warmup_scheduler, writer, loss_ws = \
        setup_train(weights_pth_sign, model, learning_params, nbatches, ckp_header, hp_prefix_sign, hp_params)

    t0 = time.time()
    try:
        model.train()
        if_break = False
        ckp_start = time.time()
        for epoch in range(learning_params["epochs"]):
            epoch_start_time = time.time()
            for batch_idx, x in enumerate(tr_loader):
                optimizer.zero_grad()
                batch_y, batch_y_hat = model_out_clf(model, x, in_feat_minmax, device)
                loss = get_loss(batch_y.squeeze(), batch_y_hat, loss_type="nll")
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
                    batch_time_tr = (time.time() - ckp_start) / nbatches

                    t1 = time.time()
                    loss_val, rate_val = evaluate_model_clf(model, val_loader, device, in_feat_minmax, obj)
                    eval_time = time.time() - t1

                    tst.update(model, epoch, batch_idx, loss_val)
                    cur_lr = optimizer.param_groups[0]["lr"]
                    writer.add_scalar("train/_loss", loss.detach().item(), epoch * nbatches + batch_idx)
                    writer.add_scalar("val/_loss", loss_val, epoch * nbatches + batch_idx)
                    writer.add_scalar("val/_rate", rate_val, epoch * nbatches + batch_idx)
                    writer.add_scalar("learning_rate", cur_lr, epoch * nbatches + batch_idx)
                    writer.add_scalar("batch_time", batch_time_tr, epoch * nbatches + batch_idx)

                    print("Epoch {:03d} | Batch {:06d} | LR: {:.8f} | TR Loss {:.6f} | VAL Loss {:.6f} | "
                          "VAL Rate {:.6f} | s/ba {:.3f} | s/eval {:.3f} ".format(
                        epoch, batch_idx, cur_lr, loss.detach().item(), loss_val, rate_val, batch_time_tr, eval_time))
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
    loss_val, rate_val = evaluate_model_clf(model, val_loader, device, in_feat_minmax, obj)
    loss_te, rate_te, y_te, y_hat_te, y_hat_flat_te = evaluate_model_clf(
        model, te_loader, device, in_feat_minmax, obj, if_y=True)

    results = {
        "hp_params": hp_params,
        "Epoch": tst.best_epoch,
        "Batch": tst.best_batch,
        "Total_time": total_time,
        "timestamp": ts,
        "loss_val": loss_val,
        "loss_te": loss_te,
        "rate_val": rate_val,
        "rate_te": rate_te,
        "y_te": y_te,
        "y_te_hat": y_hat_te,
        "y_te_hat_flat": y_hat_flat_te
    }
    th.save(results, results_pth_sign)
    show_results(results, obj)
    return model, results, hp_params, hp_prefix_sign

# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: Add a proxy model for easy use of optimization
#
# Created at 08/01/2023
import numpy as np

from model.architecture.graph_transformer_net import GraphTransformerNet
import torch as th
from torch.utils.data import DataLoader, TensorDataset
from utils.model.utils import get_tensor, denorm_obj
from utils.optimization.moo_utils import is_pareto_efficient


class ModelProxy():

    def __init__(self, model_name, ckp_path, obj_minmax, device, op_groups, n_op_types):
        assert model_name == "GTN"
        results_pth_sign = f"{ckp_path}/results.pth"
        weights_pth_sign = f"{ckp_path}/best_weight.pth"
        results = th.load(results_pth_sign, map_location=device)
        hp_params = results["hp_params"]
        hp_params = {**hp_params, **{
            "op_groups": op_groups,
            "n_op_types": n_op_types
        }}
        model = GraphTransformerNet(hp_params).to(device=device)
        model.load_state_dict(th.load(weights_pth_sign, map_location=device)["model"])
        print(f"model loaded.")
        self.results = results
        self.hp_params = hp_params
        self.model = model
        self.device = device
        self.obj_minmax = {"min": get_tensor(obj_minmax["min"].values, device=device),
                           "max": get_tensor(obj_minmax["max"].values, device=device)}

    def get_stage_emb(self, g, fmt="numpy"):
        self.model.eval()
        g = g.to(self.device)
        pos_enc = g.ndata["lap_pe"].to(self.device)
        with th.no_grad():
            stage_emb = self.model.forward(g, pos_enc, inst_feat=None)
        if fmt == "torch":
            return stage_emb
        elif fmt == "numpy":
            return stage_emb.cpu().numpy()
        else:
            raise ValueError(fmt)

    def mesh_knobs(self, ch1, ch2, ch3, ch4):
        """
        generate the input to feed into the model
        :param ch1: query plan embedding.
        :param ch2: normed input feats
        :param ch3: normed system states (zero list) - assume in the isolated env
        :param ch4: a set of normalized confs
        :return:
        """
        assert ch1.shape[0] == 1 and ch2.shape[0] == 1 and ch3.shape[0] == 1
        n_rows = ch4.shape[0]
        stage_emb = np.repeat(ch1, n_rows, axis=0)
        inst_feat = np.hstack([np.repeat(ch2, n_rows, axis=0), np.repeat(ch3, n_rows, axis=0), ch4])
        return stage_emb, inst_feat

    def get_lat(self, ch1, ch2, ch3, ch4, out_fmt="numpy", dropout=False, n_samples=100):
        stage_emb, inst_feat = self.mesh_knobs(ch1, ch2, ch3, ch4)
        if dropout:
            stage_emb = np.repeat(stage_emb, n_samples, axis=0)
            inst_feat = np.repeat(inst_feat, n_samples, axis=0)
            self.model.train()
        else:
            self.model.eval()
        if not isinstance(stage_emb, th.Tensor):
            stage_emb = get_tensor(stage_emb)
        if not isinstance(inst_feat, th.Tensor):
            inst_feat = get_tensor(inst_feat)

        loader = DataLoader(dataset=TensorDataset(stage_emb, inst_feat),
                            batch_size=1024, shuffle=False, num_workers=0)

        with th.no_grad():
            y_hat_list = []
            for (stage_emb_batch, inst_feat_batch) in loader:
                stage_emb_batch = stage_emb_batch.to(self.device)
                inst_feat_batch = inst_feat_batch.to(self.device)
                y_hat_batch = self.model.mlp_forward(stage_emb_batch, inst_feat_batch)
                y_hat_batch = denorm_obj(y_hat_batch, self.obj_minmax)
                y_hat_list.append(y_hat_batch)
        y_hat = th.vstack(y_hat_list)

        if out_fmt == "torch":
            if dropout:
                y_hat_mu = y_hat.reshape(-1, n_samples).mean(1, keepdims=True)
                y_hat_std = y_hat.reshape(-1, n_samples).std(1, keepdims=True)
                return y_hat_mu, y_hat_std
            else:
                return y_hat
        elif out_fmt == "numpy":
            if dropout:
                y_hat_mu = y_hat.reshape(-1, n_samples).mean(1, keepdims=True)
                y_hat_std = y_hat.reshape(-1, n_samples).std(1, keepdims=True)
                return y_hat_mu.cpu().numpy(), y_hat_std.cpu().numpy()
            return y_hat.cpu().numpy()
        else:
            raise ValueError(out_fmt)


def get_weight_pairs(n, seed, ndim=2):
    assert ndim > 1
    if ndim == 2:
        n1 = n // 2 # get n1 in uniformed dist
        n2 = n - n1 # get n2 randomly generated

        wp1_1 = np.hstack([np.arange(0, 1, 1 / (n1 - 1)), np.array([1])]).reshape(-1, 1)
        wp1 = np.hstack([wp1_1, 1 - wp1_1])

        np.random.seed(seed)
        wp2 = np.random.rand(n2, 2)
        wp2 = np.exp(wp2) / np.exp(wp2).sum(1, keepdims=True)
        wp = np.vstack([wp1, wp2])

    else:
        wp1 = np.eye(ndim)
        np.random.seed(seed)
        wp2 = np.random.rand(n - ndim, ndim)
        wp2 = np.exp(wp2) / np.exp(wp2).sum(1, keepdims=True)
        wp = np.vstack([wp1, wp2])
    return wp


def bf_return(objs):
    mask = is_pareto_efficient(objs)
    return np.arange(len(objs))[mask]


def ws_return(objs, n_ws_pairs, seed):
    """
    return indices of objs that is Pareto-optimal
    :param objs: a numpy list of (lat, cost)
    :param n_ws_pairs: number of weighted_sum pairs
    :param seed: random seed
    :return inds: indices of po solutions
    """
    ndim = objs.shape[1]
    weight_pairs = get_weight_pairs(n_ws_pairs, seed, ndim=ndim)
    assert len(weight_pairs) >= 2
    inds = set()

    objs_norm = (objs - objs.min(0)) / (objs.max(0) - objs.min(0))
    for w_pair in weight_pairs:
        inds.add(np.argmin((objs_norm * w_pair).sum(1)))
    inds = list(inds)
    assert len(is_pareto_efficient(objs_norm[inds])) == len(inds)
    return inds

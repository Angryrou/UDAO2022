# Copyright (c) 2021 Ecole Polytechnique
#
# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description:
#

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle, logging, math, os
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist as ed
import gc
from model.architecture.gpr_utils import GPRPT

NETWORK_SCALE = 1
CPU_SCALE = 200

DEFAULT_RIDGE=1.0
FORMAT_ERROR = 'ERROR: format of zmesg is not correct'
NOT_FOUND_ERROR = 'no valid configuration found'
NN_OBJ_LIST = [
    "latency",
    "cpu",
    "network",
    "ops",
    "simulated_cost",  # cores * latency * cost_rate1
    "simulated_cost2", # latency * (cores * cost_rate1 + net_work_IO / avg_pkt * cost_rate2),
    "simulated_cost3", # latency/1000/60/60 (hour) * #cores * cost_rate1 + ops/1000/1000 * cost_rate3
]
R_OBJ_LIST = [
    'cores'
]
GP_OBJ_LIST = [
    'latency'
]

NUMERICAL_KNOB_IDS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
SEED = 0

BATCH_OFF_TEST_JOBS = "1-7,2-2,3-2,4-7,5-1,6-2,7-2,8-5,9-3,10-0,11-2,12-3,13-4,14-0,15-4,16-3,17-5,18-1,19-7,20-4,21-1,22-7,23-4,24-3,25-7,26-8,27-7,28-0,29-1,30-0".split(',')
BATCH_ON_TEST_JOBS = "1-1,2-3,3-8,4-2,5-5,6-3,7-8,8-0,9-7,10-1,11-0,12-5,13-7,14-2,15-1,16-2,17-0,18-7,19-6,20-7,21-3,22-8,23-5,24-8,25-4,26-2,27-3,28-2,29-2,30-7".split(',')

K2K3K4_ =  np.array([
    [2,2,4],
    [3,2,4],
    [2,4,8],
    [4,2,4],
    [3,4,8],
    [4,3,6],
    [6,2,4],
    [4,4,8],
    [8,2,4],
    [6,4,8],
    [8,3,6],
    [12,2,4],
    [8,4,8],
    [16,2,4],
    [18,4,8],
    [24,3,6],
    [36,2,4]
])

class GPR_Batch_models:
    def __init__(self, cost_rate1=0.06, cost_rate2=0.0002, cost_rate3=0.2, avg_pkt_size=0.1, stress=10):
        torch.set_num_threads(1)
        self.device, self.dtype = torch.device('cpu'), torch.float32
        self.stress = stress
        self.knob_list = ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "s1", "s2", "s3", "s4"]
        self.model = GPRPT(gp_obj_list=GP_OBJ_LIST)
        gpr_data_ = self._load_obj("gpr_data_batch")
        data_ = gpr_data_['data']
        proxy_jobs_ = gpr_data_['proxy_jobs']
        wl_list_ = BATCH_OFF_TEST_JOBS + BATCH_ON_TEST_JOBS
        model_map_, scaler_map_ = self._get_gp_models(data_, proxy_jobs_, wl_list_)

        self.data = data_
        self.model_map = model_map_
        self.scaler_map = scaler_map_
        self.wl_list = wl_list_

        logging.basicConfig(level=logging.INFO,
                            filename=f'gpr_batch_models.log',
                            format=f'%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        logging.info("Models are initiated!")

    @staticmethod
    def _loss_soo_minibatch(obj_pred):
        # for single job objective, the loss can be its own value.
        loss = obj_pred ** 2
        return torch.min(loss), torch.argmin(loss)

    # normed loss, stress outside ranges
    def _loss_moo(self, target_obj, pred_dict, cst_dict):
        loss = torch.tensor(0, device=self.device, dtype=self.dtype)
        for cst_obj, [lower, upper] in cst_dict.items():
            cst_obj_pred = pred_dict[cst_obj]
            if upper != lower:
                norm_cst_obj_pred = (cst_obj_pred - lower) / (upper - lower)  # scaled
                add_loss = torch.tensor(0, device=self.device, dtype=self.dtype)
                if cst_obj == target_obj:
                    if norm_cst_obj_pred < 0 or norm_cst_obj_pred > 1:
                        add_loss += (norm_cst_obj_pred - 0.5) ** 2 + self.stress
                    else:
                        add_loss += norm_cst_obj_pred
                else:
                    if norm_cst_obj_pred < 0 or norm_cst_obj_pred > 1:
                        add_loss += (norm_cst_obj_pred - 0.5) ** 2 + self.stress
            else:
                add_loss = (cst_obj_pred - upper) ** 2 + self.stress
            loss += add_loss
        return loss


    def predict(self, zmesg):
        """
        predict the objective value given a wl and a conf
        :param zmesg: see batch_knob_format.md
        :return: a scalar objective prediction
        """
        # unseralize zmesg
        wl_id, obj, conf_val_list = self._input_unserialize_predict(zmesg)

        if wl_id is None:
            return -1

        conf = self._get_tensor(conf_val_list)
        obj_pred = self._get_tensor_prediction(wl_id, conf, obj)

        logging.info(f"{wl_id}, {conf_val_list} -> {obj}: {obj_pred}")
        return f"{obj_pred:.5f}"

    # minibatch at a time
    def opt_scenario1(self, zmesg, bs=16, lr=0.01, max_iter=100, weight_decay=0.1, patient=20, verbose=False):
        """
        minimize a target objective value given a workload
        :param zmesg: see batch_knob_format.md
        :return: a scalar objective prediction
        """
        torch.manual_seed(SEED)
        wl_id, obj = self._input_unserialize_opt_1(zmesg)
        if wl_id is None:
            return -1, None
        conf_max, conf_min = self._get_conf_range(wl_id)

        best_loss = np.inf
        best_obj = np.inf
        best_conf = None
        iter_num = 0

        for k7 in [0, 1]:
            if verbose:
                print('-' * 10)
                print(f"k7: {k7}")

            # numerical_knob_list = self.__get_tensor_numerical_knobs([0.5] * 8)
            numerical_knob_list = torch.rand(bs, len(NUMERICAL_KNOB_IDS),
                                              device=self.device, dtype=self.dtype, requires_grad=True)
            optimizer = optim.Adam([numerical_knob_list], lr=lr, weight_decay=weight_decay)

            local_best_iter = 0
            local_best_loss = np.inf
            local_best_obj = np.inf
            local_best_conf = None

            iter = 0
            for iter in range(max_iter):
                conf = self._get_tensor_conf_cat_minibatch(numerical_knob_list, k7)
                obj_pred = self._get_tensor_prediction_minibatch(wl_id, conf, obj) # Nx1
                loss, loss_id = self._loss_soo_minibatch(obj_pred)

                if iter > 0 and loss.item() < local_best_loss:
                    local_best_loss = loss.item()
                    local_best_obj = obj_pred[loss_id].item()
                    local_best_conf = conf.data.numpy()[loss_id, :].copy()
                    local_best_iter = iter

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                constrained_numerical_knob_list = self._get_tensor_numerical_constrained_knobs_minibatch(
                    numerical_knob_list.data, conf_max, conf_min)
                numerical_knob_list.data = constrained_numerical_knob_list

                if iter > local_best_iter + patient:
                    # early stop
                    break

                if verbose:
                    if iter % 10 == 11:
                        print(f'iteration {iter}, {obj}: {obj_pred:.2f}')
                        print(conf)

            logging.info(f'Local best {obj}: {local_best_obj:.5f} at {local_best_iter} with confs:\n'
                         f'{local_best_conf}')
            if verbose:
                print(f'Finished at iteration {iter}, best local {obj} found as {local_best_obj:.5f}'
                      f' \nat iteration {local_best_iter}, \nwith confs: {self._get_raw_conf(local_best_conf, conf_max, conf_min)}')

            iter_num += iter + 1
            if local_best_loss < best_loss:
                best_obj = local_best_obj
                best_loss = local_best_loss
                best_conf = local_best_conf

        best_raw_conf = self._get_raw_conf(best_conf, conf_max, conf_min)
        logging.info(f"get best {obj}: {best_obj} at {best_raw_conf} with {iter_num} iterations, loss = {best_loss}")
        if verbose:
            print()
            print("*" * 10)
            print(f"get best {obj}: {best_obj} at {best_raw_conf} with {iter_num} iterations, loss = {best_loss}")

        str1 = self._get_seralized_conf(best_raw_conf)
        str2 = f"{obj}:{best_obj:.5f}"
        return '&'.join([str1, str2])


    def opt_scenario2(self, zmesg, lr=0.01, max_iter=100, weight_decay=0.1, patient=20, verbose=False, multistart=8,
                            benchmark=False):
        """
        minimize a target objective value given a workload with K constrained objectives.
        Return the best configuration and its obj values. E.g.,
        ret="k1:8;k2:3;k3:2;k4:4;k5:384;k6:217;k7:1;k8:70;s1:50000;s2:256;s3:10;s4:8;latency:1010;cores:18"
        :param zmesg: see batch_knob_format.md
        :return: the conf and its obj values
        """
        # assume obj is got from NN model
        torch.manual_seed(SEED)
        target_obj_val = []
        ret_list = []

        for si in range(multistart):
            wl_id, obj, cst_dict = self._input_unserialize_opt_2(zmesg)
            if wl_id is None:
                return None
            conf_max, conf_min = self._get_conf_range(wl_id)

            best_obj_dict = None
            best_loss = np.inf
            best_conf = None
            iter_num = 0

            # for idx1, k2k3k4 in enumerate(normalized_k2k3k4_list):
            for k7 in [0, 1]:
                if verbose:
                    print('-' * 10)
                    print(f"k7: {k7}")

                # numerical_knob_list = self._get_tensor([0.1] * len(NUMERICAL_KNOB_IDS), requires_grad=True)
                numerical_knob_list = torch.rand(len(NUMERICAL_KNOB_IDS), device=self.device, dtype=self.dtype, requires_grad=True)
                optimizer = optim.Adam([numerical_knob_list], lr=lr, weight_decay=weight_decay)

                local_best_iter = 0
                local_best_obj_dict = None
                local_best_loss = np.inf
                local_best_conf = None

                iter = 0
                for iter in range(max_iter):
                    conf = self._get_tensor_conf_cat(numerical_knob_list, k7)
                    obj_pred_dict = {cst_obj: self._get_tensor_prediction(wl_id, conf, cst_obj) for cst_obj in cst_dict}
                    loss = self._loss_moo(obj, obj_pred_dict, cst_dict)

                    if iter > 0 and loss.item() < local_best_loss:
                        local_best_loss = loss.item()
                        local_best_obj_dict = {k: v.item() for k, v in obj_pred_dict.items()}
                        local_best_conf = conf.data.numpy().copy()
                        local_best_iter = iter

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    constrained_numerical_knob_list = self._get_tensor_numerical_constrained_knobs(
                        numerical_knob_list, conf_max, conf_min)
                    numerical_knob_list.data = constrained_numerical_knob_list

                    if iter > local_best_iter + patient:
                        # early stop
                        break

                    if verbose:
                        if iter % 10 == 0:
                            print(f'iteration {iter}, {local_best_obj_dict}, loss: {loss.item():.2f}')
                            print(conf)

                logging.info(f'Local best {local_best_obj_dict} at {local_best_iter} with confs:\n'
                             f'{local_best_conf}')
                if verbose:
                    print(f'Finished at iteration {iter}, best local {local_best_obj_dict}'
                          f' \nat iteration {local_best_iter}, \nwith confs: {self._get_raw_conf(local_best_conf, conf_max, conf_min)}')

                iter_num += iter + 1
                if self._bound_check(pred_dict=local_best_obj_dict, cst_dict=cst_dict):
                    if local_best_loss < best_loss:
                        best_obj_dict = local_best_obj_dict
                        best_loss = local_best_loss
                        best_conf = local_best_conf

            if self._bound_check(pred_dict=best_obj_dict, cst_dict=cst_dict):
                best_raw_conf = self._get_raw_conf(best_conf, conf_max, conf_min)
                obj_pred_dict = {}
                for cst_obj in cst_dict:
                    if cst_obj in R_OBJ_LIST:
                        obj_pred_dict[cst_obj] = self._get_r_obj(cst_obj, best_raw_conf)
                    elif cst_obj in GP_OBJ_LIST:
                        obj_pred_dict[cst_obj] = best_obj_dict[cst_obj]
                    else:
                        raise Exception(f"{cst_obj} cannot be found")

                ret = self._get_opt2_ret(best_raw_conf, obj_pred_dict)
                if verbose:
                    print()
                    print("*" * 10)
                    print(f"get {obj_pred_dict}")
                target_obj_val.append(obj_pred_dict[obj])
            else:
                if verbose:
                    print(NOT_FOUND_ERROR + "on GP objectives")
                logging.warning(NOT_FOUND_ERROR + "on GP objectives")
                ret = "not_found"
                target_obj_val.append(np.inf)
                # return "GP_miss" if benchmark else "not_found"

            ret_list.append(ret)
        idx = np.argmin(target_obj_val)
        return ret_list[idx]

    def opt_scenario3(self, zmesg, lr=0.01, max_iter=100, weight_decay=0.1, patient=20, verbose=False, multistart=8,
                      benchmark=False, processes=1):
        torch.manual_seed(SEED)
        zmesg_list = zmesg.split('|')
        arg_list = [(z, lr, max_iter, weight_decay, patient, verbose, multistart, benchmark) for z in zmesg_list]

        with Pool(processes=processes) as pool:
            ret_list = pool.starmap(self.opt_scenario2, arg_list)
        ret = '|'.join(ret_list)

        return ret


    ################################
    ##### get_tensor functions #####
    ################################

    def _get_tensor_r_obj(self, obj, conf, conf_max, conf_min):
        if obj == "cores":
            k2_max, k3_max = conf_max[1:3]
            k2_min, k3_min = conf_min[1:3]
            k2_norm, k3_norm = conf[1:3]
            k2 = k2_norm * (k2_max - k2_min) + k2_min
            k3 = k3_norm * (k3_max - k3_min) + k3_min
            n_exec = torch.min(torch.floor(torch.tensor(58, device=self.device, dtype=self.dtype) / k3), k2)
            return n_exec * torch.round(k3)
        else:
            raise Exception(f"{obj} cannot be found")

    def _get_tensor_r_obj_minibatch(self, obj, conf, conf_max, conf_min):
        if obj == "cores":
            k2_max, k3_max = conf_max[1:3]
            k2_min, k3_min = conf_min[1:3]
            # k2_norm, k3_norm = conf[:, 1:3]
            k2_norm = conf[:, 1]
            k3_norm = conf[:, 2]
            k2 = k2_norm * (k2_max - k2_min) + k2_min
            k3 = k3_norm * (k3_max - k3_min) + k3_min
            n_exec = torch.min(torch.floor(torch.tensor(58, device=self.device, dtype=self.dtype) / k3), k2)
            cores = (n_exec * k3).view(-1, 1)
            # print(f'cores: {cores}, {cores.shape}')
            return cores
        else:
            raise Exception(f"{obj} cannot be found")

    def _get_tensor_prediction(self, wl_id, conf, obj):
        conf_max, conf_min = self._get_conf_range(wl_id)
        if obj in GP_OBJ_LIST:
            X_train, y_dict, K_inv = self.model_map[wl_id]
            y_train = y_dict[obj]
            obj_pred = self.model.objective(conf, X_train, y_train, K_inv)[0]
        elif obj in R_OBJ_LIST:
            obj_pred = self._get_tensor_r_obj(obj, conf, conf_max=conf_max, conf_min=conf_min)
        else:
            raise Exception(f'{obj} cannot be found')
        return obj_pred

    def _get_tensor_prediction_minibatch(self, wl_id, conf, obj):
        conf_max, conf_min = self._get_conf_range(wl_id)
        if obj in GP_OBJ_LIST:
            X_train, y_dict, K_inv = self.model_map[wl_id]
            y_train = y_dict[obj]
            obj_pred = self.model.objective(conf, X_train, y_train, K_inv)
        elif obj in R_OBJ_LIST:
            obj_pred = self._get_tensor_r_obj_minibatch(obj, conf, conf_max=conf_max, conf_min=conf_min)
        else:
            raise Exception(f'{obj} cannot be found')
        return obj_pred

    def _get_tensor_conf_cat(self, numerical_knob_list, k7):
        # t_k2k3k4 = torch.tensor(k2k3k4, device=self.device, dtype=self.dtype)
        t_k7 = torch.tensor(k7, device=self.device, dtype=self.dtype)
        # [k1, k2, k3, k4, k5, k6, k8, s1, s2, s3, s4]
        conf = torch.cat([numerical_knob_list[:6],
                          t_k7.view(1), numerical_knob_list[6:]])
        return conf

    def _get_tensor_conf_cat_minibatch(self, numerical_knob_list, k7):
        # new_size = (numerical_knob_list.shape[0], -1)
        n_batch = numerical_knob_list.shape[0]
        t_k7 = torch.ones((n_batch, 1), device=self.device, dtype=self.dtype) if k7 == 1 \
            else torch.zeros((n_batch, 1), device=self.device, dtype=self.dtype)
        # [k1, k2, k3, k4, k5, k6, k8, s1, s2, s3, s4]
        conf = torch.cat([numerical_knob_list[:, :6], t_k7, numerical_knob_list[:, 6:]], dim=1)
        return conf

    def _get_tensor_bounds_list(self, lower, upper):
        t_lower = torch.tensor(lower, device=self.device, dtype=self.dtype)
        t_upper = torch.tensor(upper, device=self.device, dtype=self.dtype)
        return [t_lower, t_upper]

    def _get_tensor_numerical_constrained_knobs(self, numerical_knob_list, conf_max, conf_min):
        # Tensor[k1, k2, k3, k4, k5, k6, k8, s1, s2, s3, s4]
        bounded_np = np.array([self._get_bounded(k.item()) for k in numerical_knob_list])
        raw_np = self._get_raw_conf(bounded_np, conf_max, conf_min, normalized_ids=NUMERICAL_KNOB_IDS)
        normalized_np = self._get_normalized_conf(raw_np, conf_max, conf_min, normalized_ids=NUMERICAL_KNOB_IDS)
        return torch.tensor(normalized_np, device=self.device, dtype=self.dtype)
        # return torch.tensor(bounded_np, device=self.device, dtype=self.dtype)

    def _get_tensor_numerical_constrained_knobs_minibatch(self, numerical_knob_list, conf_max, conf_min):
        # Tensor n_batch x len(NUMERICAL_KNOB_IDS)
        # [k1, k5, k6, k8, s1, s2, s3, s4]
        # bounded_np = np.array([self.__get_bounded(k.item()) for k in numerical_knob_list])
        numerical_knob_list[numerical_knob_list > 1] = 1
        numerical_knob_list[numerical_knob_list < 0] = 0
        raw_np = self._get_raw_conf(numerical_knob_list.numpy(), conf_max, conf_min, normalized_ids=NUMERICAL_KNOB_IDS)
        normalized_np = self._get_normalized_conf(raw_np, conf_max, conf_min, normalized_ids=NUMERICAL_KNOB_IDS)
        return torch.tensor(normalized_np, device=self.device, dtype=self.dtype)

    def _get_tensor(self, x, dtype=None, device=None, requires_grad=False):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


    ###########################
    ##### "get" funcitons #####
    ###########################

    def _get_gp_models(self, data, proxy_jobs, wl_list):
        obj_lbl = data['metrics_details']['objective_label']
        obj_idx = data['metrics_details']['objective_idx']
        ##### obj_lbl and obj_idx to be matched with pred_objective as below ####

        model_map = {}
        scaler_map = {}
        for wl in wl_list:
            # Target workload data (observed)
            if wl in proxy_jobs:
                X_target = data['observed'][wl]['X_matrix'].copy()
                y_target = data['observed'][wl]['y_matrix'][:, obj_idx].copy()
                proxy_id = proxy_jobs[wl]
                X_workload = data['training'][proxy_id]['X_matrix'].copy()
                y_workload = data['training'][proxy_id]['y_matrix'][
                             :, obj_idx].copy()
                if np.ndim(y_workload) == 1:
                    y_workload = np.expand_dims(y_workload, axis=1)
                if np.ndim(y_target) == 1:
                    y_target = np.expand_dims(y_target, axis=1)
                dups_filter = np.ones(X_workload.shape[0], dtype=bool)
                target_row_tups = [tuple(row) for row in X_target]
                for i, row in enumerate(X_workload):
                    if tuple(row) in target_row_tups:
                        dups_filter[i] = False
                X_workload = X_workload[dups_filter, :]
                y_workload = y_workload[dups_filter, :]
                # Combine target (observed) & workload (mapped) Xs for preprocessing
                X_matrix = np.vstack([X_target, X_workload])
                y_matrix = np.vstack([y_target, y_workload])
                y_dict = {o_lbl: y_matrix[:, o_idx] for o_idx, o_lbl in enumerate(obj_lbl) if o_lbl in GP_OBJ_LIST}
            else:
                X_matrix = data['training'][wl]['X_matrix'].copy()
                y_matrix = data['training'][wl]['y_matrix'].copy()
                y_dict = {o_lbl: y_matrix[:, o_idx] for o_lbl, o_idx in zip(obj_lbl, obj_idx) if o_lbl in GP_OBJ_LIST}

            X_matrix[:, 7] *= 100
            # Scale to (0, 1)
            X_scaler = MinMaxScaler()
            X_scaled = X_scaler.fit_transform(X_matrix)

            y_dict['latency'] *= 1000
            model_map[wl] = self.model.fit(X_train=X_scaled, y_dict=y_dict, ridge=DEFAULT_RIDGE)
            scaler_map[wl] = X_scaler
        return model_map, scaler_map

    def _get_conf_range(self, wl_id):
        conf_max = self.scaler_map[wl_id].data_max_
        conf_min = self.scaler_map[wl_id].data_min_
        return conf_max, conf_min

    def _get_seralized_conf(self, conf):
        s_conf = ';'.join([f'{self.knob_list[i]}:{int(conf[i])}' for i in range(len(self.knob_list))])
        return s_conf

    @staticmethod
    def _get_core_num(k2, k3):
        return min(math.floor(58 / k3), k2) * k3

    def _get_r_obj(self, obj, conf):
        if obj == "cores":
            k2, k3 = conf[1:3]
            return self._get_core_num(k2, k3)

    def _get_opt2_ret(self, conf, pred_dict):
        str1 = self._get_seralized_conf(conf)
        str2 = ';'.join([f'{k}:{v:.5f}' for k, v in pred_dict.items()])
        return f'{str1}&{str2}'

    @staticmethod
    def _get_bounded(k, lower=0.0, upper=1.0):
        if k > upper:
            return upper
        if k < lower:
            return lower
        return k

    @staticmethod
    def _get_normalized_conf(raw_conf, conf_max, conf_min, normalized_ids=None):
        """
        :param real_conf: numpy.array[int]
        :return: normalized to 0-1
        """
        conf_max = conf_max if normalized_ids is None else conf_max[normalized_ids]
        conf_min = conf_min if normalized_ids is None else conf_min[normalized_ids]
        normalized_conf = (raw_conf - conf_min) / (conf_max - conf_min)
        return normalized_conf

    @staticmethod
    def _get_raw_conf(normalized_conf, conf_max, conf_min, normalized_ids=None):
        """
        :param normalized_conf: numpy.array[float] [0,1]
        :return: numpy.array[int]
        """
        conf_max = conf_max if normalized_ids is None else conf_max[normalized_ids]
        conf_min = conf_min if normalized_ids is None else conf_min[normalized_ids]
        conf = normalized_conf * (conf_max - conf_min) + conf_min
        get_raw_conf = conf.round()
        return get_raw_conf

    @staticmethod
    def _load_obj(name):
        with open('batch_confs/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    ##################################
    ##### format check functions #####
    ##################################

    def _input_unserialize_predict(self, zmesg):
        """
        E.g., "JobID:13-4;Objective:network;k1:8;k2:2;k3:2;k4:4;k5:12;k6:7;k7:0;k8:50;s1:1000;s2:32;s3:10;s4:8"
        :param zmesg:
        :return: wl_id, obj, conf_norm_val
        """
        try:
            kv_dict = {kv.split(":")[0]: kv.split(":")[1] for kv in zmesg.split(";")}
            wl_id = kv_dict['JobID']
            obj = kv_dict['Objective']
            conf_raw_val = np.array([float(kv_dict[k]) for k in self.knob_list])
            conf_max, conf_min = self._get_conf_range(wl_id)
            conf_norm_val = self._get_normalized_conf(conf_raw_val, conf_max=conf_max, conf_min=conf_min)
        except:
            logging.error(FORMAT_ERROR + f'{zmesg}')
            print(FORMAT_ERROR + f'{zmesg}')
            return None, None, None

        if not self._wl_check(wl_id):
            return None, None, None

        if not self._obj_check(obj):
            return None, None, None

        return wl_id, obj, conf_norm_val

    def _input_unserialize_opt_1(self, zmesg):
        """
        E.g. zmesg = "JobID:1-2;Objective:latency"
        :param zmesg:
        :return: wl_id, obj
        """
        try:
            kv_dict = {kv.split(":")[0]: kv.split(":")[1] for kv in zmesg.split(";")}
            wl_id = kv_dict['JobID']
            obj = kv_dict['Objective']
        except:
            logging.error(FORMAT_ERROR + f'{zmesg}')
            print(FORMAT_ERROR + f'{zmesg}')
            return None, None

        if not self._wl_check(wl_id):
            return None, None

        if not self._obj_check(obj):
            return None, None

        return wl_id, obj

    def _input_unserialize_opt_2(self, zmesg):
        """
        we assume the constraints are all in integer
        E.g., zmesg = "JobID:13-4;Objective:latency;Constraint:cores:10:20;Constraint:latency:1000:2000"
        :param zmesg:
        :return: wl_id, obj, constraint_dict{obj: (l, r)}
        """
        wl_id, obj, constraint_dict = None, None, None
        try:
            kv_list = zmesg.split(';')
            cst_dict = {}
            for kv in kv_list:
                sub_kv = kv.split(':')
                if sub_kv[0] == "JobID":
                    wl_id = sub_kv[1]
                elif sub_kv[0] == "Objective":
                    obj = sub_kv[1]
                elif sub_kv[0] == "Constraint":
                    sub_obj, lower, upper = sub_kv[1], int(sub_kv[2]), int(sub_kv[3])
                    cst_dict[sub_obj] = self._get_tensor_bounds_list(lower, upper)
                else:
                    logging.warning(f"{sub_kv[0]} is unrecognized.")
        except:
            logging.error(FORMAT_ERROR + f'{zmesg}')
            print(FORMAT_ERROR + f'{zmesg}')
            return None, None, None

        if not self._wl_check(wl_id):
            return None, None, None

        if not self._obj_check(obj):
            return None, None, None

        for cst_obj in cst_dict:
            if not self._obj_check(cst_obj):
                return None, None, None

        return wl_id, obj, cst_dict

    def _wl_check(self, wl_id):
        if wl_id not in self.wl_list:
            logging.error(f'ERROR: workload {wl_id} is not found')
            print(f'ERROR: workload {wl_id} is not found')
            return False
        return True

    @staticmethod
    def _obj_check(obj):
        if obj not in R_OBJ_LIST + GP_OBJ_LIST:
            logging.error(f'ERROR: objective {obj} is not found')
            print(f'ERROR: objective {obj} is not found')
            return False
        return True

    @staticmethod
    def _bound_check(pred_dict, cst_dict):
        """
        check if every predicted obj is within constraints
        :param pred_dict: {obj: scalar}
        :param cst_dict:
        :return: True / False
        """
        if pred_dict is None:
            return False
        for obj, obj_pred in pred_dict.items():
            lower, upper = cst_dict[obj]
            if lower.item() <= obj_pred <= upper.item():
                pass
            else:
                return False
        return True

if __name__ == "__main__":
    bm = GPR_Batch_models()
    zmesg = "JobID:13-4;Objective:latency;k1:8;k2:2;k3:2;k4:4;k5:12;k6:7;k7:0;k8:50;s1:1000;s2:32;s3:10;s4:8"
    print(f'-- test `predict` latency, input: {zmesg}')
    print(f'get: {bm.predict(zmesg)}\n')

    zmesg = "JobID:13-4;Objective:cores;k1:8;k2:2;k3:2;k4:4;k5:12;k6:7;k7:0;k8:50;s1:1000;s2:32;s3:10;s4:8"
    print(f'-- test `predict` cores, input: {zmesg}')
    print(f'get: {bm.predict(zmesg)}\n')

    for obj in ['latency', 'cores']:
        zmesg = f"JobID:13-4;Objective:{obj}"
        print(f'-- test `opt_scenario1` to find the global minimum for {obj}')
        res = bm.opt_scenario1(zmesg=zmesg, lr=0.1)
        print(f'out: {res}\n')

    print(f"-- test `opt_scenario2` to solve the CO problem")
    zmesg = "JobID:13-4;Objective:latency;Constraint:cores:10:20;Constraint:latency:20000:30000"
    print(f'input: {zmesg}')
    res = bm.opt_scenario2(zmesg, max_iter=100, lr=0.1)
    print(f'output: {res}\n')

    print(f'-- test `opt_scenario3` to solve the CO problem in parallel')
    zmesg = "JobID:13-4;Objective:latency;Constraint:cores:10:20;Constraint:latency:10000:20000|JobID:13-4;Objective:latency;Constraint:cores:10:20;Constraint:latency:20000:30000"
    print(f'input: {zmesg}')
    res = bm.opt_scenario3(zmesg, max_iter=100, lr=0.1, verbose=False)
    print(f'output: {res}\n')

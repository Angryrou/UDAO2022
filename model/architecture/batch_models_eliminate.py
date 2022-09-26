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
from model.architecture.gpr_utils import GPRPT
from sklearn.preprocessing import MinMaxScaler, StandardScaler

NETWORK_SCALE = 1
CPU_SCALE = 200

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
DEFAULT_RIDGE = 1.0
BATCH_OFF_TEST_JOBS = "1-7,2-2,3-2,4-7,5-1,6-2,7-2,8-5,9-3,10-0,11-2,12-3,13-4,14-0,15-4,16-3,17-5,18-1,19-7,20-4,21-1,22-7,23-4,24-3,25-7,26-8,27-7,28-0,29-1,30-0".split(',')
NUMERICAL_KNOB_IDS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
SEED = 0

class Batch_models:
    def __init__(self, cost_rate1=0.06, cost_rate2=0.0002, cost_rate3=0.2, avg_pkt_size=0.1, stress=10,
                 accurate=True, alpha=1.0, **kwargs):
        torch.set_num_threads(1)
        self.device, self.dtype = torch.device('cpu'), torch.float32
        self.cost_rate1 = cost_rate1
        self.cost_rate2 = cost_rate2
        self.cost_rate3 = cost_rate3
        self.avg_pkt_size = avg_pkt_size
        self.stress = stress
        self.knob_list = ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "s1", "s2", "s3", "s4"]
        wrapped_model = self.__load_obj("wrapped_model")

        self.wlId_to_alias_dict = wrapped_model["wlId_to_alias_dict"]
        self.wl_encode_list = wrapped_model["centroids"]

        # a list of knob max/min value list from k1 - s4
        self.conf_max = np.array(wrapped_model['conf_max'])
        self.conf_min = np.array(wrapped_model['conf_min'])

        if "conf_constraints" in kwargs:
            self.conf_cons_max = np.array(kwargs['conf_constraints']['conf_max'])
            self.conf_cons_min = np.array(kwargs['conf_constraints']['conf_min'])
        else:
            self.conf_cons_max = self.conf_max.copy()
            self.conf_cons_min = self.conf_min.copy()

        # transfer k8 to a percentage integer
        self.conf_max[7] *= 100
        self.conf_min[7] *= 100

        self.conf_cons_max[7] *= 100
        self.conf_cons_min[7] *= 100
        self.conf_cons_max_norm = (self.conf_cons_max - self.conf_min) / (self.conf_max - self.conf_min)
        self.conf_cons_min_norm = (self.conf_cons_min - self.conf_min) / (self.conf_max - self.conf_min)

        X_scaler_ = MinMaxScaler()
        X_scaler_.fit(np.concatenate([self.conf_max.reshape(1, -1), self.conf_min.reshape(1, -1)], axis=0))
        self.X_scaler = X_scaler_

        self.model_weights_dict = self.__load_weights(wrapped_model["extracted_weights_dict"])

        self.accurate = accurate
        if not accurate:
            self.alpha = alpha
            self.gpr_model = GPRPT(gp_obj_list=GP_OBJ_LIST)
            gpr_data_ = self.__load_obj("gpr_data_batch")
            data_ = gpr_data_['data']
            proxy_jobs_ = gpr_data_['proxy_jobs']
            wl_list_ = BATCH_OFF_TEST_JOBS
            model_meta_map_, scale_y_map_ = self._get_gp_models(data_, proxy_jobs_, wl_list_)
            self.gpr_metr_map = model_meta_map_
            self.scale_y_map = scale_y_map_

        logging.basicConfig(level=logging.INFO,
                            filename=f'batch_models.log',
                            format=f'%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        logging.info("Models are initiated!")

    def _get_gp_models(self, data, proxy_jobs, wl_list):
        obj_lbl = data['metrics_details']['objective_label']
        obj_idx = data['metrics_details']['objective_idx']
        ##### obj_lbl and obj_idx to be matched with pred_objective as below ####

        model_map = {}
        scale_y_map = {}
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
            X_scaled = self.X_scaler.transform(X_matrix)
            y_dict['latency'] *= 1000

            y_scale_obj_dict = {obj: self.get_y_scale_in_SS(y_dict[obj].reshape(-1, 1))for obj in GP_OBJ_LIST}

            model_map[wl] = self.gpr_model.fit(X_train=X_scaled, y_dict=y_dict, ridge=DEFAULT_RIDGE)
            scale_y_map[wl] = y_scale_obj_dict
        return model_map, scale_y_map

    def get_y_scale_in_SS(self, y):
        y_scaler_ = StandardScaler()
        y_scaler_.fit(y)
        return y_scaler_.scale_[0]

    def opt_scenario3(self, zmesg, lr=0.01, max_iter=100, weight_decay=0.1, patient=20, verbose=False, benchmark=False,
                      processes=1, multistart=1):
        torch.manual_seed(SEED)
        zmesg_list = zmesg.split('|')
        arg_list = [(z, lr, max_iter, weight_decay, patient, verbose, benchmark, multistart) for z in zmesg_list]

        with Pool(processes=processes) as pool:
            ret_list = pool.starmap(self.opt_scenario2, arg_list)
        ret = '|'.join(ret_list)

        return ret

    def predict(self, zmesg):
        """
        predict the objective value given a wl and a conf
        :param zmesg: see batch_knob_format.md
        :return: a scalar objective prediction
        """
        # unseralize zmesg
        wl_id, obj, conf_val_list = self.__input_unserialize_predict(zmesg)

        if wl_id is None:
            return -1

        # get values in terms of tensor or dict of tensors
        wl_encode = self.__get_tensor_wl_encode(wl_id)
        conf = self.__get_tensor_conf_for_predict(conf_val_list)

        obj_predict = self.__get_tensor_prediction(wl_encode, conf, obj).item()
        logging.info(f"{wl_id}, {conf_val_list} -> {obj}: {obj_predict}")

        return f"{obj_predict:.5f}"

    # minibatch at a time
    def opt_scenario1(self, zmesg, bs=16, lr=0.01, max_iter=100, weight_decay=0.1, patient=20, verbose=False):
        """
        minimize a target objective value given a workload
        :param zmesg: see batch_knob_format.md
        :return: a scalar objective prediction
        """
        torch.manual_seed(SEED)
        wl_id, obj = self.__input_unserialize_opt_1(zmesg)
        if wl_id is None:
            return -1, None

        wl_encode = self.__get_tensor_wl_encode(wl_id)
        # normalized_k2k3k4_list = self.__get_normalized_k2k3k4_list(K2K3K4_LIST)

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
                conf = self.__get_tensor_conf_cat_minibatch(numerical_knob_list, k7)
                obj_pred = self.__get_tensor_prediction_minibatch(wl_encode, conf, obj) # Nx1
                loss, loss_id = self.__loss_soo_minibatch(wl_id, conf, obj, obj_pred)

                if iter > 0 and loss.item() < local_best_loss:
                    local_best_loss = loss.item()
                    local_best_obj = obj_pred[loss_id].item()
                    local_best_conf = conf.data.numpy()[loss_id, :].copy()
                    local_best_iter = iter

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                constrained_numerical_knob_list = self.__get_tensor_numerical_constrained_knobs_minibatch(
                    numerical_knob_list.data)
                numerical_knob_list.data = constrained_numerical_knob_list
                # print(numerical_knob_list.data)

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
                      f' \nat iteration {local_best_iter}, \nwith confs: {self.__get_raw_conf(local_best_conf)}')

            iter_num += iter + 1
            if local_best_loss < best_loss:
                best_obj = local_best_obj
                best_loss = local_best_loss
                best_conf = local_best_conf

        best_raw_conf = self.__get_raw_conf(best_conf)
        logging.info(f"get best {obj}: {best_obj} at {best_raw_conf} with {iter_num} iterations, loss = {best_loss}")
        if verbose:
            print()
            print("*" * 10)
            print(f"get best {obj}: {best_obj} at {best_raw_conf} with {iter_num} iterations, loss = {best_loss}")

        str1 = self.__get_seralized_conf(best_raw_conf)
        str2 = f"{obj}:{best_obj:.5f}"
        return '&'.join([str1, str2])

    # a conf at a time
    def opt_scenario1_1_conf_at_a_time(self, zmesg, lr=0.01, max_iter=100, weight_decay=0.1, patient=30, verbose=False, init_numerical_knob=None):
        """
        minimize a target objective value given a workload
        :param zmesg: see batch_knob_format.md
        :return: a scalar objective prediction
        """
        wl_id, obj = self.__input_unserialize_opt_1(zmesg)
        if wl_id is None:
            return -1, None

        wl_encode = self.__get_tensor_wl_encode(wl_id)
        # normalized_k2k3k4_list = self.__get_normalized_k2k3k4_list(K2K3K4_LIST)

        best_loss = np.inf
        best_obj = np.inf
        best_conf = None
        iter_num = 0

        # for idx1, k2k3k4 in enumerate(normalized_k2k3k4_list):
        for k7 in [0, 1]:
            if verbose:
                print('-' * 10)
                print(f"k7: {k7}")

            if init_numerical_knob is None:
                numerical_knob_list = self.__get_tensor_numerical_knobs([0] * len(NUMERICAL_KNOB_IDS))
            else:
                numerical_knob_list = self.__get_tensor_numerical_knobs(init_numerical_knob)
            optimizer = optim.Adam([numerical_knob_list], lr=lr, weight_decay=weight_decay)

            local_best_iter = 0
            local_best_loss = np.inf
            local_best_obj = np.inf
            local_best_conf = None

            iter = 0
            for iter in range(max_iter):
                conf = self.__get_tensor_conf_cat(numerical_knob_list, k7)
                obj_pred = self.__get_tensor_prediction(wl_encode, conf, obj)
                loss = self.__loss_soo(obj_pred, obj)

                if iter > 0 and loss.item() < local_best_loss:
                    local_best_obj = obj_pred.item()
                    local_best_loss = loss.item()
                    local_best_conf = conf.data.numpy().copy()
                    local_best_iter = iter

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                constrained_numerical_knob_list = self.__get_tensor_numerical_constrained_knobs(numerical_knob_list)
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
                      f' \nat iteration {local_best_iter}, \nwith confs: {self.__get_raw_conf(local_best_conf)}')

            iter_num += iter + 1
            if local_best_loss < best_loss:
                best_obj = local_best_obj
                best_loss = local_best_loss
                best_conf = local_best_conf

        best_raw_conf = self.__get_raw_conf(best_conf)
        logging.info(f"get best {obj}: {best_obj} at {best_raw_conf} with {iter_num} iterations, loss = {best_loss}")
        if verbose:
            print()
            print("*" * 10)
            print(f"get best {obj}: {best_obj} at {best_raw_conf} with {iter_num} iterations, loss = {best_loss}")

        str1 = self.__get_seralized_conf(best_raw_conf)
        str2 = f"{obj}:{best_obj:.5f}"
        return '&'.join([str1, str2])

    def opt_scenario2(self, zmesg, lr=0.01, max_iter=100, weight_decay=0.1, patient=20, verbose=False,
                            benchmark=False, multistart=1):
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
            wl_id, obj, cst_dict = self.__input_unserialize_opt_2(zmesg)
            if wl_id is None:
                return None

            wl_encode = self.__get_tensor_wl_encode(wl_id)

            # k2k3k4_list = self.__get_constrained_k2k3k4(r_cst_dict)
            # normalized_k2k3k4_list = self.__get_normalized_k2k3k4_list(k2k3k4_list)

            # if len(normalized_k2k3k4_list) == 0:
            #     logging.warning(NOT_FOUND_ERROR + "on Resource objectives")
            #     if verbose:
            #         print(NOT_FOUND_ERROR + "on Resource objectives")
            #     return "R_miss" if benchmark else "not_found"

            best_obj_dict = None
            best_loss = np.inf
            best_conf = None
            iter_num = 0

            # for idx1, k2k3k4 in enumerate(normalized_k2k3k4_list):
            for k7 in [0, 1]:
                if verbose:
                    print('-' * 10)
                    print(f"k7: {k7}")

                # numerical_knob_list = self.__get_tensor_numerical_knobs([0.5] * len(NUMERICAL_KNOB_IDS))
                numerical_knob_list = torch.rand(len(NUMERICAL_KNOB_IDS), device=self.device, dtype=self.dtype, requires_grad=True)
                optimizer = optim.Adam([numerical_knob_list], lr=lr, weight_decay=weight_decay)

                local_best_iter = 0
                local_best_obj_dict = None
                local_best_loss = np.inf
                local_best_conf = None

                iter = 0
                for iter in range(max_iter):
                    conf = self.__get_tensor_conf_cat(numerical_knob_list, k7)
                    obj_pred_dict = {cst_obj: self.__get_tensor_prediction(wl_encode, conf, cst_obj) for cst_obj in
                                     cst_dict}
                    loss = self.__loss_moo(wl_id, conf, target_obj=obj, pred_dict=obj_pred_dict, cst_dict=cst_dict)

                    if iter > 0 and loss.item() < local_best_loss:
                        local_best_loss = loss.item()
                        local_best_obj_dict = {k: v.item() for k, v in obj_pred_dict.items()}
                        local_best_conf = conf.data.numpy().copy()
                        local_best_iter = iter

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    constrained_numerical_knob_list = self.__get_tensor_numerical_constrained_knobs(numerical_knob_list)
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
                          f' \nat iteration {local_best_iter}, \nwith confs: {self.__get_raw_conf(local_best_conf)}')

                iter_num += iter + 1
                if self.__bound_check(pred_dict=local_best_obj_dict, cst_dict=cst_dict):
                    if local_best_loss < best_loss:
                        best_obj_dict = local_best_obj_dict
                        best_loss = local_best_loss
                        best_conf = local_best_conf

            if self.__bound_check(pred_dict=best_obj_dict, cst_dict=cst_dict):
                best_raw_conf = self.__get_raw_conf(best_conf)
                obj_pred_dict = {}
                for cst_obj in cst_dict:
                    if cst_obj in R_OBJ_LIST:
                        obj_pred_dict[cst_obj] = self.__get_r_obj(cst_obj, best_raw_conf)
                    elif cst_obj in NN_OBJ_LIST:
                        obj_pred_dict[cst_obj] = best_obj_dict[cst_obj]
                    else:
                        raise Exception(f"{cst_obj} cannot be found")

                ret = self.__get_opt2_ret(best_raw_conf, obj_pred_dict)
                if verbose:
                    print()
                    print("*" * 10)
                    print(f"get {obj_pred_dict}")
                target_obj_val.append(obj_pred_dict[obj])
            else:
                if verbose:
                    print(NOT_FOUND_ERROR + "on NN objectives")
                logging.warning(NOT_FOUND_ERROR + "on NN objectives")
                ret = "not_found"
                target_obj_val.append(np.inf)
                # return "NN_miss" if benchmark else "not_found"
            ret_list.append(ret)
        idx = np.argmin(target_obj_val)
        return ret_list[idx]

    def forward(self, fixed_weights, wl_encode, conf, obj):
        X = torch.cat([conf, wl_encode]).view(1, -1)
        n_layers = len(fixed_weights) // 2
        for i in range(n_layers):
            X = X.mm(fixed_weights[f'w{i}']) + fixed_weights[f'b{i}'].view(1, -1)
            X = F.relu(X)
        # x = X[0, 0]
        if obj == "simulated_cost":
            cores = self.__get_tensor_r_obj(obj="cores", conf = conf)
            X *= cores.view(-1, 1) * self.cost_rate1
        if obj == "cpu":
            return X[0,0] / CPU_SCALE
        elif obj == "network":
            return X[0,0] / NETWORK_SCALE
        else:
            return X[0, 0]

    def forward_minibatch(self, fixed_weights, wl_encode, conf, obj):
        # new_size = (conf.shape[0], -1)
        n_batch = conf.shape[0]
        wk = wl_encode.shape[0]
        X = torch.cat([conf, wl_encode * torch.ones((n_batch, wk), device=self.device, dtype=self.dtype)], dim=1)
        n_layers = len(fixed_weights) // 2
        for i in range(n_layers):
            X = X.mm(fixed_weights[f'w{i}']) + fixed_weights[f'b{i}'].view(1, -1)
            X = F.relu(X)
        # X.shape = n_batch x 1
        if obj == "simulated_cost":
            cores = self.__get_tensor_r_obj(obj="cores", conf = conf)
            X *= cores.view(-1, 1) * self.cost_rate1
        if obj == "cpu":
            return X / CPU_SCALE
        elif obj == "network":
            return X / NETWORK_SCALE
        return X

    @staticmethod
    def __loss_soo(obj_pred, obj):
        # for single job objective, the loss can be its own value.
        if obj in NN_OBJ_LIST:
            return obj_pred ** 2
        elif obj == "cores":
            return (obj_pred - 4) ** 2
        else:
            raise Exception(f"{obj} not found")

    def get_tensor_gpr_std(self, wl_id, conf, obj):
        X_train, _, K_inv = self.gpr_metr_map[wl_id]
        y_scale = self.scale_y_map[wl_id][obj] # float
        if conf.ndim < 2:
            conf = conf.reshape(1, -1)
        std = self.gpr_model.objective_std(X_test=conf, X_train=X_train, K_inv=K_inv, y_scale=y_scale)
        return std

    def __loss_soo_minibatch(self, wl_id, conf, obj, obj_pred):
        # for single job objective, the loss can be its own value.
        if not self.accurate and obj in GP_OBJ_LIST:
            std = self.get_tensor_gpr_std(wl_id, conf, obj).reshape(-1, 1)
            loss = (obj_pred + std * self.alpha) ** 2
        else:
            loss = obj_pred ** 2
        return torch.min(loss), torch.argmin(loss)

    # normed loss, stress outside ranges
    def __loss_moo(self, wl_id, conf, target_obj, pred_dict, cst_dict):
        loss = torch.tensor(0, device=self.device, dtype=self.dtype)
        for cst_obj, [lower, upper] in cst_dict.items():
            cst_obj_pred_raw = pred_dict[cst_obj]

            if not self.accurate and cst_obj in GP_OBJ_LIST:
                std = self.get_tensor_gpr_std(wl_id, conf, cst_obj).squeeze()
                cst_obj_pred = cst_obj_pred_raw + std * self.alpha
            else:
                cst_obj_pred = cst_obj_pred_raw

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

    #############################
    ##### unserialize zmesg #####
    #############################

    def __input_unserialize_predict(self, zmesg):
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
            conf_norm_val = self.__get_normalized_conf(conf_raw_val)
        except:
            logging.error(FORMAT_ERROR + f'{zmesg}')
            print(FORMAT_ERROR + f'{zmesg}')
            return None, None, None

        if not self.__wl_check(wl_id):
            return None, None, None

        if not self.__obj_check(obj):
            return None, None, None

        if not self.__conf_check(conf=conf_norm_val):
            return None, None, None

        return wl_id, obj, conf_norm_val

    def __input_unserialize_opt_1(self, zmesg):
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

        if not self.__wl_check(wl_id):
            return None, None

        if not self.__obj_check(obj):
            return None, None

        return wl_id, obj

    def __input_unserialize_opt_2(self, zmesg):
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
                    cst_dict[sub_obj] = self.__get_tensor_bounds_list(lower, upper)
                else:
                    logging.warning(f"{sub_kv[0]} is unrecognized.")
        except:
            logging.error(FORMAT_ERROR + f'{zmesg}')
            print(FORMAT_ERROR + f'{zmesg}')
            return None, None, None

        if not self.__wl_check(wl_id):
            return None, None, None

        if not self.__obj_check(obj):
            return None, None, None

        for cst_obj in cst_dict:
            if not self.__obj_check(cst_obj):
                return None, None, None

        return wl_id, obj, cst_dict

    ################################
    ##### get_tensor functions #####
    ################################

    def __get_tensor_wl_encode(self, wl_id):
        return torch.tensor(self.__get_wl_encode(wl_id), device=self.device, dtype=self.dtype)

    def __get_tensor_conf_for_predict(self, conf_val_list):
        return torch.tensor(conf_val_list, device=self.device, dtype=self.dtype)

    def __get_tensor_numerical_knobs(self, numerical_knob_list):
        assert len(numerical_knob_list) == len(NUMERICAL_KNOB_IDS)
        return torch.tensor(numerical_knob_list, device=self.device, dtype=self.dtype, requires_grad=True)

    def __get_tensor_conf_cat(self, numerical_knob_list, k7):
        # t_k2k3k4 = torch.tensor(k2k3k4, device=self.device, dtype=self.dtype)
        t_k7 = torch.tensor(k7, device=self.device, dtype=self.dtype)
        # [k1, k2, k3, k4, k5, k6, k8, s1, s2, s3, s4]
        conf = torch.cat([numerical_knob_list[:6],
                          t_k7.view(1), numerical_knob_list[6:]])
        return conf

    def __get_tensor_conf_cat_minibatch(self, numerical_knob_list, k7):
        # new_size = (numerical_knob_list.shape[0], -1)
        n_batch = numerical_knob_list.shape[0]
        t_k7 = torch.ones((n_batch, 1), device=self.device, dtype=self.dtype) if k7 == 1 \
            else torch.zeros((n_batch, 1), device=self.device, dtype=self.dtype)
        # [k1, k2, k3, k4, k5, k6, k8, s1, s2, s3, s4]
        conf = torch.cat([numerical_knob_list[:, :6], t_k7, numerical_knob_list[:, 6:]], dim=1)
        return conf

    def __get_tensor_numerical_constrained_knobs(self, numerical_knob_list):
        # Tensor[k1, k2, k3, k4, k5, k6, k8, s1, s2, s3, s4]
        conf_cons_max_ = self.conf_cons_max_norm[NUMERICAL_KNOB_IDS]
        conf_cons_min_ = self.conf_cons_min_norm[NUMERICAL_KNOB_IDS]
        bounded_np = np.array([self.__get_bounded(k.item(), lower=conf_cons_min_[kid], upper=conf_cons_max_[kid]) for kid, k in enumerate(numerical_knob_list)])
        raw_np = self.__get_raw_conf(bounded_np, normalized_ids=NUMERICAL_KNOB_IDS)
        normalized_np = self.__get_normalized_conf(raw_np, normalized_ids=NUMERICAL_KNOB_IDS)
        return torch.tensor(normalized_np, device=self.device, dtype=self.dtype)

    # def __get_tensor_numerical_constrained_knobs_minibatch(self, numerical_knob_list):
    #     # Tensor n_batch x len(NUMERICAL_KNOB_IDS)
    #     # [k1, k5, k6, k8, s1, s2, s3, s4]
    #     # bounded_np = np.array([self.__get_bounded(k.item()) for k in numerical_knob_list])
    #     numerical_knob_list[numerical_knob_list > 1] = 1
    #     numerical_knob_list[numerical_knob_list < 0] = 0
    #     raw_np = self.__get_raw_conf(numerical_knob_list.numpy(), normalized_ids=NUMERICAL_KNOB_IDS)
    #     normalized_np = self.__get_normalized_conf(raw_np, normalized_ids=NUMERICAL_KNOB_IDS)
    #     return torch.tensor(normalized_np, device=self.device, dtype=self.dtype)

    def __get_tensor_numerical_constrained_knobs_minibatch(self, numerical_knob_list):
        # Tensor n_batch x len(NUMERICAL_KNOB_IDS)
        # Tensor[k1, k2, k3, k4, k5, k6, k8, s1, s2, s3, s4]

        conf_cons_max_ = self.conf_cons_max_norm[NUMERICAL_KNOB_IDS]
        conf_cons_min_ = self.conf_cons_min_norm[NUMERICAL_KNOB_IDS]
        bounded_np = np.array([self.__get_bounded(k.numpy(), lower=conf_cons_min_, upper=conf_cons_max_) for k in numerical_knob_list])
        # return torch.tensor(bounded_np, device=self.device, dtype=self.dtype)
        # numerical_knob_list[numerical_knob_list > 1] = 1
        # numerical_knob_list[numerical_knob_list < 0] = 0
        # return numerical_knob_list
        raw_np = self.__get_raw_conf(bounded_np, normalized_ids=NUMERICAL_KNOB_IDS)
        normalized_np = self.__get_normalized_conf(raw_np, normalized_ids=NUMERICAL_KNOB_IDS)
        return torch.tensor(normalized_np, device=self.device, dtype=self.dtype)


    def __get_tensor_bounds_list(self, lower, upper):
        t_lower = torch.tensor(lower, device=self.device, dtype=self.dtype)
        t_upper = torch.tensor(upper, device=self.device, dtype=self.dtype)
        return [t_lower, t_upper]

    def __get_tensor_r_obj(self, obj, conf):
        if obj == "cores":
            k2_max, k3_max = self.conf_max[1:3]
            k2_min, k3_min = self.conf_min[1:3]
            k2_norm, k3_norm = conf[1:3]
            k2 = k2_norm * (k2_max - k2_min) + k2_min
            k3 = k3_norm * (k3_max - k3_min) + k3_min
            n_exec = torch.min(torch.floor(torch.tensor(58, device=self.device, dtype=self.dtype) / k3), k2)
            return n_exec * k3
        else:
            raise Exception(f"{obj} cannot be found")

    def __get_tensor_r_obj_minibatch(self, obj, conf):
        if obj == "cores":
            k2_max, k3_max = self.conf_max[1:3]
            k2_min, k3_min = self.conf_min[1:3]
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

    def __get_tensor_simulated_cost2(self, wl_encode, conf):
        latency_pred = self.forward(self.model_weights_dict["latency"], wl_encode, conf, "latency")
        cores_pred = self.__get_tensor_r_obj("cores", conf)
        net_pkt_pred = self.forward(self.model_weights_dict["network"], wl_encode, conf, "network") * self.avg_pkt_size
        obj_predict = self.cost_rate1 * cores_pred * latency_pred + self.cost_rate2 * latency_pred * net_pkt_pred
        return obj_predict

    def __get_tensor_simulated_cost2_minibatch(self, wl_encode, conf):
        latency_pred = self.forward_minibatch(self.model_weights_dict["latency"], wl_encode, conf, "latency")
        cores_pred = self.__get_tensor_r_obj_minibatch("cores", conf)
        net_pkt_pred = self.forward_minibatch(self.model_weights_dict["network"], wl_encode, conf, "network") * self.avg_pkt_size
        obj_predict = self.cost_rate1 * cores_pred * latency_pred + self.cost_rate2 * latency_pred * net_pkt_pred
        return obj_predict

    def __get_tensor_simulated_cost3(self, wl_encode, conf):
        latency_pred = self.forward(self.model_weights_dict["latency"], wl_encode, conf, "latency")
        cores_pred = self.__get_tensor_r_obj("cores", conf)
        ops_pred = self.forward(self.model_weights_dict["ops"], wl_encode, conf, "ops")
        obj_predict = self.cost_rate1 * cores_pred * latency_pred / 1000 / 3600 + self.cost_rate3 * ops_pred
        return obj_predict

    def __get_tensor_simulated_cost3_minibatch(self, wl_encode, conf):
        latency_pred = self.forward_minibatch(self.model_weights_dict["latency"], wl_encode, conf, "latency")
        cores_pred = self.__get_tensor_r_obj_minibatch("cores", conf)
        ops_pred = self.forward_minibatch(self.model_weights_dict["ops"], wl_encode, conf, "ops")
        obj_predict = self.cost_rate1 * cores_pred * latency_pred / 1000 / 3600 + self.cost_rate3 * ops_pred
        return obj_predict

    def __get_tensor_prediction(self, wl_encode, conf, obj):
        if obj == "simulated_cost2":
            obj_predict = self.__get_tensor_simulated_cost2(wl_encode, conf)
        elif obj == "simulated_cost3":
            obj_predict = self.__get_tensor_simulated_cost3(wl_encode, conf)
        elif obj in NN_OBJ_LIST:
            fixed_weights = self.model_weights_dict[obj]
            obj_predict = self.forward(fixed_weights, wl_encode, conf, obj)
        else:
            obj_predict = self.__get_tensor_r_obj(obj, conf)
        return obj_predict

    def __get_tensor_prediction_minibatch(self, wl_encode, conf, obj):
        if obj == "simulated_cost2":
            obj_predict = self.__get_tensor_simulated_cost2_minibatch(wl_encode, conf)
        elif obj == "simulated_cost3":
            obj_predict = self.__get_tensor_simulated_cost3_minibatch(wl_encode, conf)
        elif obj in NN_OBJ_LIST:
            fixed_weights = self.model_weights_dict[obj]
            obj_predict = self.forward_minibatch(fixed_weights, wl_encode, conf, obj)
        else:
            obj_predict = self.__get_tensor_r_obj_minibatch(obj, conf)
        return obj_predict

    def __unwrap_weights(self, fixed_weights):
        fixed_weights_dict = {}
        n_layers = len(fixed_weights) // 2
        for i in range(n_layers):
            fixed_weights_dict[f'w{i}'] = torch.from_numpy(fixed_weights[2 * i]) \
                .to(device=self.device, dtype=self.dtype)
            fixed_weights_dict[f'b{i}'] = torch.from_numpy(fixed_weights[2 * i + 1]) \
                .to(device=self.device, dtype=self.dtype)
        return fixed_weights_dict

    ###########################
    ##### "get" funcitons #####
    ###########################

    # def __get_normalized_k2k3k4_list(self, k2k3k4_list):
    #     k2k3k4_max = np.array(self.conf_max[1:4])
    #     k2k3k4_min = np.array(self.conf_min[1:4])
    #     normalized_k2k3k4_list = []
    #     for k2k3k4 in k2k3k4_list:
    #         normalized_k2k3k4 = (k2k3k4 - k2k3k4_min) / (k2k3k4_max - k2k3k4_min)
    #         normalized_k2k3k4_list.append(list(normalized_k2k3k4))
    #     return normalized_k2k3k4_list

    # def __get_constrained_k2k3k4(self, r_cst_dict):
    #     k2k3k4_list = K2K3K4_LIST
    #     for k, [lower, upper] in r_cst_dict.items():
    #         if k == 'cores':
    #             k2k3k4_list = [[k2,k3,k4] for k2, k3, k4 in k2k3k4_list if upper.item() >= self.__get_core_num(k2, k3) >= lower.item()]
    #     return k2k3k4_list

    def __get_wl_encode(self, wl_id):
        return self.wl_encode_list[self.wlId_to_alias_dict[wl_id]]

    def __get_normalized_conf(self, raw_conf, normalized_ids=None):
        """
        :param real_conf: numpy.array[int]
        :return: normalized to 0-1
        """
        conf_max = self.conf_max if normalized_ids is None else self.conf_max[normalized_ids]
        conf_min = self.conf_min if normalized_ids is None else self.conf_min[normalized_ids]
        normalized_conf = (raw_conf - conf_min) / (conf_max - conf_min)
        return normalized_conf

    def __get_raw_conf(self, normalized_conf, normalized_ids=None):
        """
        :param normalized_conf: numpy.array[float] [0,1]
        :return: numpy.array[int]
        """
        conf_max = self.conf_max if normalized_ids is None else self.conf_max[normalized_ids]
        conf_min = self.conf_min if normalized_ids is None else self.conf_min[normalized_ids]
        conf = normalized_conf * (conf_max - conf_min) + conf_min
        get_raw_conf = conf.round()
        return get_raw_conf

    @staticmethod
    def __get_core_num(k2, k3):
        return min(math.floor(58 / k3), k2) * k3

    def __get_r_obj(self, obj, conf):
        if obj == "cores":
            k2, k3 = conf[1:3]
            return self.__get_core_num(k2, k3)

    def __get_seralized_conf(self, conf):
        s_conf = ';'.join([f'{self.knob_list[i]}:{int(conf[i])}' for i in range(len(self.knob_list))])
        return s_conf

    def __get_opt2_ret(self, conf, pred_dict):
        str1 = self.__get_seralized_conf(conf)
        str2 = ';'.join([f'{k}:{v:.5f}' for k, v in pred_dict.items()])
        return f'{str1}&{str2}'

    @staticmethod
    def __get_bounded(k, lower=0.0, upper=1.0):
        k = np.maximum(k, lower)
        k = np.minimum(k, upper)
        # if k > upper:
        #     return upper
        # if k < lower:
        #     return lower
        return k

    def __load_weights(self, extracted_weights_dict):
        obj_weights_dict = {}
        for obj, obj_weights in extracted_weights_dict.items():
            obj_weights_dict[obj] = self.__unwrap_weights(obj_weights)
        # adding weights for simulated_cost, same as latency
        obj_weights_dict['simulated_cost'] = self.__unwrap_weights(extracted_weights_dict['latency'])
        obj_weights_dict['simulated_cost2'] = None
        obj_weights_dict['simulated_cost3'] = None
        return obj_weights_dict

    @staticmethod
    def __load_obj(name):
        print(os.getcwd())
        with open('batch_confs/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    ##################################
    ##### format check functions #####
    ##################################

    def __wl_check(self, wl_id):
        if wl_id not in self.wlId_to_alias_dict:
            logging.error(f'ERROR: workload {wl_id} is not found')
            print(f'ERROR: workload {wl_id} is not found')
            return False
        return True

    @staticmethod
    def __obj_check(obj):
        if obj not in NN_OBJ_LIST + R_OBJ_LIST:
            logging.error(f'ERROR: objective {obj} is not found')
            print(f'ERROR: objective {obj} is not found')
            return False
        return True

    @staticmethod
    def __conf_check(conf):
        if conf.min() < 0 or conf.max() > 1:
            logging.error(f'ERROR: knob value out of range, check conf_min and conf_max')
            print(f'ERROR: knob value out of range, check conf_min and conf_max')
            return False
        return True

    def __bound_check(self, pred_dict, cst_dict):
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
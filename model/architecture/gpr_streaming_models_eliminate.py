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

DEFAULT_RIDGE=1.0
FORMAT_ERROR = 'ERROR: format of zmesg is not correct'
NOT_FOUND_ERROR = 'no valid configuration found'
NN_OBJ_LIST = [
    "latency",
    "throughput"
]
R_OBJ_LIST = [
    'cost'
]
GP_OBJ_LIST = [
    'latency',
    'throughput'
]

C1 = 21.5
C2 = 30
NUMERICAL_KNOB_IDS = [0, 1, 2, 3, 6, 7, 8, 9]
SEED = 0
INPUTRATE_SCALE = 1
INPUTRATE_IDX = 3
MF_SCALE = 100
MF_IDX = 8

MULTIVAR_CONS = False

class GPR_Streaming_models:
    def __init__(self, stress=10):
        torch.set_num_threads(1)
        self.device, self.dtype = torch.device('cpu'), torch.float32
        self.stress = stress
        self.knob_list = [
            "batchInterval",
            "blockInterval",
            "parallelism",
            "inputRate",
            "broadcastCompressValues",  # --> binary
            "rddCompressValues",  # --> binary
            "maxSizeInFlightValues",
            "bypassMergeThresholdValues",
            "memoryFractionValues",
            "executorMemoryValues"
        ]
        self.model = GPRPT(gp_obj_list=GP_OBJ_LIST)
        gpr_data_ = self._load_obj("gpr_data_streaming")
        data_ = gpr_data_['data']
        proxy_jobs_ = gpr_data_['proxy_jobs']
        wl_list_ = ["10", "14", "21", "23", "26", "30", "34", "41", "42", "44", "49", "54", "56", "60", "67"]
        # get this information from https://github.com/shenoy1/UDAO/blob/ZMQ_connection/MOO/src/main/java/concrete/iid/enter/Configuration.java
        self.conf_min_moo = np.array([1, 100, 18, 100000, 0, 0, 24, 10, 0.4, 512])
        self.conf_max_moo = np.array([10, 1000, 90, 1200000, 1, 1, 96, 200, 0.8, 6144]) # 1200000 --> 1500000
        self.conf_min_moo[MF_IDX] *= MF_SCALE
        self.conf_max_moo[MF_IDX] *= MF_SCALE
        self.conf_min_moo[INPUTRATE_IDX] *= INPUTRATE_SCALE
        self.conf_max_moo[INPUTRATE_IDX] *= INPUTRATE_SCALE

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

    def _loss_soo_minibatch(self, wl_id, conf, obj, obj_pred, conf_max, conf_min):
        # for single job objective, the loss can be its own value.
        loss = obj_pred ** 2 * self._get_direction(obj)

        if MULTIVAR_CONS:
            # batch_interval
            bi = self._get_tensor_r_obj_minibatch("batch_interval", conf, conf_max, conf_min)

            if obj == "latency":
                loss += self._constraintSlideWindowAndLatency_penalty_minibatch(wl_id, bi, obj_pred)
            else:
                latency_pred = self._get_tensor_prediction_minibatch(wl_id, conf, "latency")
                loss += self._constraintSlideWindowAndLatency_penalty_minibatch(wl_id, bi, latency_pred)

        return torch.min(loss), torch.argmin(loss)

    def _constraintSlideWindowAndLatency_penalty(self, wl_id, bi, latency):
        if (26 <= int(wl_id) <= 31) or (56 <= int(wl_id) <= 67):
            bound = bi * 1000
        else:
            bound = torch.tensor(10000, dtype=self.dtype, device=self.device)

        if latency < bound:
            return torch.tensor(0, dtype=self.dtype, device=self.device)
        else:
            return (bound - latency) ** 2 + self.stress

    def _constraintSlideWindowAndLatency_penalty_minibatch(self, wl_id, bi, latency):
        n_batch = bi.shape[0]
        if (26 <= int(wl_id) <= 31) or (56 <= int(wl_id) <= 67):
            bound = bi * 1000
        else:
            bound = torch.tensor(10000, dtype=self.dtype, device=self.device).repeat(n_batch, 1)

        penalty = (bound - latency) ** 2 + self.stress
        penalty[latency < bound] = 0
        return penalty

    # normed loss, stress outside ranges
    def _loss_moo(self, wl_id, conf, target_obj, pred_dict, cst_dict, conf_max, conf_min):
        loss = torch.tensor(0, device=self.device, dtype=self.dtype)
        for cst_obj, [lower, upper] in cst_dict.items():
            cst_obj_pred = pred_dict[cst_obj]
            if upper != lower:
                norm_cst_obj_pred = (cst_obj_pred - lower) / (upper - lower)
                if cst_obj == target_obj:
                    if norm_cst_obj_pred < 0 or norm_cst_obj_pred > 1:
                        add_loss = (norm_cst_obj_pred - 0.5) ** 2 + self.stress
                    else:
                        add_loss = norm_cst_obj_pred * self._get_direction(target_obj)
                else:
                    if norm_cst_obj_pred < 0 or norm_cst_obj_pred > 1:
                        add_loss = (norm_cst_obj_pred - 0.5) ** 2 + self.stress
                    else:
                        add_loss = torch.tensor(0, device=self.device, dtype=self.dtype)
            else:
                add_loss = (cst_obj_pred - upper) ** 2

            loss += add_loss

        if MULTIVAR_CONS:
            bi = self._get_tensor_r_obj("batch_interval", conf, conf_max, conf_min)
            loss += self._constraintSlideWindowAndLatency_penalty(wl_id, bi, pred_dict['latency'])
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
    def opt_scenario1(self, zmesg, bs=16, lr=0.01, max_iter=100, weight_decay=0.1, patient=30, verbose=False):
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

        for broadcastCompressValues in [0, 1]:
            for rddCompressValues in [0, 1]:
                if verbose:
                    print('-' * 10)
                    print(f"broadcastCompressValues: {broadcastCompressValues}, rddCompressValues: {rddCompressValues}")
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
                    conf = self._get_tensor_conf_cat_minibatch(numerical_knob_list, broadcastCompressValues, rddCompressValues)
                    obj_pred = self._get_tensor_prediction_minibatch(wl_id, conf, obj) # Nx1
                    loss, loss_id = self._loss_soo_minibatch(wl_id, conf, obj, obj_pred, conf_max, conf_min)

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


    def opt_scenario2(self, zmesg, lr=0.01, max_iter=100, weight_decay=0.1, patient=20, verbose=False,
                            benchmark=False):
        """
        minimize a target objective value given a workload with K constrained objectives.
        Return the best configuration and its obj values. E.g.,
        ret="k1:8;k2:3;k3:2;k4:4;k5:384;k6:217;k7:1;k8:70;s1:50000;s2:256;s3:10;s4:8;latency:1010;cores:18"
        :param zmesg: see batch_knob_format.md
        :return: the conf and its obj values
        """
        # assume obj is got from NN model
        wl_id, obj, cst_dict = self._input_unserialize_opt_2(zmesg)
        if wl_id is None:
            return None
        conf_max, conf_min = self._get_conf_range(wl_id)

        best_obj_dict = None
        best_loss = np.inf
        best_conf = None
        iter_num = 0

        # for idx1, k2k3k4 in enumerate(normalized_k2k3k4_list):
        for broadcastCompressValues in [0, 1]:
            for rddCompressValues in [0, 1]:
                if verbose:
                    print('-' * 10)
                    print(f"broadcastCompressValues: {broadcastCompressValues}, rddCompressValues: {rddCompressValues}")

                numerical_knob_list = self._get_tensor([0.5] * len(NUMERICAL_KNOB_IDS), requires_grad=True)
                optimizer = optim.Adam([numerical_knob_list], lr=lr, weight_decay=weight_decay)

                local_best_iter = 0
                local_best_obj_dict = None
                local_best_loss = np.inf
                local_best_conf = None

                iter = 0
                for iter in range(max_iter):
                    conf = self._get_tensor_conf_cat(numerical_knob_list, broadcastCompressValues, rddCompressValues)
                    obj_pred_dict = {cst_obj: self._get_tensor_prediction(wl_id, conf, cst_obj) for cst_obj in cst_dict}
                    loss = self._loss_moo(wl_id, conf, obj, obj_pred_dict, cst_dict, conf_max, conf_min)

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
            pass
        else:
            if verbose:
                print(NOT_FOUND_ERROR + "on GP objectives")
            logging.warning(NOT_FOUND_ERROR + "on GP objectives")
            return "GP_miss" if benchmark else "not_found"

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
        return ret

    def opt_scenario3(self, zmesg, lr=0.01, max_iter=100, weight_decay=0.1, patient=20, verbose=False, benchmark=False,
                      processes=1):
        torch.manual_seed(SEED)
        zmesg_list = zmesg.split('|')
        arg_list = [(z, lr, max_iter, weight_decay, patient, verbose, benchmark) for z in zmesg_list]

        with Pool(processes=processes) as pool:
            ret_list = pool.starmap(self.opt_scenario2, arg_list)
        ret = '|'.join(ret_list)

        return ret


    ################################
    ##### get_tensor functions #####
    ################################

    def _get_tensor_r_obj(self, obj, conf, conf_max, conf_min):
        # conf is a tensor list
        if obj == "cost":
            # cost = C1 * batchInterval + C2 * parallelism
            # C1 = 21.5, C2 = 30
            conf_0_max, conf_2_max = conf_max[0], conf_max[2]
            conf_0_min, conf_2_min = conf_min[0], conf_min[2]
            conf_0_norm, conf_2_norm = conf[0], conf[2]
            conf_0 = conf_0_norm * (conf_0_max - conf_0_min) + conf_0_min
            conf_2 = conf_2_norm * (conf_2_max - conf_2_min) + conf_2_min
            cost = C1 * conf_0 + C2 * conf_2
            return cost
        elif obj == "batch_interval":
            conf_max, conf_min = conf_max[0], conf_min[0]
            conf_norm = conf[0]
            conf = conf_norm * (conf_max - conf_min) + conf_min
            return conf
        else:
            raise Exception(f"{obj} cannot be found")

    def _get_tensor_r_obj_minibatch(self, obj, conf, conf_max, conf_min):
        # conf is a tensor list
        if obj == "cost":
            # cost = C1 * batchInterval + C2 * parallelism
            # C1 = 21.5, C2 = 30
            conf_0_max, conf_2_max = conf_max[0], conf_max[2]
            conf_0_min, conf_2_min = conf_min[0], conf_min[2]
            conf_0_norm, conf_2_norm = conf[:, 0], conf[:, 2]
            conf_0 = conf_0_norm * (conf_0_max - conf_0_min) + conf_0_min
            conf_2 = conf_2_norm * (conf_2_max - conf_2_min) + conf_2_min
            cost = C1 * conf_0 + C2 * conf_2
            return cost.view(-1, 1)
        elif obj == "batch_interval":
            conf_max, conf_min = conf_max[0], conf_min[0]
            conf_norm = conf[:, 0]
            conf = conf_norm * (conf_max - conf_min) + conf_min
            return conf.view(-1, 1)
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
            obj_pred = self.model.objective(conf, X_train, y_train, K_inv).view(-1, 1)
        elif obj in R_OBJ_LIST:
            obj_pred = self._get_tensor_r_obj_minibatch(obj, conf, conf_max=conf_max, conf_min=conf_min)
        else:
            raise Exception(f'{obj} cannot be found')
        return obj_pred

    def _get_tensor_conf_cat(self, numerical_knob_list, broadcastCompressValues, rddCompressValues):
        t_bb = torch.tensor([broadcastCompressValues, rddCompressValues], device=self.device, dtype=self.dtype)
        # to a kxk_streaming knob order
        conf = torch.cat([numerical_knob_list[:4], t_bb, numerical_knob_list[4:]])
        return conf

    def _get_tensor_conf_cat_minibatch(self, numerical_knob_list, broadcastCompressValues, rddCompressValues):
        n_batch = numerical_knob_list.shape[0]
        t_bb = torch.tensor([broadcastCompressValues, rddCompressValues], device=self.device, dtype=self.dtype)\
            .repeat(n_batch, 1)
        conf = torch.cat([numerical_knob_list[:, :4], t_bb, numerical_knob_list[:, 4:]], dim=1)
        return conf

    def _get_tensor_bounds_list(self, lower, upper):
        t_lower = torch.tensor(lower, device=self.device, dtype=self.dtype)
        t_upper = torch.tensor(upper, device=self.device, dtype=self.dtype)
        return [t_lower, t_upper]

    def _get_tensor_numerical_constrained_knobs(self, numerical_knob_list, conf_max, conf_min):
        bounded_np = np.array([self._get_bounded(k.item()) for k in numerical_knob_list])
        raw_np = self._get_raw_conf(bounded_np, conf_max, conf_min, normalized_ids=NUMERICAL_KNOB_IDS)
        normalized_np = self._get_normalized_conf(raw_np, conf_max, conf_min, normalized_ids=NUMERICAL_KNOB_IDS)
        return torch.tensor(normalized_np, device=self.device, dtype=self.dtype)

    def _get_tensor_numerical_constrained_knobs_minibatch(self, numerical_knob_list, conf_max, conf_min):
        # Tensor n_batch x len(NUMERICAL_KNOB_IDS)
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

    @staticmethod
    def _get_direction(obj):
        if obj == "throughput":
            return -1
        else:
            return 1

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


            # Scale to (0, 1)
            X_matrix[:, MF_IDX] *= MF_SCALE
            X_scaler = MinMaxScaler()
            # X_scaled = X_scaler.fit_transform(X_matrix)

            X_scaler.fit(np.concatenate([self.conf_max_moo.reshape(1, -1), self.conf_min_moo.reshape(1, -1)], axis=0))
            X_scaled = X_scaler.transform(X_matrix)

            # in streaming case, the latency was generated as ms in the PKL file
            # y_dict['latency'] *= 1000
            model_map[wl] = self.model.fit(X_train=X_scaled, y_dict=y_dict, ridge=DEFAULT_RIDGE)
            scaler_map[wl] = X_scaler
        return model_map, scaler_map

    def _get_conf_range(self, wl_id):
        conf_max = self.scaler_map[wl_id].data_max_
        conf_min = self.scaler_map[wl_id].data_min_
        return conf_max, conf_min

    def _get_seralized_conf(self, conf):
        conf[MF_IDX] /= MF_SCALE
        conf[INPUTRATE_IDX] /= INPUTRATE_SCALE

        conf_s = []
        for i in range(len(self.knob_list)):
            if i != 8:
                conf_s.append(f'{self.knob_list[i]}:{int(conf[i])}')
            else:
                conf_s.append(f'{self.knob_list[i]}:{conf[i]:.2f}')
        return ';'.join(conf_s)

    def _get_r_obj(self, obj, conf):
        """
        :param obj:
        :param conf: np.array, raw value
        :return:
        """
        if obj == "cost":
            conf_0, conf_2 = conf[0], conf[2]
            return C1 * conf_0 + C2 * conf_2

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
        E.g., "JobID:10;Objective:Latency;batchInterval:1;blockInterval:100;parallelism:18;inputRate:100000;
                maxSizeInFlightValues:24;bypassMergeThresholdValues:10;memoryFractionValues:40;
                executorMemoryValues:1024;rddCompressValues:1;broadcastCompressValues:0"
        :param zmesg:
        :return: wl_id, obj, conf_norm_val
        """
        try:
            kv_dict = {kv.split(":")[0]: kv.split(":")[1] for kv in zmesg.split(";")}
            wl_id = kv_dict['JobID']
            obj = kv_dict['Objective']
            conf_raw_val = np.array([float(kv_dict[k]) for k in self.knob_list])
            conf_raw_val[MF_IDX] *= MF_SCALE
            conf_raw_val[INPUTRATE_IDX] *= INPUTRATE_SCALE
            conf_max, conf_min = self._get_conf_range(wl_id)
            conf_norm_val = self._get_normalized_conf(conf_raw_val, conf_max=conf_max, conf_min=conf_min)
        except:
            logging.error(FORMAT_ERROR + f'{zmesg}')
            print(FORMAT_ERROR + f'{zmesg}')
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
    sm = GPR_Streaming_models()
    for zmesg in [
        "JobID:10;Objective:latency;batchInterval:1;blockInterval:100;parallelism:18;inputRate:100000;maxSizeInFlightValues:24;bypassMergeThresholdValues:10;memoryFractionValues:0.4;executorMemoryValues:1024;rddCompressValues:1;broadcastCompressValues:0",
        "JobID:10;Objective:throughput;batchInterval:1;blockInterval:100;parallelism:18;inputRate:100000;maxSizeInFlightValues:24;bypassMergeThresholdValues:10;memoryFractionValues:0.4;executorMemoryValues:1024;rddCompressValues:1;broadcastCompressValues:0",
        "JobID:56;Objective:latency;batchInterval:1;blockInterval:100;parallelism:18;inputRate:100000;maxSizeInFlightValues:24;bypassMergeThresholdValues:10;memoryFractionValues:0.4;executorMemoryValues:1024;rddCompressValues:1;broadcastCompressValues:0",
        "JobID:56;Objective:throughput;batchInterval:1;blockInterval:100;parallelism:18;inputRate:100000;maxSizeInFlightValues:24;bypassMergeThresholdValues:10;memoryFractionValues:0.4;executorMemoryValues:1024;rddCompressValues:1;broadcastCompressValues:0"
    ]:
        print(f'-- test `predict` latency, input: {zmesg}')
        pred = sm.predict(zmesg)
        print(f'get: {sm.predict(zmesg)}\n')


    for obj in ['latency', 'throughput', 'cost']:
        zmesg = f"JobID:10;Objective:{obj}"
        print(f'-- test `opt_scenario1` to find the global minimum for {obj}')
        res = sm.opt_scenario1(zmesg=zmesg, max_iter=100, lr=0.1, verbose=False)
        print(f'out: {res}\n')

    for zmesg in [
        "JobID:10;Objective:latency;Constraint:throughput:0:400000;Constraint:latency:0000:4000",
        "JobID:10;Objective:throughput;Constraint:throughput:0:400000;Constraint:latency:0000:4000",
        "JobID:10;Objective:throughput;Constraint:throughput:0:400000;Constraint:latency:0000:4000;Constraint:cost:0:5000",
        "JobID:10;Objective:throughput;Constraint:throughput:200000:400000;Constraint:latency:0000:4000;Constraint:cost:0:5000",
        "JobID:10;Objective:throughput;Constraint:throughput:200000:400000;Constraint:latency:0000:4000;Constraint:cost:0:5000"
    ]:
        print(f"-- test `opt_scenario2` to solve the CO problem")
        print(f'input: {zmesg}')
        res = sm.opt_scenario2(zmesg, max_iter=100, lr=0.1)
        print(f'output: {res}\n')

    print(f'-- test `opt_scenario3` to solve the CO problem in parallel')
    zmesg = "JobID:10;Objective:throughput;Constraint:throughput:200000:400000;Constraint:latency:0000:4000;Constraint:cost:0:5000|JobID:10;Objective:throughput;Constraint:throughput:200000:400000;Constraint:latency:0000:4000;Constraint:cost:0:5000"
    print(f'input: {zmesg}')
    res = sm.opt_scenario3(zmesg, max_iter=100, lr=0.1, verbose=False)
    print(f'output: {res}\n')



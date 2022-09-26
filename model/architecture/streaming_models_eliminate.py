#
# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description:
#


import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle, logging, math
from multiprocessing import Pool

FORMAT_ERROR = 'ERROR: format of zmesg is not correct'
NOT_FOUND_ERROR = 'no valid configuration found'
NN_OBJ_LIST = [
    "latency",
    "throughput"
]
R_OBJ_LIST = [
    'cost'
]

C1 = 21.5
C2 = 30

# ['BatchInterval(s)',
#  'BlockInterval(ms)',
#  'Parallelism',
#  'InputRate(r/s)',
#  'broadcast_compress', --> binary
#  'rdd_compress', --> binary
#  'max_size_in_flight(MB)',
#  'bypass_merge_threshold',
#  'memory_fraction',
#  'executor_memory(MB)']

NUMERICAL_KNOB_IDS = [0, 1, 2, 3, 6, 7, 8, 9]
SEED = 30
INPUTRATE_SCALE = 1
INPUTRATE_IDX = 3
MF_SCALE = 100
MF_IDX = 8
BATCHI_IDX = 0


class Streaming_models:
    def __init__(self, stress=10):
        torch.set_num_threads(1)
        torch.random.manual_seed(SEED)
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
        wrapped_model = self.__load_obj("wrapped_streaming_model")
        self.wlId_to_alias_dict = wrapped_model["wlId_to_alias_dict"]
        self.wl_encode_list = wrapped_model["centroids"]
        self.wlId_id_sql = wrapped_model["job_id_sql"]
        self.wlId_id_ml = wrapped_model["job_id_ml"]
        self.cal_offset_dict = {
            "latency": wrapped_model["latency_cal_offset"],
            "throughput": wrapped_model["throughput_cal_offset"]
        }
        # a list of knob max/min value list from the 10 knobs
        # [1.0, 100.0, 18.0, 199998.0, 0.0, 0.0, 24.0, 10.0, 0.4, 512.0]
        self.conf_min = np.array(wrapped_model['conf_min'])
        # [20.0, 1000.0, 90.0, 1500000.0, 1.0, 1.0, 96.0, 200.0, 0.8, 6144.0]
        self.conf_max = np.array(wrapped_model['conf_max'])
        self.conf_min[MF_IDX] *= MF_SCALE
        self.conf_max[MF_IDX] *= MF_SCALE
        self.conf_min[INPUTRATE_IDX] *= INPUTRATE_SCALE
        self.conf_max[INPUTRATE_IDX] *= INPUTRATE_SCALE

        # get this information from https://github.com/shenoy1/UDAO/blob/ZMQ_connection/MOO/src/main/java/concrete/iid/enter/Configuration.java
        self.conf_min_moo = np.array([1, 100, 18, 100000, 0, 0, 24, 10, 0.4, 512])
        self.conf_max_moo = np.array([10, 1000, 90, 1200000, 1, 1, 96, 200, 0.8, 6144]) # 1200000 --> 1500000
        self.conf_min_moo[MF_IDX] *= MF_SCALE
        self.conf_max_moo[MF_IDX] *= MF_SCALE
        self.conf_min_moo[INPUTRATE_IDX] *= INPUTRATE_SCALE
        self.conf_max_moo[INPUTRATE_IDX] *= INPUTRATE_SCALE

        self.model_weights_dict_sql = self.__load_weights(wrapped_model["extracted_weights_dict_without_wl"])
        self.model_weights_dict_ml = self.__load_weights(wrapped_model["extracted_weights_dict_with_wl"])

        logging.basicConfig(level=logging.INFO,
                            filename=f'streaming_models.log',
                            format=f'%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        logging.info("Models are initiated!")

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

        obj_predict = self.__get_tensor_prediction(wl_id, wl_encode, conf, obj).item()
        logging.info(f"{wl_id}, {conf_val_list} -> {obj}: {obj_predict}")

        return f"{obj_predict:.5f}"


    # minibatch at a time
    def opt_scenario1(self, zmesg, bs=16, lr=0.01, max_iter=100, weight_decay=0.1, patient=30, verbose=False):
        """
        minimize a target objective value given a workload
        :param zmesg: see batch_knob_format.md
        :return: a scalar objective prediction
        """
        wl_id, obj = self.__input_unserialize_opt_1(zmesg)
        if wl_id is None:
            return -1, None
        wl_encode = self.__get_tensor_wl_encode(wl_id)

        best_loss = np.inf
        best_obj = np.inf
        best_conf = None
        iter_num = 0

        for broadcastCompressValues in [0, 1]:
            for rddCompressValues in [0, 1]:
                if verbose:
                    print('-' * 10)
                    print(f"broadcastCompressValues: {broadcastCompressValues}, rddCompressValues: {rddCompressValues}")
                # numerical_knob_list = self.__get_tensor_numerical_knobs([0] * len(NUMERICAL_KNOB_IDS))
                numerical_knob_list = torch.rand(bs, len(NUMERICAL_KNOB_IDS),
                                                 device=self.device, dtype=self.dtype, requires_grad=True)

                optimizer = optim.Adam([numerical_knob_list], lr=lr, weight_decay=weight_decay)

                local_best_iter = 0
                local_best_loss = np.inf
                local_best_obj = np.inf
                local_best_conf = None

                iter = 0
                for iter in range(max_iter):
                    conf = self.__get_tensor_conf_cat_minibatch(numerical_knob_list, broadcastCompressValues, rddCompressValues)
                    obj_pred = self.__get_tensor_prediction_minibatch(wl_id, wl_encode, conf, obj)
                    loss, loss_id = self.__loss_soo_minibatch(wl_id, wl_encode, conf, obj, obj_pred)

                    if iter > 0 and loss.item() < local_best_loss:
                        local_best_loss = loss.item()
                        local_best_obj = obj_pred[loss_id].item()
                        local_best_conf = conf.data.numpy()[loss_id, :].copy()
                        local_best_iter = iter

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    constrained_numerical_knob_list = self.__get_tensor_numerical_constrained_knobs_minibatch(
                        wl_id, numerical_knob_list.data)
                    numerical_knob_list.data = constrained_numerical_knob_list

                    if iter > local_best_iter + patient:
                        # early stop
                        break

                    if verbose:
                        if iter % 10 == 0:
                            print(f'iteration {iter}, {obj}: {obj_pred:.2f}')
                            # print(conf)

                logging.info(f'Local best {obj}: {local_best_obj:.5f} at {local_best_iter} with confs:\n'
                             f'{local_best_conf}')
                if verbose:
                    print(f'Finished at iteration {iter}, best local {obj} found as {local_best_obj:.5f}'
                          f' \nat iteration {local_best_iter}, \nwith confs: {self.get_raw_conf(local_best_conf)}')

                iter_num += iter + 1
                if local_best_loss < best_loss:
                    best_obj = local_best_obj
                    best_loss = local_best_loss
                    best_conf = local_best_conf

        best_raw_conf = self.get_raw_conf(best_conf)
        logging.info(f"get best {obj}: {best_obj} at {best_raw_conf} with {iter_num} iterations, loss = {best_loss}")
        if verbose:
            print()
            print("*" * 10)
            print(f"get best {obj}: {best_obj} at {best_raw_conf} with {iter_num} iterations, loss = {best_loss}")

        str1 = self.__get_seralized_conf(best_raw_conf)
        str2 = f"{obj}:{best_obj:.5f}"
        return '&'.join([str1, str2])

    # a conf at a time
    def opt_scenario1_1_conf_at_a_time(self, zmesg, lr=0.01, max_iter=100, weight_decay=0.1, patient=30, verbose=False):
        """
        minimize a target objective value given a workload
        :param zmesg: see batch_knob_format.md
        :return: a scalar objective prediction
        """
        wl_id, obj = self.__input_unserialize_opt_1(zmesg)
        if wl_id is None:
            return -1, None
        wl_encode = self.__get_tensor_wl_encode(wl_id)

        best_loss = np.inf
        best_obj = np.inf
        best_conf = None
        iter_num = 0

        for broadcastCompressValues in [0, 1]:
            for rddCompressValues in [0, 1]:
                if verbose:
                    print('-' * 10)
                    print(f"broadcastCompressValues: {broadcastCompressValues}, rddCompressValues: {rddCompressValues}")
                numerical_knob_list = self.__get_tensor_numerical_knobs([0] * len(NUMERICAL_KNOB_IDS))
                optimizer = optim.Adam([numerical_knob_list], lr=lr, weight_decay=weight_decay)

                local_best_iter = 0
                local_best_loss = np.inf
                local_best_obj = np.inf
                local_best_conf = None

                iter = 0
                for iter in range(max_iter):
                    conf = self.__get_tensor_conf_cat(numerical_knob_list, broadcastCompressValues, rddCompressValues)
                    obj_pred = self.__get_tensor_prediction(wl_id, wl_encode, conf, obj)
                    loss = self.__loss_soo(wl_id, wl_encode, conf, obj, obj_pred)

                    if iter > 0 and loss.item() < local_best_loss:
                        local_best_obj = obj_pred.item()
                        local_best_loss = loss.item()
                        local_best_conf = conf.data.numpy().copy()
                        local_best_iter = iter

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    constrained_numerical_knob_list = self.__get_tensor_numerical_constrained_knobs(wl_id, numerical_knob_list.data)
                    numerical_knob_list.data = constrained_numerical_knob_list

                    if iter > local_best_iter + patient:
                        # early stop
                        break

                    if verbose:
                        if iter % 10 == 0:
                            print(f'iteration {iter}, {obj}: {obj_pred:.2f}')
                            # print(conf)

                logging.info(f'Local best {obj}: {local_best_obj:.5f} at {local_best_iter} with confs:\n'
                             f'{local_best_conf}')
                if verbose:
                    print(f'Finished at iteration {iter}, best local {obj} found as {local_best_obj:.5f}'
                          f' \nat iteration {local_best_iter}, \nwith confs: {self.get_raw_conf(local_best_conf)}')

                iter_num += iter + 1
                if local_best_loss < best_loss:
                    best_obj = local_best_obj
                    best_loss = local_best_loss
                    best_conf = local_best_conf

        best_raw_conf = self.get_raw_conf(best_conf)
        logging.info(f"get best {obj}: {best_obj} at {best_raw_conf} with {iter_num} iterations, loss = {best_loss}")
        if verbose:
            print()
            print("*" * 10)
            print(f"get best {obj}: {best_obj} at {best_raw_conf} with {iter_num} iterations, loss = {best_loss}")

        str1 = self.__get_seralized_conf(best_raw_conf)
        str2 = f"{obj}:{best_obj:.5f}"
        return '&'.join([str1, str2])

    def opt_scenario2(self, zmesg, lr=0.01, max_iter=100, weight_decay=0.1, patient=20, verbose=False,
                      benchmark=False):
        wl_id, obj, cst_dict = self.__input_unserialize_opt_2(zmesg)
        if wl_id is None:
            return None

        wl_encode = self.__get_tensor_wl_encode(wl_id)

        best_obj_dict = None
        best_loss = np.inf
        best_conf = None
        iter_num = 0

        for broadcastCompressValues in [0, 1]:
            for rddCompressValues in [0, 1]:
                if verbose:
                    print('-' * 10)
                    print(f"broadcastCompressValues: {broadcastCompressValues}, rddCompressValues: {rddCompressValues}")

                numerical_knob_list = self.__get_tensor_numerical_knobs([0] * len(NUMERICAL_KNOB_IDS))
                optimizer = optim.Adam([numerical_knob_list], lr=lr, weight_decay=weight_decay)

                local_best_iter = 0
                local_best_obj_dict = None
                local_best_loss = np.inf
                local_best_conf = None

                iter = 0
                for iter in range(max_iter):
                    conf = self.__get_tensor_conf_cat(numerical_knob_list, broadcastCompressValues, rddCompressValues)
                    obj_pred_dict = {cst_obj: self.__get_tensor_prediction(wl_id, wl_encode, conf, cst_obj) for cst_obj in cst_dict}
                    # obj_pred = self.__get_tensor_prediction(wl_id, wl_encode, conf, obj)
                    loss = self.__loss_moo(wl_id, conf, obj, obj_pred_dict, cst_dict)

                    if iter > 0 and loss.item() < local_best_loss:
                        local_best_loss = loss.item()
                        local_best_obj_dict = {k: v.item() for k, v in obj_pred_dict.items()}
                        local_best_conf = conf.data.numpy().copy()
                        local_best_iter = iter

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    constrained_numerical_knob_list = self.__get_tensor_numerical_constrained_knobs(wl_id, numerical_knob_list)
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
                          f' \nat iteration {local_best_iter}, \nwith confs: {self.get_raw_conf(local_best_conf)}')

                iter_num += iter + 1
                if self.__bound_check(pred_dict=local_best_obj_dict, cst_dict=cst_dict):
                    if local_best_loss < best_loss:
                        best_obj_dict = local_best_obj_dict
                        best_loss = local_best_loss
                        best_conf = local_best_conf

        if self.__bound_check(pred_dict=best_obj_dict, cst_dict=cst_dict):
            pass
        else:
            if verbose:
                print(NOT_FOUND_ERROR + "on NN objectives")
            logging.warning(NOT_FOUND_ERROR + "on NN objectives")
            return "NN_miss" if benchmark else "not_found"

        best_raw_conf = self.get_raw_conf(best_conf)
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
        return ret

    def opt_scenario3(self, zmesg, lr=0.01, max_iter=100, weight_decay=0.1, patient=20, verbose=False, benchmark=False,
                      processes=1):
        zmesg_list = zmesg.split('|')
        arg_list = [(z, lr, max_iter, weight_decay, patient, verbose, benchmark) for z in zmesg_list]

        with Pool(processes=processes) as pool:
            ret_list = pool.starmap(self.opt_scenario2, arg_list)
        ret = '|'.join(ret_list)

        return ret

    def forward(self, fixed_weights, wl_id, wl_encode, conf, obj):
        X = torch.cat([conf, wl_encode]).view(1, -1)
        n_layers = len(fixed_weights) // 2
        for i in range(n_layers):
            X = X.mm(fixed_weights[f'w{i}']) + fixed_weights[f'b{i}'].view(1, -1)
            X = F.relu(X)
        # x = X[0, 0]
        offset = self.cal_offset_dict[obj][self.wlId_to_alias_dict[wl_id]]
        X_cali = X + offset
        if X_cali < 0:
            return X[0, 0]
        else:
            return X_cali[0, 0]

    def forward_minibatch(self, fixed_weights, wl_id, wl_encode, conf, obj):
        n_batch = conf.shape[0]
        X = torch.cat([conf, wl_encode.repeat(n_batch, 1)], dim=1)
        n_layers = len(fixed_weights) // 2
        for i in range(n_layers):
            X = X.mm(fixed_weights[f'w{i}']) + fixed_weights[f'b{i}'].view(1, -1)
            X = F.relu(X)
        # x = X[0, 0]
        offset = self.cal_offset_dict[obj][self.wlId_to_alias_dict[wl_id]]
        X_cali = X + offset
        X_cali[X_cali < 0] = X[X_cali < 0]
        return X_cali
        # if X_cali < 0:
        #     return X[0, 0]
        # else:
        #     return X_cali[0, 0]



    def __constraintSlideWindowAndLatency_penalty(self, wl_id, bi, latency):
        if (26 <= int(wl_id) <= 31) or (56 <= int(wl_id) <= 67):
            bound = bi * 1000
        else:
            bound = torch.tensor(10000, dtype=self.dtype, device=self.device)

        if latency < bound:
            return torch.tensor(0, dtype=self.dtype, device=self.device)
        else:
            return (bound - latency) ** 2 + self.stress

    def __constraintSlideWindowAndLatency_penalty_minibatch(self, wl_id, bi, latency):
        n_batch = bi.shape[0]
        if (26 <= int(wl_id) <= 31) or (56 <= int(wl_id) <= 67):
            bound = bi * 1000
        else:
            bound = torch.tensor(10000, dtype=self.dtype, device=self.device).repeat(n_batch, 1)

        penalty = (bound - latency) ** 2 + self.stress
        penalty[latency < bound] = 0
        return penalty

        # if latency < bound:
        #     return torch.tensor(0, dtype=self.dtype, device=self.device)
        # else:
        #     return (bound - latency) ** 2 + self.stress



    def __loss_soo(self, wl_id, wl_encode, conf, obj, obj_pred):
        # for single job objective, the loss can be its own value.
        loss = obj_pred ** 2 * self.get_direction(obj)

        # return loss

        # batch_interval
        bi = self.__get_tensor_r_obj("batch_interval", conf)

        if obj == "latency":
            loss += self.__constraintSlideWindowAndLatency_penalty(wl_id, bi, obj_pred)
        else:
            latency_pred = self.__get_tensor_prediction(wl_id, wl_encode, conf, "latency")
            loss += self.__constraintSlideWindowAndLatency_penalty(wl_id, bi, latency_pred)
        return loss

    def __loss_soo_minibatch(self, wl_id, wl_encode, conf, obj, obj_pred):
        # for single job objective, the loss can be its own value.
        loss = obj_pred ** 2 * self.get_direction(obj)

        # batch_interval
        bi = self.__get_tensor_r_obj_minibatch("batch_interval", conf)

        if obj == "latency":
            loss += self.__constraintSlideWindowAndLatency_penalty_minibatch(wl_id, bi, obj_pred)
        else:
            latency_pred = self.__get_tensor_prediction_minibatch(wl_id, wl_encode, conf, "latency")
            loss += self.__constraintSlideWindowAndLatency_penalty_minibatch(wl_id, bi, latency_pred)

        return torch.min(loss), torch.argmin(loss)

    def __loss_moo(self, wl_id, conf, target_obj, obj_pred_dict, cst_dict):
        loss = torch.tensor(0, device=self.device, dtype=self.dtype)
        for cst_obj, [lower, upper] in cst_dict.items():
            cst_obj_pred = obj_pred_dict[cst_obj]
            if upper != lower:
                norm_cst_obj_pred = (cst_obj_pred - lower) / (upper - lower)
                if cst_obj == target_obj:
                    if norm_cst_obj_pred < 0 or norm_cst_obj_pred > 1:
                        add_loss = (norm_cst_obj_pred - 0.5) ** 2 + self.stress
                    else:
                        add_loss = norm_cst_obj_pred * self.get_direction(target_obj)
                else:
                    if norm_cst_obj_pred < 0 or norm_cst_obj_pred > 1:
                        add_loss = (norm_cst_obj_pred - 0.5) ** 2 + self.stress
                    else:
                        add_loss = torch.tensor(0, device=self.device, dtype=self.dtype)
            else:
                add_loss = (cst_obj_pred - upper) ** 2
            loss += add_loss
        bi = self.__get_tensor_r_obj("batch_interval", conf)
        loss += self.__constraintSlideWindowAndLatency_penalty(wl_id, bi, obj_pred_dict['latency'])
        return loss

    #############################
    ##### unserialize zmesg #####
    #############################

    def __input_unserialize_predict(self, zmesg):
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
        E.g., zmesg = "JobID:14;Objective:latency;Constraint:throughput:10000:20000;Constraint:latency:0:2000"
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

    def __get_tensor_prediction(self, wl_id, wl_encode, conf, obj):
        if obj == "cost":
            return self.__get_tensor_r_obj(obj, conf)
        if wl_id in self.wlId_id_ml:
            model_weights_dict = self.model_weights_dict_ml
        elif wl_id in self.wlId_id_sql:
            model_weights_dict = self.model_weights_dict_sql
        else:
            print(f'Job {wl_id} does not found')
            return torch.tensor(-1)
        fixed_weights = model_weights_dict[obj]
        obj_predict = self.forward(fixed_weights, wl_id, wl_encode, conf, obj)
        return obj_predict

    def __get_tensor_prediction_minibatch(self, wl_id, wl_encode, conf, obj):
        if obj == "cost":
            return self.__get_tensor_r_obj_minibatch(obj, conf)
        if wl_id in self.wlId_id_ml:
            model_weights_dict = self.model_weights_dict_ml
        elif wl_id in self.wlId_id_sql:
            model_weights_dict = self.model_weights_dict_sql
        else:
            print(f'Job {wl_id} does not found')
            return torch.tensor(-1)
        fixed_weights = model_weights_dict[obj]
        obj_predict = self.forward_minibatch(fixed_weights, wl_id, wl_encode, conf, obj)
        return obj_predict

    def __get_tensor_wl_encode(self, wl_id):
        return torch.tensor(self.__get_wl_encode(wl_id), device=self.device, dtype=self.dtype)

    def __get_tensor_conf_for_predict(self, conf_val_list):
        return torch.tensor(conf_val_list, device=self.device, dtype=self.dtype)

    def __get_tensor_numerical_knobs(self, numerical_knob_list):
        return torch.tensor(numerical_knob_list, device=self.device, dtype=self.dtype, requires_grad=True)

    def __get_tensor_conf_cat(self, numerical_knob_list, broadcastCompressValues, rddCompressValues):
        t_bb = torch.tensor([broadcastCompressValues, rddCompressValues], device=self.device, dtype=self.dtype)
        # to a kxk_streaming knob order
        conf = torch.cat([numerical_knob_list[:4], t_bb, numerical_knob_list[4:]])
        return conf

    def __get_tensor_conf_cat_minibatch(self, numerical_knob_list, broadcastCompressValues, rddCompressValues):
        n_batch = numerical_knob_list.shape[0]
        t_bb = torch.tensor([broadcastCompressValues, rddCompressValues], device=self.device, dtype=self.dtype)\
            .repeat(n_batch, 1)
        # to a kxk_streaming knob order
        conf = torch.cat([numerical_knob_list[:, :4], t_bb, numerical_knob_list[:, 4:]], dim=1)
        return conf

    def __get_tensor_numerical_constrained_knobs(self, wl_id, numerical_knob_list):
        # with NUMERICAL_KNOB_IDS = [0, 1, 2, 3, 6, 7, 8, 9]
        unbounded_norm_np = numerical_knob_list.data.numpy()
        unbounded_raw_np = self.get_raw_conf(unbounded_norm_np, normalized_ids=NUMERICAL_KNOB_IDS)
        bounded_raw_np = np.array([self.__get_bounded(k, wl_id, NUMERICAL_KNOB_IDS[idx],
                                                      self.conf_min_moo[NUMERICAL_KNOB_IDS[idx]],
                                                      self.conf_max_moo[NUMERICAL_KNOB_IDS[idx]]) for idx, k in
                                   enumerate(unbounded_raw_np)])
        normalized_np = self.__get_normalized_conf(bounded_raw_np, normalized_ids=NUMERICAL_KNOB_IDS)
        return torch.tensor(normalized_np, device=self.device, dtype=self.dtype)

    def __get_tensor_numerical_constrained_knobs_minibatch(self, wl_id, numerical_knob_list):
        # with NUMERICAL_KNOB_IDS = [0, 1, 2, 3, 6, 7, 8, 9]
        numerical_knob_list[numerical_knob_list > 1] = 1
        numerical_knob_list[numerical_knob_list < 0] = 0
        raw_np = self.get_raw_conf(numerical_knob_list.numpy(), normalized_ids=NUMERICAL_KNOB_IDS)
        normalized_np = self.__get_normalized_conf(raw_np, normalized_ids=NUMERICAL_KNOB_IDS)
        return torch.tensor(normalized_np, device=self.device, dtype=self.dtype)

    def __get_tensor_bounds_list(self, lower, upper):
        t_lower = torch.tensor(lower, device=self.device, dtype=self.dtype)
        t_upper = torch.tensor(upper, device=self.device, dtype=self.dtype)
        return [t_lower, t_upper]

    def __get_tensor_r_obj(self, obj, conf):
        # conf is a tensor list
        if obj == "cost":
            # cost = C1 * batchInterval + C2 * parallelism
            # C1 = 21.5, C2 = 30
            conf_0_max, conf_2_max = self.conf_max[0], self.conf_max[2]
            conf_0_min, conf_2_min = self.conf_min[0], self.conf_min[2]
            conf_0_norm, conf_2_norm = conf[0], conf[2]
            conf_0 = conf_0_norm * (conf_0_max - conf_0_min) + conf_0_min
            conf_2 = conf_2_norm * (conf_2_max - conf_2_min) + conf_2_min
            cost = C1 * conf_0 + C2 * conf_2
            return cost
        elif obj == "batch_interval":
            conf_max, conf_min = self.conf_max[0], self.conf_min[0]
            conf_norm = conf[0]
            conf = conf_norm * (conf_max - conf_min) + conf_min
            return conf
        else:
            raise Exception(f"{obj} cannot be found")

    def __get_tensor_r_obj_minibatch(self, obj, conf):
        # conf is a tensor list
        if obj == "cost":
            # cost = C1 * batchInterval + C2 * parallelism
            # C1 = 21.5, C2 = 30
            conf_0_max, conf_2_max = self.conf_max[0], self.conf_max[2]
            conf_0_min, conf_2_min = self.conf_min[0], self.conf_min[2]
            conf_0_norm, conf_2_norm = conf[:, 0], conf[:, 2]
            conf_0 = conf_0_norm * (conf_0_max - conf_0_min) + conf_0_min
            conf_2 = conf_2_norm * (conf_2_max - conf_2_min) + conf_2_min
            cost = C1 * conf_0 + C2 * conf_2
            return cost.view(-1, 1)
        elif obj == "batch_interval":
            conf_max, conf_min = self.conf_max[0], self.conf_min[0]
            conf_norm = conf[:, 0]
            conf = conf_norm * (conf_max - conf_min) + conf_min
            return conf.view(-1, 1)
        else:
            raise Exception(f"{obj} cannot be found")

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

    @staticmethod
    def get_direction(obj):
        if obj == "throughput":
            return -1
        else:
            return 1

    def __get_seralized_conf(self, conf):
        conf[MF_IDX] /= MF_SCALE
        conf[INPUTRATE_IDX] /= INPUTRATE_SCALE

        conf_s = []
        for i in range(len(self.knob_list)):
            if i != 8:
                conf_s.append(f'{self.knob_list[i]}:{int(conf[i])}')
            else:
                conf_s.append(f'{self.knob_list[i]}:{conf[i]:.2f}')
        return ';'.join(conf_s)

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

    def get_raw_conf(self, normalized_conf, normalized_ids=None):
        """
        :param normalized_conf: numpy.array[float] [0,1]
        :return: numpy.array[int]
        """
        conf_max = self.conf_max if normalized_ids is None else self.conf_max[normalized_ids]
        conf_min = self.conf_min if normalized_ids is None else self.conf_min[normalized_ids]
        conf = normalized_conf * (conf_max - conf_min) + conf_min
        get_raw_conf = conf.round()
        return get_raw_conf

    def __get_r_obj(self, obj, conf):
        """
        :param obj:
        :param conf: np.array, raw value
        :return:
        """
        if obj == "cost":
            conf_0, conf_2 = conf[0], conf[2]
            return C1 * conf_0 + C2 * conf_2

    def __get_opt2_ret(self, conf, pred_dict):
        str1 = self.__get_seralized_conf(conf)
        str2 = ';'.join([f'{k}:{v:.5f}' for k, v in pred_dict.items()])
        return f'{str1}&{str2}'

    @staticmethod
    def __get_bounded(k, wl_id, knob_idx, lower=0.0, upper=1.0):
        if k > upper:
            return upper
        if k < lower:
            return lower
        if knob_idx == 0:
            # constrain2 included
            if (26 <= int(wl_id) <= 31) or (56 <= int(wl_id) <= 67):
                return k
            else:
                if k in [1,2,5,10]:
                    return k
                elif k >= 8:
                    return 10
                elif k == 3:
                    return 2
                elif k in [4,6,7]:
                    return 5
        return k

    def __load_weights(self, extracted_weights_dict):
        obj_weights_dict = {}
        for obj, obj_weights in extracted_weights_dict.items():
            obj_weights_dict[obj] = self.__unwrap_weights(obj_weights)
        # adding weights for simulated_cost, same as latency
        # obj_weights_dict['simulated_cost'] = self.__unwrap_weights(extracted_weights_dict['latency'])
        # obj_weights_dict['simulated_cost2'] = None
        return obj_weights_dict

    @staticmethod
    def __load_obj(name):
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

    def __conf_check(self, conf):
        for i, knob_name in enumerate(self.knob_list):
            if knob_name == "inputRate":
                continue
            else:
                if conf[i] < 0 or conf[i] > 1:
                    logging.error(
                        f'ERROR: knob {i} {knob_name}: {conf[i]} out of range, check conf_min and conf_max {conf}')
                    print(f'ERROR: {knob_name}: {conf[i]} out of range, check conf_min and conf_max {conf}')
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

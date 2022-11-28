# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: GPR Objective functions in ICDE paper
#
# Created at 21/09/2022

import utils.optimization.solver_utils as solver_ut
from tests.optimization.solver.gpr import GPRPT
from utils.optimization.configs_parser import ConfigsParser


import torch as th
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

BATCH_OFF_TEST_JOBS = "1-7,2-2,3-2,4-7,5-1,6-2,7-2,8-5,9-3,10-0,11-2,12-3,13-4,14-0,15-4,16-3,17-5,18-1,19-7,20-4,21-1,22-7,23-4,24-3,25-7,26-8,27-7,28-0,29-1,30-0".split(
    ',')
BATCH_ON_TEST_JOBS = "1-1,2-3,3-8,4-2,5-5,6-3,7-8,8-0,9-7,10-1,11-0,12-5,13-7,14-2,15-1,16-2,17-0,18-7,19-6,20-7,21-3,22-8,23-5,24-8,25-4,26-2,27-3,28-2,29-2,30-7".split(
    ',')

COST_LIST = [
    'cost-amazon'  # latency/1000/60/60 (hour) * #cores * cost_rate1 + ops/1000/1000 * cost_rate3
]
R_OBJ_LIST = [
    'cores'
]
GP_OBJ_LIST = [
    'latency',
    'ops'
]
COST_RATIO_C_X_L, COST_RATIO_IO = 0.06, 0.2

DEFAULT_DEVICE = th.device("cpu")
DEFAULT_DTYPE = th.float32

##########################
## functions used for model initialization ##
##########################

def _get_conf_range_for_wl(wl_id):
    conf_max = scaler_map[wl_id].data_max_
    conf_min = scaler_map[wl_id].data_min_
    return conf_max, conf_min

def _get_gp_models(data, proxy_jobs, wl_list, ridge):
    obj_lbl = data['metrics_details']['objective_label']
    obj_idx = data['metrics_details']['objective_idx']
    ##### obj_lbl and obj_idx to be matched with pred_objective as below ####

    model_map = {}
    scaler_map = {}
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

        # Scale to (0, 1)
        X_scaler = MinMaxScaler()
        X_scaled = X_scaler.fit_transform(X_matrix)

        y_dict['latency'] *= 1000
        y_scale_obj_dict = {obj: _get_y_scale_in_SS(y_dict[obj].reshape(-1, 1)) for obj in GP_OBJ_LIST}

        model_map[wl] = model.fit(X_train=X_scaled, y_dict=y_dict, ridge=ridge)
        scaler_map[wl] = X_scaler
        scale_y_map[wl] = y_scale_obj_dict
    return model_map, scaler_map, scale_y_map

def _get_y_scale_in_SS(y):
    y_scaler_ = StandardScaler()
    y_scaler_.fit(y)
    return y_scaler_.scale_[0]

##########################
## model initialization ##
##########################

model_params = ConfigsParser().parse_details(option="model")
try:
    gpr_weights_path = model_params['gpr_weights_path']
    gpr_data_ = solver_ut.load_pkl(gpr_weights_path)
    default_ridge = model_params['default_ridge']
    accurate = model_params['accurate']
    alpha = model_params['alpha']
except:
    raise Exception('model_params is not fully correct')
model = GPRPT(gp_obj_list=GP_OBJ_LIST)
wl_list_ = BATCH_OFF_TEST_JOBS + BATCH_ON_TEST_JOBS
data_ = gpr_data_['data']
proxy_jobs_ = gpr_data_['proxy_jobs']
model_map, scaler_map, scale_y_map = _get_gp_models(data_, proxy_jobs_, wl_list_, default_ridge)
if accurate:
    conf_constraints = None
else:
    conf_constraints = {"vars_min": np.array([64, 8, 2, 6, 24, 35, 0, 0.5, 5000, 64, 10, 36]),
                        "vars_max": np.array([144, 24, 4, 8, 192, 145, 1, 0.75, 20000, 256, 100, 144])}

#########################
## objective functions ##
#########################

def obj_func1(wl_id, vars):
    if not th.is_tensor(vars):
        vars = solver_ut._get_tensor(vars)
    obj = "latency"
    vars_copy = vars.clone()

    # # do make sure the var_ranges are the same as in the config.json file
    # vars_max, vars_min = _get_vars_range_for_wl(wl_id)
    # # conf_norm = np.array([(vars_copy[i].data.numpy() - vars_min) / (vars_max - vars_min) for i in range(vars_copy.shape[0])])
    # conf_norm = get_normalized_vars(vars_copy.data.numpy(), vars_max, vars_min)
    # vars_copy.data = solver_ut._get_tensor(conf_norm)

    X_train, y_dict, K_inv = model_map[wl_id]
    y_train = y_dict[obj]
    obj_pred = model.objective(vars_copy, X_train, y_train, K_inv).view(-1, 1)

    assert (obj_pred.ndimension() == 2)
    return obj_pred

def obj_func2(wl_id, vars):
    # obj = "cores"
    if not th.is_tensor(vars):
        vars = solver_ut._get_tensor(vars)

    vars_copy = vars.clone()
    conf_max, conf_min = _get_conf_range_for_wl(wl_id)

    # conf_norm = np.array(
    #     [(vars_copy[i].data.numpy() - vars_min) / (vars_max - vars_min) for i in range(vars_copy.shape[0])])
    # vars_copy.data = solver_ut._get_tensor(conf_norm)

    k2_max, k3_max = conf_max[1:3]
    k2_min, k3_min = conf_min[1:3]
    if vars_copy.ndimension() == 1:
        k2_norm, k3_norm = [k.view(-1, 1) for k in vars_copy[1:3]]
    else:
        k2_norm, k3_norm = vars_copy[:, 1], vars_copy[:, 2]
    k2 = k2_norm * (k2_max - k2_min) + k2_min
    k3 = k3_norm * (k3_max - k3_min) + k3_min
    n_exec = th.min(th.floor(th.tensor(58, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE) / k3), k2)
    cores = n_exec * k3
    obj_pred = cores.reshape(-1, 1)

    assert (obj_pred.ndimension() == 2)
    return obj_pred

def obj_func3(wl_id, vars):
    if not th.is_tensor(vars):
        vars = solver_ut._get_tensor(vars)
    obj = "ops"
    vars_copy = vars.clone()
    # conf_max, conf_min = _get_conf_range_for_wl(wl_id)
    # conf_norm = (vars_copy[0].data.numpy() - conf_min) / (conf_max - conf_min)
    # vars_copy.data = solver_ut._get_tensor(conf_norm).reshape(1,-1)
    # vars = vars.reshape(1,-1)

    X_train, y_dict, K_inv = model_map[wl_id]
    y_train = y_dict[obj]
    obj_pred = model.objective(vars_copy, X_train, y_train, K_inv).view(-1, 1)

    assert (obj_pred.ndimension() == 2)
    return obj_pred

def obj_func4(wl_id, vars):
    obj = "cost-amazon"
    if not th.is_tensor(vars):
        vars = solver_ut._get_tensor(vars)
    vars_copy = vars.clone()
    lat_pred = obj_func1(wl_id, vars_copy)
    cor_pred = obj_func2(wl_id, vars_copy)
    ops_pred = obj_func3(wl_id, vars_copy)
    # obj_pred = (COST_RATIO_C_X_L * lat_pred / 1000 / 3600 * cor_pred
    #             + COST_RATIO_IO * ops_pred / 1000 / 1000) * 1000  # 0.001 dollars
    obj_pred = (COST_RATIO_C_X_L * lat_pred / 1000 * cor_pred
                + COST_RATIO_IO * ops_pred / 1000 / 1000) * 1000  # 0.001 dollars

    assert (obj_pred.ndimension() == 2)
    return obj_pred


def _get_tensor_obj_std(wl_id, conf, obj):
    """return shape (-1, 1)"""
    if obj in R_OBJ_LIST:
        std = solver_ut._get_tensor([[0.]])
    elif obj in GP_OBJ_LIST:
        X_train, _, K_inv = model_map[wl_id]
        y_scale = scale_y_map[wl_id][obj]  # float
        if conf.ndimension() < 2:
            conf = conf.reshape(1, -1)
        std = model.objective_std(X_test=conf, X_train=X_train, K_inv=K_inv, y_scale=y_scale).reshape(-1, 1)
    else:
        raise Exception(f'does not have support for {obj}')
    assert(std.ndimension() == 2)
    return std
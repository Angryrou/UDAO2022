import random
import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as skl_shuffle
import copy
import csv
import time


SEED = 692
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DTYPE = torch.float32

def save_obj(obj, dest_path):
    with open(dest_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(dest_path):
    with open(dest_path, 'rb') as f:
        return pickle.load(f)

def get_wl_dict(configs):
    if "wl_dict" in configs:
        print('found wl_dict in configs.')
        return configs['wl_dict']
    print('not found wl_dict in configs, start generating from the path ...')
    wl_dict = {
        "offline": [],
        "online": []
    }
    off_wl = [x[:-4] for x in os.listdir(f"{configs['data_path']['off_wl']}") if x[-4:] == ".csv"]
    wl_dict["offline"] = sorted(off_wl)
    on_wl = [x[:-4] for x in os.listdir(f"{configs['data_path']['on_wl']}") if x[-4:] == ".csv"]
    wl_dict["online"] = sorted(on_wl)
    print('wl_dict generated.')
    return wl_dict

def split_wl_train_test(wl_dict, temp_ids):
    random.seed(SEED)
    on_wls = wl_dict['online']
    off_wls = wl_dict["offline"]
    # online_wl --> 0.6 for train, 0.2 for val, 0.2 for test
    on_wl_tr = []
    wl_val = []
    wl_te = []
    random.shuffle(on_wls)

    for tid in temp_ids:
        t_wls = [wl for wl in on_wls if wl.split('-')[0] == tid]
        t_wls_size = len(t_wls)
        t_val_size = round(t_wls_size * 0.2)
        t_te_size = round(t_wls_size * 0.2)
        wl_val += t_wls[:t_val_size]
        wl_te += t_wls[t_val_size: t_val_size + t_te_size]
        on_wl_tr += t_wls[t_val_size + t_te_size:]

    return [off_wls, on_wl_tr, wl_val, wl_te]


def load_data(configs, dest_path="cache/692t_data.pkl"):

    if os.path.exists(dest_path):
        print(f'found cached data at {dest_path}')
        data = load_obj(dest_path)
        return data
    else:
        print(f'not found cached data at {dest_path}, start generating data ...')

    configs_wl = configs['workloads']
    configs_csv = configs['csv']
    wl_dict = get_wl_dict(configs=configs_wl)
    off_wls, on_wl_tr, wl_val, wl_te = split_wl_train_test(wl_dict=wl_dict, temp_ids=configs_wl['temp_list'])
    wls = off_wls + on_wl_tr + wl_val + wl_te
    temp_list = configs_wl['temp_list']
    max_seen_size = configs_wl['max_seen_size']
    temp_dict = {t: [wl for wl in wls if wl.split('-')[0] == t] for t in temp_list}

    cols_to_keep = configs_csv['COLS_TO_KEEP']
    cols_knobs = configs_csv['COLS_KNOBS']
    cols_prep_toint = configs_csv['COLS_PREP_TOINT']
    cols_targets = configs_csv['COLS_TARGETS']
    cols_to_drop = configs_csv['COLS_TO_DROP']
    cols_internal_metrics = sorted(list(set(cols_to_keep) - set(cols_knobs + cols_to_drop + cols_targets)))

    wl2aid, aid2wl = {}, {}
    data_tr, data_val, data_te = {}, {}, {}
    off_aids_set, on_aids_set = set(), set()

    # speed up for search
    wl_off_set = set(wl_dict['offline'])
    wl_tr_set = set(off_wls + on_wl_tr)
    wl_val_set = set(wl_val)

    random.seed(SEED)
    for aid, wl in enumerate(off_wls + on_wl_tr + wl_val + wl_te):
        wl2aid[wl] = aid
        aid2wl[aid] = wl
        if wl in wl_off_set:
            off_aids_set.add(aid)
            wl_path = f"{configs_wl['data_path']['off_wl']}/{wl}.csv"
        else:
            on_aids_set.add(aid)
            wl_path = f"{configs_wl['data_path']['on_wl']}/{wl}.csv"

        wl_df = pd.read_csv(wl_path)
        for col in cols_prep_toint:
            wl_df[col] = list(map(lambda val: int(val), wl_df[col]))
        wl_C = wl_df[cols_knobs].values
        wl_l = wl_df[cols_targets].values
        wl_O = wl_df[cols_internal_metrics].values

        if wl in wl_tr_set:
            data_tr[wl] = {"confs": wl_C, "objs": wl_l, "obses": wl_O}
        else:
            data_val[wl] = {}
            # seen for blackbox testing, unseen for testing
            wl_C_unseen, wl_C_seen, wl_l_unseen, wl_l_seen, \
            wl_O_unseen, wl_O_seen = train_test_split(wl_C, wl_l, wl_O, random_state=SEED+aid, test_size=max_seen_size)
            wl_dict = {
                "seen": {"confs": wl_C_seen, "objs": wl_l_seen, "obses": wl_O_seen},
                "unseen": {"confs": wl_C_unseen, "objs": wl_l_unseen, "obses": wl_O_unseen}
            }
            if wl in wl_val_set:
                data_val[wl] = wl_dict
            else:
                data_te[wl] = wl_dict

    data = {
        "wl_dict": {
            "offline": off_wls,
            "online": configs_wl['wl_dict']['online'],
            "tr": off_wls + on_wl_tr,
            "val": wl_val,
            "te": wl_te
        },
        "meta": {
            "internal_metric_names": cols_internal_metrics,
            "knob_names": cols_knobs,
            "objective_names": cols_targets,
            "wl2aid": wl2aid,
            "aid2wl": aid2wl,
            "off_aids_set": off_aids_set,
            "on_aids_set": on_aids_set,
            "temp_list": temp_list,
            "temp_dict": temp_dict
        },
        "tr": data_tr,
        # job -> [C, l, O]
        # {},
        "val": data_val,
        # { # job -> {'seen': [], 'unseen': []}
        #     'seen': [],
        #     'unseen': []
        # },
        "te": data_te
        # { # job -> {'seen': [], 'unseen': []}
        #     'seen': [],
        #     'unseen': []
        # }
    }

    save_obj(obj=data, dest_path=dest_path)
    print(f'data is generated and saved at {dest_path}')
    return data


def split_wl_train_val_test(wl_dict, wl_split, temp_ids):
    random.seed(SEED)
    on_wls = wl_dict['online']
    off_wls = wl_dict["offline"]
    # online_wl --> 0.6 for train, 0.2 for val, 0.2 for test
    on_wl_tr = []
    wl_val = []
    wl_te = []
    random.shuffle(on_wls)

    split_ratio = wl_split["online"]
    t_wl_size = 0.0
    for i in split_ratio:
        t_wl_size += i
    assert t_wl_size == 1.0 or len(split_ratio) != 3, "Check split ratio total or length"

    for tid in temp_ids:
        t_wls = [wl for wl in on_wls if wl.split('-')[0] == tid]
        t_wls_size = len(t_wls)
        t_val_size = round(t_wls_size * split_ratio[1])
        t_te_size = round(t_wls_size * split_ratio[2])
        wl_val += t_wls[:t_val_size]
        wl_te += t_wls[t_val_size: t_val_size + t_te_size]
        on_wl_tr += t_wls[t_val_size + t_te_size:]

    return [off_wls, on_wl_tr, wl_val, wl_te]


def load_data_icde(configs, dest_path="cache/icde_data.pkl"):

    if os.path.exists(dest_path):
        print(f'found cached data at {dest_path}')
        data = load_obj(dest_path)
        return data
    else:
        print(f'not found cached data at {dest_path}, start generating data ...')

    configs_wl = configs['workloads']
    configs_csv = configs['csv']
    wl_dict = get_wl_dict(configs=configs_wl)
    wl_split = configs_wl['wl_split']
    # len(on_wl_val) = round((40 - 2) * 0.2) * 28 + round((20-1) * 0.2) * 1 = 8 * 28 + 4 * 2 = 232
    off_wl, on_wl_tr, on_wl_val, on_wl_te = split_wl_train_val_test(wl_dict=wl_dict, wl_split=wl_split, temp_ids=configs_wl['temp_list'])

    wls = off_wl + on_wl_tr + on_wl_val + on_wl_te
    temp_wl_list = configs_wl['temp_list']
    max_unseen_size_on_val_te = configs_wl['max_unseen_size_on_val_te']
    max_unseen_size_off_val_te = configs_wl['max_unseen_size_off_val_te']
    temp_wl_dict = {t: [wl for wl in wls if wl.split('-')[0] == t] for t in temp_wl_list}

    cols_to_keep = configs_csv['COLS_TO_KEEP']
    cols_knobs = configs_csv['COLS_KNOBS']
    cols_prep_toint = configs_csv['COLS_PREP_TOINT']
    cols_targets = configs_csv['COLS_TARGETS']
    cols_to_drop = configs_csv['COLS_TO_DROP']
    cols_internal_metrics = sorted(list(set(cols_to_keep) - set(cols_knobs + cols_to_drop + cols_targets)))

    wl2aid, aid2wl = {}, {}
    data_tr, data_val, data_te = {}, {}, {}
    off_aids_set, on_aids_set = set(), set()

    # speed up for search
    wl_off_set = set(wl_dict['offline'])
    wl_tr_set = set(off_wl + on_wl_tr)
    wl_val_set = set(off_wl + on_wl_val)
    wl_te_set = set(off_wl + on_wl_te)

    random.seed(SEED)
    for aid, wl in enumerate(off_wl + on_wl_tr + on_wl_val + on_wl_te):
        wl2aid[wl] = aid
        aid2wl[aid] = wl
        if wl in wl_off_set:
            off_aids_set.add(aid)
            wl_path = f"{configs_wl['data_path']['off_wl']}/{wl}.csv"
            offline = True
        else:
            on_aids_set.add(aid)
            wl_path = f"{configs_wl['data_path']['on_wl']}/{wl}.csv"
            offline = False

        wl_df = pd.read_csv(wl_path)
        for col in cols_prep_toint:
            wl_df[col] = list(map(lambda val: int(val), wl_df[col]))
        wl_C = wl_df[cols_knobs].values
        wl_l = wl_df[cols_targets].values
        wl_O = wl_df[cols_internal_metrics].values

        if offline:
            wl_C_tr, wl_C_unseen, wl_l_tr, wl_l_unseen, \
            wl_O_tr, wl_O_unseen = train_test_split(wl_C, wl_l, wl_O, random_state=SEED + aid,
                                                      test_size=(max_unseen_size_off_val_te*2))
            wl_C_unseen_val, wl_C_unseen_te, wl_l_unseen_val, wl_l_unseen_te, \
            wl_O_unseen_val, wl_O_unseen_te = train_test_split(wl_C_unseen, wl_l_unseen, wl_O_unseen, random_state=SEED + aid,
                                                    test_size=max_unseen_size_off_val_te)

            data_tr[wl] = {"confs": wl_C_tr, "objs": wl_l_tr, "obses": wl_O_tr}
            data_val[wl] = {
                "seen": {"confs": [], "objs": [], "obses": []},
                "unseen": {"confs": wl_C_unseen_val, "objs": wl_l_unseen_val, "obses": wl_O_unseen_val}
            }
            data_te[wl] = {
                "seen": {"confs": [], "objs": [], "obses": []},
                "unseen": {"confs": wl_C_unseen_te, "objs": wl_l_unseen_te, "obses": wl_O_unseen_te}
            }

        else:
            if wl in wl_tr_set:
                data_tr[wl] = {"confs": wl_C, "objs": wl_l, "obses": wl_O}
            elif wl in wl_val_set:
                # seen for blackbox testing, unseen for testing
                wl_C_seen_val, wl_C_unseen_val, wl_l_seen_val, wl_l_unseen_val, \
                wl_O_seen_val, wl_O_unseen_val = train_test_split(wl_C, wl_l, wl_O, random_state=SEED + aid,
                                                                   test_size=max_unseen_size_on_val_te)
                data_val[wl] = {
                    "seen": {"confs": wl_C_seen_val, "objs": wl_l_seen_val, "obses": wl_O_seen_val},
                    "unseen": {"confs": wl_C_unseen_val, "objs": wl_l_unseen_val, "obses": wl_O_unseen_val}
                }
            elif wl in wl_te_set:
                wl_C_seen_te, wl_C_unseen_te, wl_l_seen_te, wl_l_unseen_te, \
                wl_O_seen_te, wl_O_unseen_te = train_test_split(wl_C, wl_l, wl_O, random_state=SEED + aid,
                                                                  test_size=max_unseen_size_on_val_te)
                data_te[wl] = {
                    "seen": {"confs": wl_C_seen_te, "objs": wl_l_seen_te, "obses": wl_O_seen_te},
                    "unseen": {"confs": wl_C_unseen_te, "objs": wl_l_unseen_te, "obses": wl_O_unseen_te}
                }

    #load the 30 recommended offline workloads and add it to the test data
    is_reco_te = wl_split["test_cases"]
    reco_te_data = wl_split["test_data"]
    if is_reco_te:
        wl_df_reco_te = pd.read_csv(reco_te_data)
        for col in cols_prep_toint:
            wl_df_reco_te[col] = list(map(lambda val: int(val), wl_df_reco_te[col]))
        wl_C_reco_te = wl_df_reco_te[cols_knobs].values
        wl_l_reco_te = wl_df_reco_te[cols_targets].values
        #wl_O_reco_te = wl_df_reco_te[cols_internal_metrics].values#np.zeros((wl_l_reco_te[0],571))
        #wl_O_reco_te = np.zeros((wl_l_reco_te[0], 571))
        wl_reco_te = wl_df_reco_te["wl_id"].values
        for i, wl in enumerate(wl_reco_te):
            data_te_confs = data_te[wl]['unseen']['confs']
            data_te_objs = data_te[wl]['unseen']['objs']
            data_te_obses = data_te[wl]['unseen']['obses']
            data_te[wl] = {
                "seen": {"confs": [], "objs": [], "obses": []},
                "unseen": {"confs": np.vstack((data_te_confs, wl_C_reco_te[i])),
                           "objs": np.vstack((data_te_objs, wl_l_reco_te[i])),
                           "obses": np.vstack((data_te_obses, np.zeros(data_te_obses.shape[1])))}
            }
    data = {
        "wl_dict": {
            "offline": off_wl,
            "online": wl_dict['online'],
            "tr": off_wl + on_wl_tr,
            "val": off_wl + on_wl_val,
            "te": off_wl + on_wl_te
        },
        "meta": {
            "internal_metric_names": cols_internal_metrics,
            "knob_names": cols_knobs,
            "objective_names": cols_targets,
            "wl2aid": wl2aid,
            "aid2wl": aid2wl,
            "off_aids_set": off_aids_set,
            "on_aids_set": on_aids_set,
            "temp_list": temp_wl_list,
            "temp_dict": temp_wl_dict
        },
        "tr": data_tr,
        # {job: {"confs": [], "objs": [], "obses": []}},
        "val": data_val,
        # { # job_off: {'seen': [empty], 'unseen': {"confs": [], "objs": [], "obses": []}}
        # },
        # { # job_on: {'seen': {"confs": [], "objs": [], "obses": []}, 'unseen': {"confs": [], "objs": [], "obses": []}}
        # },
        "te": data_te
        # { # job_off: {'seen': [empty], 'unseen': {"confs": [], "objs": [], "obses": []}}
        # },
        # { # job_on: {'seen': {"confs": [], "objs": [], "obses": []}, 'unseen': {"confs": [], "objs": [], "obses": []}}
        # },
    }

    save_obj(obj=data, dest_path=dest_path)
    print(f'data is generated and saved at {dest_path}')
    return data


def get_global_tr(data_dict, wl_list, p_name):
    # for data['tr'] reshape()
    try:
        np.concatenate([data_dict[wl][p_name] for wl in wl_list])
    except:
        print('err')
    p_global = np.concatenate([data_dict[wl][p_name] for wl in wl_list])
    return p_global

def get_global_te_obs(data_dict, wl_list, wl_list_off, p_name, obs="seen", obs_num=None):
    if obs == "seen":
        if obs_num is None:
            p_global_obs = np.concatenate([data_dict[wl][obs][p_name] for wl in wl_list if wl not in wl_list_off])
        else:
            p_global_obs = np.concatenate([data_dict[wl][obs][p_name][:obs_num] for wl in wl_list if wl not in wl_list_off])
    else:
        p_global_obs = np.concatenate([data_dict[wl][obs][p_name] for wl in wl_list])
    return p_global_obs

def get_global_te(data_dict, wl_list, wl_list_off, p_name):
    # for data['val']/data['te']
    p_seen_global = get_global_te_obs(data_dict, wl_list, wl_list_off, p_name, obs="seen")
    p_unseen_global = get_global_te_obs(data_dict, wl_list, wl_list_off, p_name, obs="unseen")
    return np.concatenate([p_seen_global, p_unseen_global])

def get_matched_rate(nnb_dict_verbose, obs_num):
    matched_case = 0
    for wl, wl_nnbs in nnb_dict_verbose.items():
        wl_nnb = wl_nnbs[f'seen_{obs_num}']
        matched_case += wl.split('-')[0] == wl_nnb.split('-')[0]
    return matched_case / len(nnb_dict_verbose)

def get_proxy_jobs(nnb_dict_verbose, obs_num):
    proxy_jobs = {
        wl: wl_nnbs[f'seen_{obs_num}'] for wl, wl_nnbs in nnb_dict_verbose.items()
    }
    return proxy_jobs

def get_signature_conf(conf):
    return '&'.join([f"{k:.2f}" for k in conf])

def compute_MAPE(l_pred, l_real):
    return abs(l_pred - l_real) / l_real

########
# UDAO
########

def get_ae_params(ae_params):
    lr = ae_params['lr']
    bs = ae_params['bs']
    epochs = ae_params['epochs']
    weight_str = ae_params['weight_str']
    W_dim = ae_params['W_dim']
    return [weight_str, lr, bs, epochs, W_dim]

def get_nnr_params(nnr_params):
    lr = nnr_params['lr']
    bs = nnr_params['bs']
    epochs = nnr_params['epochs']
    cap_str = nnr_params['cap_str']
    return [lr, bs, epochs, cap_str]

def get_sign(params_list):
    return ','.join([f"{p}" for p in params_list])

def get_ae_sign(params_all):
    ae_params = params_all['ae']
    ae_params_list = get_ae_params(ae_params)
    ae_sign = get_sign(ae_params_list)
    return ae_sign

def prep_ae_tr(data_dict, wl_list, wl2aid):
    C_tr, l_tr, O_tr = [get_global_tr(data_dict, wl_list, p_name) for p_name in ['confs', 'objs', 'obses']]
    A_tr, temp_tr = np.array([]), np.array([])
    for wl in wl_list:
        N = data_dict[wl]['confs'].shape[0]
        A_tr = np.concatenate([A_tr, [wl2aid[wl]] * N])
        temp_tr = np.concatenate([temp_tr, [int(wl.split('-')[0])] * N])
    return [A_tr, temp_tr, C_tr, O_tr, l_tr.squeeze()]

def prep_ae_te(data_dict, wl_list, wl_list_off, wl2aid):
    C_, l_, O_ = [get_global_te(data_dict, wl_list, wl_list_off, p_name) for p_name in ['confs', 'objs', 'obses']]
    A_, temp_ = np.array([]), np.array([])
    for wl in wl_list:
        if wl in wl_list_off:
            continue
        N = data_dict[wl]['seen']['confs'].shape[0]
        A_ = np.concatenate([A_, [wl2aid[wl]] * N])
        temp_ = np.concatenate([temp_, [int(wl.split('-')[0])] * N])
    for wl in wl_list:
        N = data_dict[wl]['unseen']['confs'].shape[0]
        A_ = np.concatenate([A_, [wl2aid[wl]] * N])
        temp_ = np.concatenate([temp_, [int(wl.split('-')[0])] * N])
    return [A_, temp_, C_, O_, l_.squeeze()]

def prep_nnr_tr(data_dict, wl_list, wl2aid, wl2wle):
    C_tr, l_tr = [get_global_tr(data_dict, wl_list, p_name) for p_name in ['confs', 'objs']]
    A_tr = np.array([])
    wle_tr = None
    for wl in wl_list:
        N = data_dict[wl]['confs'].shape[0]
        A_tr = np.concatenate([A_tr, [wl2aid[wl]] * N])
        wle_local = np.tile(wl2wle[wl], (N, 1))
        wle_tr = np.concatenate([wle_tr, wle_local]) if wle_tr is not None else wle_local
    X_tr = np.concatenate([C_tr, wle_tr], axis=1)
    data_nnr_tr = {
        "A": A_tr,
        "X": X_tr,
        "y": l_tr.squeeze(),
    }
    return data_nnr_tr

def prep_nnr_te(data_dict, wl_list, wl_list_off, wl_list_tr, wl2aid, wl2wle, nnb_dict_verbose, obs_num):
    C_, l_ = [get_global_te_obs(data_dict, wl_list, wl_list_off, p_name, obs="unseen") for p_name in ['confs', 'objs']]
    A_, X_, X_proxy_ = prep_nnr_te_internal(data_dict, C_, wl_list, wl_list_off, wl_list_tr,
                                            wl2aid, wl2wle, nnb_dict_verbose, obs_num=obs_num, obs="unseen")
    C_seen_, l_seen_ = [get_global_te_obs(data_dict, wl_list, wl_list_off, p_name, obs="seen", obs_num=obs_num)
                        for p_name in ['confs', 'objs']]
    A_seen_, X_seen_, X_seen_proxy_ = prep_nnr_te_internal(data_dict, C_seen_, wl_list, wl_list_off, wl_list_tr,
                                                           wl2aid, wl2wle, nnb_dict_verbose,
                                                           obs_num=obs_num, obs="seen")
    data_nnr = {
        'A': A_,
        'X': X_,
        'X_proxy': X_proxy_,
        'y': l_.squeeze(),
        'A_seen': A_seen_,
        'X_seen': X_seen_,
        'X_seen_proxy': X_seen_proxy_,
        'y_seen': l_seen_.squeeze(),
    }
    return data_nnr


def prep_nnr_te_internal(data_dict, C, wl_list, wl_list_off, wl_list_tr,
                         wl2aid, wl2wle, nnb_dict_verbose, obs_num, obs='unseen'):
    A = np.array([])
    wle = None
    wle_proxy = None
    for wl in wl_list:
        if obs == "seen" and wl in wl_list_off:
            continue
        N = data_dict[wl][obs]['confs'].shape[0]
        A = np.concatenate([A, [wl2aid[wl]] * N])
        wle_local = np.tile(wl2wle[wl], (N, 1)) if wl in wl_list_tr else np.tile(wl2wle[wl][f'seen_{obs_num}'], (N, 1))
        wle = np.concatenate([wle, wle_local]) if wle is not None else wle_local
        wle_proxy_local = np.tile(wl2wle[wl], (N, 1)) if wl in wl_list_tr else np.tile(wl2wle[nnb_dict_verbose[wl][f'seen_{obs_num}']], (N, 1))
        wle_proxy = np.concatenate([wle_proxy, wle_proxy_local]) if wle_proxy is not None else wle_proxy_local
    X = np.concatenate([C, wle], axis=1)
    X_proxy = np.concatenate([C, wle_proxy], axis=1)
    return A, X, X_proxy


def get_node_index(configs):
    wl_split = configs['workloads']['wl_split']
    node_list = wl_split["worker_nodes"]
    curr_node = os.uname().nodename
    if curr_node in node_list:
        return node_list.index(curr_node), len(node_list)
    else:
        return -1, len(node_list)

def get_train_order(configs):
    wl_split = configs['workloads']['wl_split']
    order_list = wl_split["train_order"]
    return order_list

def runtime_log(configs, params_all, node_index, off_data_used, on_data_used, runtime_secs, mape, mape_nnb, train_cycle, fine_tune=False, filename="runtime_csv"):
    try:
        ae_params = params_all['ae']
        nnr_params = params_all['nnr']
        ae_params_list = get_ae_params(ae_params)
        nnr_params_list = get_nnr_params(nnr_params)
    except:
        raise Exception('hyperparams invalid')

    wl_split = configs['workloads']['wl_split']
    log_path_prefix = wl_split["log_path_prefix"]

    node_name = wl_split["worker_nodes"][node_index]

    ae_sign = get_sign(ae_params_list)
    nnr_sign = get_sign(nnr_params_list)
    des_path = f"{log_path_prefix}/{filename}.log"
    exists = False
    header = ['timestamp', 'node', 'train_cycle', 'fine_tune', 'off_data_used' , 'on_data_used' ,'runtime(secs)', 'ae_sign', 'nnr_sign', 'mape_te', 'mape_te_nnb']
    line = [time.time(), node_name, train_cycle, fine_tune, off_data_used, on_data_used, runtime_secs, ae_sign, nnr_sign, mape, mape_nnb]
    if os.path.exists(des_path):
        exists = True
    else:
        try:
            os.stat(f"{log_path_prefix}")
        except:
            os.makedirs(f"{log_path_prefix}")
    # if exists:
    #     mode = 'a'
    # else:
    #     mode = 'w'
    with open(des_path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        if not exists:
            writer.writerow(header)  # write the header
        # write the actual content line by line
        writer.writerow(line)
        f.flush()


def update_for_sync(configs, params_all, node_index, mape, train_cycle, ckp_path="checkpoints/udao"):
    try:
        ae_params = params_all['ae']
        nnr_params = params_all['nnr']
        ae_params_list = get_ae_params(ae_params)
        nnr_params_list = get_nnr_params(nnr_params)
    except:
        raise Exception('hyperparams invalid')

    wl_split = configs['workloads']['wl_split']
    #log_path_prefix = wl_split["log_path_prefix"]
    #ckp_path_prefix_base = configs['workloads']['ckp_path_prefix']

    node_name = wl_split["worker_nodes"][node_index]

    ae_sign = get_sign(ae_params_list)
    nnr_sign = get_sign(nnr_params_list)
    sync_file_des_path = f"{ckp_path}/{train_cycle}/{train_cycle}.txt"
    file_exists = False
    if os.path.exists(sync_file_des_path):
        file_exists = True
    else:
        try:
            os.stat(f"{ckp_path}/{train_cycle}")
        except:
            os.makedirs(f"{ckp_path}/{train_cycle}")

    header = ['node_name', 'train_cycle', 'ae_sign', 'nnr_sign', 'mape']
    line_status = [node_name, train_cycle, ae_sign, nnr_sign, mape]
    with open(sync_file_des_path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        if not file_exists:
            writer.writerow(header)  # write the header
        # write the actual content line by line
        writer.writerow(line_status)
        f.flush()


def all_job_complete(configs, hyper_combis, train_cycle, ckp_path="checkpoints/udao"):
    #wl_split = configs['workloads']['wl_split']
    #log_path_prefix = wl_split["log_path_prefix"]
    #ckp_path_prefix_base = configs['workloads']['ckp_path_prefix']
    sync_file_des_path = f"{ckp_path}/{train_cycle}/{train_cycle}.txt"
    if os.path.exists(sync_file_des_path):
        try:
            jobs_df = pd.read_csv(sync_file_des_path)
        except:
            os.makedirs(f"{ckp_path}/{train_cycle}")
    if hyper_combis == len(jobs_df.index):
        return True
    else:
        return False


def fetch_fine_tune_data(configs, train_cycle, ckp_path = "checkpoints/udao"):
    #wl_split = configs['workloads']['wl_split']
    #log_path_prefix = wl_split["log_path_prefix"]
    #ckp_path_prefix_base = configs['workloads']['ckp_path_prefix']
    sync_file_des_path = f"{ckp_path}/{train_cycle}/{train_cycle}.txt"
    if os.path.exists(sync_file_des_path):
        try:
            jobs_df = pd.read_csv(sync_file_des_path)
        except:
            os.makedirs(f"{ckp_path}/{train_cycle}")
    #min_mape_row_index = jobs_df.idxmin(axis=0)['mape']
    min_mape_row_index = list(jobs_df['mape']).index(jobs_df['mape'].min())
    #min_mape_row = jobs_df.iloc[min_mape_row_index]
    #ae_sign = jobs_df.iloc[min_mape_row_index,['ae_sign']]#"1_1_0_0,0.0001,128,2,12"
    ae_sign = jobs_df.iloc[min_mape_row_index]['ae_sign']#"1_1_0_0,0.0001,128,2,12"
    #nnr_sign = jobs_df.iloc[min_mape_row_index,['nnr_sign']]#"0.0001,64,2,128_128_128_128"
    nnr_sign = jobs_df.iloc[min_mape_row_index]['nnr_sign']  # "0.0001,64,2,128_128_128_128"
    ae_sign_list = ae_sign.split(",")
    nnr_sign_list = nnr_sign.split(",")
    hyper_params = {
            "ae": {
                "lr": float(ae_sign_list[1]),#1e-4,
                "bs": int(ae_sign_list[2]),#128,
                "epochs": int(ae_sign_list[3]),#2,
                "weight_str": ae_sign_list[0],#"1_1_0_0",
                "W_dim": int(ae_sign_list[4])#12
            },
            "nnr": {
                "lr": float(nnr_sign_list[0]),#1e-4,
                "bs": int(nnr_sign_list[1]),#64,
                "epochs": int(nnr_sign_list[2]),#2,
                "cap_str": nnr_sign_list[3]#"128_128_128_128"
            }
        }
    des_path = f"{ckp_path}/{train_cycle}/{ae_sign}/ae.pth"
    return hyper_params, des_path

# data = {
#             "wl_dict": {
#                 "offline": off_wl,
#                 "online": wl_dict['online'],
#                 "tr": off_wl + on_wl_tr,
#                 "val": off_wl + on_wl_val,
#                 "te": off_wl + on_wl_te
#             },
#             "meta": {
#                 "internal_metric_names": cols_internal_metrics,
#                 "knob_names": cols_knobs,
#                 "objective_names": cols_targets,
#                 "wl2aid": wl2aid,
#                 "aid2wl": aid2wl,
#                 "off_aids_set": off_aids_set,
#                 "on_aids_set": on_aids_set,
#                 "temp_list": temp_wl_list,
#                 "temp_dict": temp_wl_dict
#             },
#             "tr": data_tr,
#             # {job: {"confs": [], "objs": [], "obses": []}},
#             "val": data_val,
#             # { # job_off: {'seen': [empty], 'unseen': {"confs": [], "objs": [], "obses": []}}
#             # },
#             # { # job_on: {'seen': {"confs": [], "objs": [], "obses": []}, 'unseen': {"confs": [], "objs": [], "obses": []}}
#             # },
#             "te": data_te
#             # { # job_off: {'seen': [empty], 'unseen': {"confs": [], "objs": [], "obses": []}}
#             # },
#             # { # job_on: {'seen': {"confs": [], "objs": [], "obses": []}, 'unseen': {"confs": [], "objs": [], "obses": []}}
#             # },
#         }


class DataManager:
    def __init__(self, data):
        self.data_master = data
        self.off_data_pos = 0
        self.on_data_pos = 0
        self.tr_wls = self.data_master['wl_dict']['tr']
        self.off_wls = self.data_master['wl_dict']['offline']
        self.on_wls = list(set(self.tr_wls)-set(self.off_wls))
        # wl_tr_first = self.tr_wls[0]
        # self.off_wl_index, self.off_wl_C_tr, self.off_wl_l_tr, self.off_wl_O_tr = \
        #     np.empty((0,1)), np.empty((0,self.data_master['tr'][wl_tr_first]['confs'].shape[1])), \
        #     np.empty((0,1)), np.empty((0,self.data_master['tr'][wl_tr_first]['obses'].shape[1]))
        # self.on_wl_index, self.on_wl_C_tr, self.on_wl_l_tr, self.on_wl_O_tr = \
        #     np.empty((0,1)), np.empty((0,self.data_master['tr'][wl_tr_first]['confs'].shape[1])), \
        #     np.empty((0,1)), np.empty((0,self.data_master['tr'][wl_tr_first]['obses'].shape[1]))
        #np.vstack((data_te_obses, np.zeros(data_te_obses.shape[1])))
        self.off_wl_index, self.off_wl_C_tr, self.off_wl_l_tr, self.off_wl_O_tr = \
            None, None, None, None
        self.on_wl_index, self.on_wl_C_tr, self.on_wl_l_tr, self.on_wl_O_tr = \
            None, None, None, None
        for wl in self.tr_wls:
            # if wl in self.off_wls:
            #     self.off_wl_C_tr = np.vstack((self.off_wl_C_tr, self.data_master['tr'][wl]['confs']))
            #     self.off_wl_l_tr = np.vstack((self.off_wl_l_tr, self.data_master['tr'][wl]['objs']))
            #     self.off_wl_O_tr = np.vstack((self.off_wl_O_tr, self.data_master['tr'][wl]['obses']))
            #     self.off_wl_index = np.vstack((self.off_wl_index, np.full(self.data_master['tr'][wl]['objs'].shape, wl)))
            # elif wl in self.on_wls:
            #     self.on_wl_C_tr = np.vstack((self.on_wl_C_tr,self.data_master['tr'][wl]['confs']))
            #     self.on_wl_l_tr = np.vstack((self.on_wl_l_tr, self.data_master['tr'][wl]['objs']))
            #     self.on_wl_O_tr = np.vstack((self.on_wl_O_tr, self.data_master['tr'][wl]['obses']))
            #     self.on_wl_index = np.vstack((self.on_wl_index, np.full(self.data_master['tr'][wl]['objs'].shape, wl)))
            if wl in self.off_wls:
                if self.off_wl_C_tr is None:
                    self.off_wl_C_tr = self.data_master['tr'][wl]['confs']
                else:
                    self.off_wl_C_tr = np.vstack((self.off_wl_C_tr, self.data_master['tr'][wl]['confs']))
                if self.off_wl_l_tr is None:
                    self.off_wl_l_tr = self.data_master['tr'][wl]['objs']
                else:
                    self.off_wl_l_tr = np.vstack((self.off_wl_l_tr, self.data_master['tr'][wl]['objs']))
                if self.off_wl_O_tr is None:
                    self.off_wl_O_tr = self.data_master['tr'][wl]['obses']
                else:
                    self.off_wl_O_tr = np.vstack((self.off_wl_O_tr, self.data_master['tr'][wl]['obses']))
                if self.off_wl_index is None:
                    self.off_wl_index = np.full(self.data_master['tr'][wl]['objs'].shape, wl)
                else:
                    self.off_wl_index = np.vstack((self.off_wl_index, np.full(self.data_master['tr'][wl]['objs'].shape, wl)))
            elif wl in self.on_wls:
                if self.on_wl_C_tr is None:
                    self.on_wl_C_tr = self.data_master['tr'][wl]['confs']
                else:
                    self.on_wl_C_tr = np.vstack((self.on_wl_C_tr,self.data_master['tr'][wl]['confs']))
                if self.on_wl_l_tr is None:
                    self.on_wl_l_tr = self.data_master['tr'][wl]['objs']
                else:
                    self.on_wl_l_tr = np.vstack((self.on_wl_l_tr, self.data_master['tr'][wl]['objs']))
                if self.on_wl_O_tr is None:
                    self.on_wl_O_tr = self.data_master['tr'][wl]['obses']
                else:
                    self.on_wl_O_tr = np.vstack((self.on_wl_O_tr, self.data_master['tr'][wl]['obses']))
                if self.on_wl_index is None:
                    self.on_wl_index = np.full(self.data_master['tr'][wl]['objs'].shape, wl)
                else:
                    self.on_wl_index = np.vstack((self.on_wl_index, np.full(self.data_master['tr'][wl]['objs'].shape, wl)))
        self.off_wl_index, self.off_wl_C_tr, self.off_wl_l_tr, self.off_wl_O_tr = \
            skl_shuffle(self.off_wl_index, self.off_wl_C_tr, self.off_wl_l_tr, self.off_wl_O_tr, random_state=0)
        self.on_wl_index, self.on_wl_C_tr, self.on_wl_l_tr, self.on_wl_O_tr = \
            skl_shuffle(self.on_wl_index, self.on_wl_C_tr, self.on_wl_l_tr, self.on_wl_O_tr, random_state=1)
        self.off_wl_index = self.off_wl_index.flatten()
        self.on_wl_index = self.on_wl_index.flatten()

    def get_off_tr_data_rem(self):
        return (self.off_wl_index.shape[0] - self.off_data_pos)

    def get_on_tr_data_rem(self):
        return (self.on_wl_index.shape[0] - self.on_data_pos)

    def get_off_data_pos(self):
        return self.off_data_pos

    def get_on_data_pos(self):
        return self.on_data_pos

    def _get_off_tr_data(self, size=5000):
        size_rem = self.get_off_tr_data_rem()
        assert size_rem > 0, "No more training data remaining"
        if size_rem < size:
            print(f'Fetched training data size={size_rem} less than requested')
            fetch = size_rem
        else:
            fetch = size
        data_tr = {}
        wl_dict_tr_updated = []
        data = None
        #for i, wl in enumerate(self.off_wl_index[self.off_data_pos: self.off_data_pos + fetch]):
        for i, wl in enumerate(self.off_wl_index[0: self.off_data_pos+fetch]):
            wl_C = self.off_wl_C_tr[i:i+1,]
            wl_l = self.off_wl_l_tr[i:i+1]
            wl_O = self.off_wl_O_tr[i:i+1,]
            data = "Added"
            try:#data_tr[wl] = {"confs": wl_C, "objs": wl_l, "obses": wl_O}
                tr_wl_conf = data_tr[wl]['confs']
                tr_wl_objs = data_tr[wl]['objs']
                tr_wl_obses = data_tr[wl]['obses']
                found_wl = True
            except:
                found_wl=False
                #tr_wl_conf = np.empty((0,wl_C.shape[0]))
                #tr_wl_objs = np.empty((0,1))
                #tr_wl_obses = np.empty((0,wl_O.shape[0]))
            if found_wl:
                data_tr[wl] = {
                    "confs": np.vstack((tr_wl_conf, wl_C)),
                    "objs": np.vstack((tr_wl_objs, wl_l)),
                    "obses": np.vstack((tr_wl_obses, wl_O))
                }
            else:
                data_tr[wl] = {"confs": wl_C, "objs": wl_l, "obses": wl_O}
            wl_dict_tr_updated.append(wl)
        self.off_data_pos += size
        wl_dict_tr_updated = list(set(wl_dict_tr_updated))
        data_copy = copy.deepcopy(self.data_master)
        data_copy['tr'] = data_tr
        data_copy['wl_dict']['tr'] = wl_dict_tr_updated
        data_copy['data'] = data
        return data_copy

    def _get_on_inc_tr_data(self, size=1000):
        size_rem = self.get_on_tr_data_rem()
        assert size_rem > 0, "No more incremental training data remaining"
        if size_rem < size:
            print(f'Fetched incremental train data size={size_rem} less than requested')
            fetch = size_rem
        else:
            fetch = size
        data_tr = {}
        wl_dict_tr_updated = []
        data = None
        #for i, wl in enumerate(self.on_wl_index[self.on_data_pos: self.on_data_pos+fetch]):
        for i, wl in enumerate(self.on_wl_index[0: self.on_data_pos + fetch]):
            #wl = self.off_wl_index[i]
            wl_C = self.on_wl_C_tr[i:i+1,]
            wl_l = self.on_wl_l_tr[i:i+1,]
            wl_O = self.on_wl_O_tr[i:i+1,]
            data = "Added"
            try:#data_tr[wl] = {"confs": wl_C, "objs": wl_l, "obses": wl_O}
                tr_wl_conf = data_tr[wl]['confs']
                tr_wl_objs = data_tr[wl]['objs']
                tr_wl_obses = data_tr[wl]['obses']
                found_wl = True
            except:
                found_wl=False
                #tr_wl_conf = np.empty((0,wl_C.shape[0]))
                #tr_wl_objs = np.empty((0,1))
                #tr_wl_obses = np.empty((0,wl_O.shape[0]))
            if found_wl:
                data_tr[wl] = {
                    "confs": np.vstack((tr_wl_conf, wl_C)),
                    "objs": np.vstack((tr_wl_objs, wl_l)),
                    "obses": np.vstack((tr_wl_obses, wl_O))
                }
            else:
                data_tr[wl] = {"confs": wl_C, "objs": wl_l, "obses": wl_O}
            wl_dict_tr_updated.append(wl)
        self.on_data_pos += size
        wl_dict_tr_updated = list(set(wl_dict_tr_updated))
        data_copy = copy.deepcopy(self.data_master)
        data_copy['tr'] = data_tr
        data_copy['wl_dict']['tr'] = wl_dict_tr_updated
        data_copy['data'] = data
        return data_copy

    def get_tr_data(self, incremental=True, size=1000):
        data_tr = {}
        wl_dict = []
        on_wl_tr = []
        if incremental:
            on_data = self._get_on_inc_tr_data(size=size)
            off_data = self._get_off_tr_data(size=0)
        else:
            on_data = self._get_on_inc_tr_data(size=0)
            off_data = self._get_off_tr_data(size=size)
        if on_data['data'] is not None:
            on_data_tr = on_data['tr']
            on_wl_tr = on_wl_dict = on_data['wl_dict']['tr']
            data_tr.update(on_data_tr)
            wl_dict = list(set(on_wl_dict) | set(wl_dict))
        if off_data['data'] is not None:
            off_data_tr = off_data['tr']
            off_wl_dict = off_data['wl_dict']['tr']
            data_tr.update(off_data_tr)
            wl_dict = list(set(off_wl_dict) | set(wl_dict))
        data_copy = copy.deepcopy(self.data_master)
        data_copy['tr'] = data_tr
        data_copy['wl_dict']['tr'] = wl_dict

        ##Update metadata of the dataset
        off_wl = data_copy['wl_dict']['offline']
        on_wl = data_copy['wl_dict']['online']
        #off_wl_dict #off_wl_tr not required. same as off_wl
        #on_wl_dict #on_wl_tr
        on_wl_val = list(set(data_copy['wl_dict']['val']) - set(off_wl))
        on_wl_te = list(set(data_copy['wl_dict']['te']) - set(off_wl))

        wls = off_wl + on_wl_tr + on_wl_val + on_wl_te
        temp_list = data_copy['meta']['temp_list']
        temp_dict = {t: [wl for wl in wls if wl.split('-')[0] == t] for t in temp_list}

        wl2aid, aid2wl = {}, {}
        off_aids_set, on_aids_set = set(), set()

        # speed up for search
        wl_off_set = set(off_wl)
        #wl_tr_set = set(on_wl_tr + off_wl_dict)
        #wl_val_set = set(data_copy['wl_dict']['val'])

        for aid, wl in enumerate(off_wl + on_wl_tr + on_wl_val + on_wl_te):
            wl2aid[wl] = aid
            aid2wl[aid] = wl
            if wl in wl_off_set:
                off_aids_set.add(aid)
            else:
                on_aids_set.add(aid)
        data_copy['meta']['wl2aid'] = wl2aid
        data_copy['meta']['aid2wl'] = aid2wl
        data_copy['meta']['off_aids_set'] = off_aids_set
        data_copy['meta']['on_aids_set'] = on_aids_set
        data_copy['meta']['temp_dict'] = temp_dict
        return data_copy

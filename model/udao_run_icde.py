##################
import yaml
import numpy as np
import model.utils as ut
import itertools as itools
from model.udao import UDAOrunner
from model.utils import DataManager
import time
from sys import exit
import os

CONFIG_PATH = "configs/icde.yaml"
print(os.getcwd())
with open(CONFIG_PATH) as f:
    configs_ = yaml.load(f, Loader=yaml.FullLoader)
data_ = ut.load_data_icde(configs_, "cache/icde_data.pkl")

index, workers = ut.get_node_index(configs_)
if index == -1:
    print(f"This \"{os.uname().nodename}\" is not in worker_nodes. Check icde.yaml")
    exit()

datman_ = DataManager(data_)

#distributed execution hyper-parameters
l_p_all_0_ae_lr = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]#defines min number of nodes req
l_p_all_0_ae_bs = [64, 128, 256]
l_p_all_0_nnr_lr = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
l_p_all_0_nnr_bs = [64, 128, 256]

#1-node test cases
#l_p_all_0_ae_lr = [1e-4]
#l_p_all_0_ae_bs = [64, 128]
#l_p_all_0_nnr_lr = [1e-4]
#l_p_all_0_nnr_bs = [64]

# if len(l_p_all_0_ae_lr)>len(l_p_all_0_nnr_lr):
#     req_workers = len(l_p_all_0_ae_lr)
# else:
#     req_workers = len(l_p_all_0_nnr_lr)

if index > len(l_p_all_0_ae_lr):
    print(f"Excess worker node. No task assigned. Stopping")
    exit()
# elif workers < len(l_p_all_0_ae_lr):
#     print(f"Required {req_workers} worker nodes. Include more")

train_order = ut.get_train_order(configs_)
ckp_path_prefix_base = configs_['workloads']['ckp_path_prefix']
train_round = 0
inc_round = 0
train_cycle = ""
best_hyper_param = None
best_ae_path = None

for order in train_order:
    if order == "tr":
        train_data = datman_.get_tr_data(incremental=False, size=5000)
        train_round += 1
        inc_round = 0
        train_cycle = str(train_round) + "-retrain"
        configs_['workloads']['ckp_path_prefix'] = \
            ckp_path_prefix_base + "/" + train_cycle
        print(configs_['workloads']['ckp_path_prefix'])
    elif order=="inc":
        train_data = datman_.get_tr_data()
        inc_round += 1
        train_cycle = str(train_round) + "." + str(inc_round) + "-incr"
        configs_['workloads']['ckp_path_prefix'] = \
            ckp_path_prefix_base + "/" + train_cycle
        print(configs_['workloads']['ckp_path_prefix'])

    #l_p_all_0_ae_lr_selected = [l_p_all_0_ae_lr[index]]
    #l_p_all_0_nnr_lr_selected = l_p_all_0_nnr_lr[index]

    hyp_combi = itools.product(l_p_all_0_ae_lr, l_p_all_0_ae_bs, l_p_all_0_nnr_lr, l_p_all_0_nnr_bs)
    comb_list = [comb for comb in hyp_combi]
    hypara_combinations = len(comb_list)

    if order == "tr":

        start_tr = time.time()
        for hypara_combi in itools.product([l_p_all_0_ae_lr[index]], l_p_all_0_ae_bs, l_p_all_0_nnr_lr, l_p_all_0_nnr_bs):
        #for hypara_combi in itools.product([l_p_all_0_ae_lr[index]], l_p_all_0_ae_bs, l_p_all_0_nnr_lr, l_p_all_0_nnr_bs):
            local_params_all_0 = {
                "ae": {
                    "lr": hypara_combi[0], # learning rate: 1e-4, 3e-4, 1e-3, 3e-3, 1e-2
                    "bs": hypara_combi[1], # batch_size: 64, 128, 256
                    "epochs": 2, # 200 as big enough
                    "weight_str": "1_1_0_0", # fixed
                    "W_dim": 12 # fixed
                },
                "nnr": {
                    "lr": hypara_combi[2], # 1e-4, 3e-4, 1e-3, 3e-3, 1e-2
                    "bs": hypara_combi[3], # 64, 128, 256
                    "epochs": 2, # 200 as big enough
                    "cap_str": "128_128_128_128" # fixed
                }
            }

            params_in_one = local_params_all_0
            fine_tune_path = None
            ur_ = UDAOrunner(data = train_data, max_seen_size=configs_['workloads']['max_seen_size'],
                             ckp_path=configs_['workloads']['ckp_path_prefix'])
            ur_.run_in_one(params_in_one, fine_tune_path=fine_tune_path)

            wl2wle_, nnb_dict_, nnb_dict_verbose_ = ur_.wl2wle, ur_.nnb_dict, ur_.nnb_dict_verbose
            obs_num = configs_['workloads']['max_seen_size']
            # temp_list = data_['meta']['temp_list']
            # temp_dict = data_['meta']['temp_dict']

            print(f"Observed_samples={obs_num}")
            # print(f"mape_all, match_rate, mape_{', mape_'.join(temp_list)}, mape_sign")
            # matched = ut.get_matched_rate(nnb_dict_verbose_, obs_num)
            # stats = ur_.get_calibrate_MAPE_dict(obs_num=obs_num, is_te=True)
            stats = ur_.get_MAPE_dict(obs_num=obs_num, is_te=True)

            # for mape_sign in ['mape-unseen','mape-unseen-nnb']:
            #     mape_dict = stats[mape_sign]
            #     matched = ut.get_matched_rate(nnb_dict_verbose_, obs_num)
            #     mape_avg = np.mean(list(mape_dict.values()))
            #     mape_list = [np.mean([mape_dict[wl_] for wl_ in temp_dict[t] if wl_ in mape_dict]) for t in temp_list]
            #     mape_list_str = [f"{m:.5f}" for m in mape_list]
            #     print(f"{mape_avg:.5f}, {matched*100:.0f}%, {', '.join(mape_list_str)}, {mape_sign}")

            mape_te = np.mean(list(stats['mape-unseen'].values()))
            mape_te_nnb = np.mean(list(stats['mape-unseen-nnb'].values()))
            print(f'MAPE_te={mape_te:.5f}, MAPE_te(with nearest neighbor)={mape_te_nnb:.5f}')

            stop_tr = time.time()
            off_data_used = datman_.get_off_data_pos()
            on_data_used = datman_.get_on_data_pos()
            ut.runtime_log(configs_, params_in_one, index, off_data_used, on_data_used, stop_tr - start_tr, mape_te, mape_te_nnb, train_cycle=train_cycle, fine_tune=False)
            ut.update_for_sync(configs_, params_in_one, index, mape_te, train_cycle, ckp_path_prefix_base)

        if index == 0:
            while not ut.all_job_complete(configs_, hypara_combinations, train_cycle, ckp_path_prefix_base):
                print(f'Retrain_cycle={train_cycle}, waiting to sync with worker nodes...')
                time.sleep(180)
            #best_hyper_param, best_ae_path = ut.fetch_fine_tune_data(configs_, train_cycle, ckp_path_prefix_base)
            best_hyper_param, fine_tune_path = ut.fetch_fine_tune_data(configs_, train_cycle, ckp_path_prefix_base)

    if index == 0 and order=="inc":

        local_params_all_1 = best_hyper_param
        #fine_tune_path = best_ae_path
        start_ftune = time.time()
        # local_params_all_1 = {
        #     "ae": {
        #         "lr": 1e-4,
        #         "bs": 128,
        #         "epochs": 2,
        #         "weight_str": "1_1_0_0",
        #         "W_dim": 12
        #     },
        #     "nnr": {
        #         "lr": 1e-4,
        #         "bs": 64,
        #         "epochs": 2,
        #         "cap_str": "128_128_128_128"
        #     }
        # }
        # fine_tune_path = "checkpoints/udao/1-retrain/1_1_0_0,0.0001,128,2,12/ae.pth"
        ur_ = UDAOrunner(data = train_data, max_seen_size=configs_['workloads']['max_seen_size'],
                             ckp_path=configs_['workloads']['ckp_path_prefix'])
        ur_.run_in_one(local_params_all_1, fine_tune_path=fine_tune_path)
        wl2wle_, nnb_dict_, nnb_dict_verbose_ = ur_.wl2wle, ur_.nnb_dict, ur_.nnb_dict_verbose
        obs_num = configs_['workloads']['max_seen_size']
        stats = ur_.get_MAPE_dict(obs_num=obs_num, is_te=True)
        mape_te = np.mean(list(stats['mape-unseen'].values()))
        mape_te_nnb = np.mean(list(stats['mape-unseen-nnb'].values()))
        stop_ftune = time.time()
        best_hyper_param_ae_sign = ut.get_ae_sign(best_hyper_param)
        fine_tune_path = f"{ckp_path_prefix_base}/{train_cycle}/{best_hyper_param_ae_sign}/ae.pth"
        print(f'MAPE_te={mape_te:.5f}, MAPE_te(with nearest neighbor)={mape_te_nnb:.5f}')
        off_data_used = datman_.get_off_data_pos()
        on_data_used = datman_.get_on_data_pos()
        ut.runtime_log(configs_, local_params_all_1, index, off_data_used, on_data_used,stop_ftune - start_ftune, mape_te, mape_te_nnb, train_cycle=train_cycle, fine_tune=True)
timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
print(f"End of task for {os.uname().nodename} at {timestamp}")

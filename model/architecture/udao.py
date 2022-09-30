import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from model.architecture.autoencoder import AutoEncoder
from model.architecture.ae_nnr_adaptor import AENetNNRNetAdaptor
from model.architecture.neuralnetworkregressor import NeuralNetworkRegressor
import utils.model.utils as utils
import utils.model.model_utils as ut

class UDAOrunner:
    def __init__(self, data, max_seen_size, ckp_path="checkpoints/udao"):
        self.wl_dict = data['wl_dict']
        self.meta = data['meta']
        self.data_tr, self.data_val, self.data_te = data['tr'], data['val'], data['te']
        self.max_seen_size = max_seen_size
        self._normalized()
        self.ae_data = self.prep_data_ae()
        self.ae = None
        self.nnr_data_dict = None
        self.nnr_dict = None
        self.ckp_path = ckp_path
        self.dtype, self.device = ut.DTYPE, ut.DEVICE
        self.wl2wle, self.nnb_dict, self.nnb_dict_verbose = None, None, None

    def _normalized(self):
        data_tr, data_val, data_te = self.data_tr, self.data_val, self.data_te
        C_tr, l_tr, O_tr = [ut.get_global_tr(data_tr, self.wl_dict['tr'], p_name)
                            for p_name in ['confs', 'objs', 'obses']]

        # normalize C and O by MinMax
        C_scaler = MinMaxScaler(copy=False)
        C_scaler.fit(C_tr)
        # remove constant metrics
        O_ind = np.where(O_tr.std(0) != 0)[0]
        O_tr = O_tr[:, O_ind]
        O_scaler = MinMaxScaler(copy=False)
        O_scaler.fit_transform(O_tr)
        self.O_dim = len(O_ind)

        for wl in self.wl_dict['tr']:
            data_tr[wl]['confs'] = C_scaler.transform(data_tr[wl]['confs'])
            data_tr[wl]['obses'] = O_scaler.transform(data_tr[wl]['obses'][:, O_ind])
        for wl in self.wl_dict['val']:
            if wl in self.wl_dict['online']:
                data_val[wl]['seen']['confs'] = C_scaler.transform(data_val[wl]['seen']['confs'])
                data_val[wl]['seen']['obses'] = O_scaler.transform(data_val[wl]['seen']['obses'][:, O_ind])
            data_val[wl]['unseen']['confs'] = C_scaler.transform(data_val[wl]['unseen']['confs'])
            data_val[wl]['unseen']['obses'] = O_scaler.transform(data_val[wl]['unseen']['obses'][:, O_ind])
        for wl in self.wl_dict['te']:
            if wl in self.wl_dict['online']:
                data_te[wl]['seen']['confs'] = C_scaler.transform(data_te[wl]['seen']['confs'])
                data_te[wl]['seen']['obses'] = O_scaler.transform(data_te[wl]['seen']['obses'][:, O_ind])
            data_te[wl]['unseen']['confs'] = C_scaler.transform(data_te[wl]['unseen']['confs'])
            data_te[wl]['unseen']['obses'] = O_scaler.transform(data_te[wl]['unseen']['obses'][:, O_ind])

    def prep_data_ae(self):
        ae_tr_np = ut.prep_ae_tr(data_dict=self.data_tr, wl_list=self.wl_dict['tr'], wl2aid=self.meta['wl2aid'])
        ae_val_np = ut.prep_ae_te(data_dict=self.data_val, wl_list=self.wl_dict['val'],
                                  wl_list_off=self.wl_dict['offline'], wl2aid=self.meta['wl2aid'])
        # ae_te_np = ut.prep_ae_te(data_dict=self.data_te, wl_list=self.wl_dict['te'],
        #                          wl_list_off=self.wl_dict['offline'], wl2aid=self.meta['wl2aid'])
        ae_te_np = None
        return [ae_tr_np, ae_val_np, ae_te_np]

    def prep_data_nnr(self, wl2wle, nnb_dict_verbose, obs_num=None):
        obs_num = self.max_seen_size if obs_num is None else obs_num
        data_tr, data_val, data_te = self.data_tr, self.data_val, self.data_te
        nnr_tr_np = ut.prep_nnr_tr(data_tr, self.wl_dict['tr'], self.meta['wl2aid'], wl2wle)
        # TODO: the val and te need to be adjusted
        nnr_val_np = ut.prep_nnr_te(data_val, self.wl_dict['val'], self.wl_dict['offline'], self.wl_dict['tr'],
                                    self.meta['wl2aid'], wl2wle, nnb_dict_verbose, obs_num)
        nnr_te_np = ut.prep_nnr_te(data_te, self.wl_dict['te'], self.wl_dict['offline'], self.wl_dict['tr'],
                                   self.meta['wl2aid'], wl2wle, nnb_dict_verbose, obs_num)
        return [nnr_tr_np, nnr_val_np, nnr_te_np]

    def run_ae(self, ae_params_list, fine_tune_path=None):
        ae_sign = ut.get_sign(ae_params_list)
        weight_str, lr, bs, epochs, W_dim = ae_params_list
        weight_list = [float(w) for w in weight_str.split('_')]
        des_path = f"{self.ckp_path}/{ae_sign}/ae.pth"
        ae = AutoEncoder(W_dim=W_dim, cap_list=None, O_dim=self.O_dim)
        if os.path.exists(des_path):
            print(f'ae model {ae_sign} found!')
            ckp = torch.load(des_path, map_location=self.device)
            saved_model_state = ckp["model_state"]
            ae.model.load_state_dict(saved_model_state)
        else:
            if fine_tune_path is not None:
                if os.path.exists(fine_tune_path):
                    print(f'got an initial model to fine tune')
                    ckp = torch.load(fine_tune_path, map_location=lambda storage, loc: storage)
                    saved_model_state = ckp["model_state"]
                    ae.model.load_state_dict(saved_model_state)
                else:
                    raise Exception(f'fine tuned path {fine_tune_path} not found')
            print(f'ae model {ae_sign} not found, start training ...')

            ae_tr_np, ae_val_np, _ = self.ae_data
            loss_tr, loss_val = ae.fit(tr_np=ae_tr_np, val_np=ae_val_np, lr=lr, bs=bs, epochs=epochs, patience=10,
                                       verbose=True, weight_list=weight_list, to_plot=True)
            try:
                os.stat(f"{self.ckp_path}/{ae_sign}")
            except:
                os.makedirs(f"{self.ckp_path}/{ae_sign}")
            torch.save({
                "loss_tr": loss_tr,
                "loss_val": loss_val,
                "model_state": ae.model.state_dict()
            }, des_path)
            self.plot_ae_loss(loss_tr, loss_val, ae_sign)

        return ae

    def run_ana_wle(self, ae, ae_params_list):
        ae_sign = ut.get_sign(ae_params_list)
        des_path = f"{self.ckp_path}/{ae_sign}/ana_wle_and_nnb.pth"
        if os.path.exists(des_path):
            print(f'found cached nn_input for {ae_sign}!')
            d = torch.load(des_path,  map_location=self.device)
            return [d['wl2wle'], d['nnb_dict'], d['nnb_dict_verbose']]
        else:
            print(f'not found cached nn_input for {ae_sign}, start generating...')
            ana = AENetNNRNetAdaptor(ae=ae)
            wl_lists = [self.wl_dict['tr'], self.wl_dict['val'], self.wl_dict['te']]
            data_all = [self.data_tr, self.data_val, self.data_te]
            mss = self.max_seen_size
            wl2wle = ana.get_wle(data_all=data_all, wl_lists=wl_lists, mss=mss)
            nnb_dict, nnb_dict_verbose = ana.get_nnb_dict(wl2wle, wl_lists, mss)
            torch.save({
                "wl2wle": wl2wle,
                "nnb_dict": nnb_dict,
                "nnb_dict_verbose": nnb_dict_verbose
            }, des_path)
            return [wl2wle, nnb_dict, nnb_dict_verbose]

    def run_nnr(self, ae_params_list, nnr_params_list, nnr_data, obs_num):
        ae_sign = ut.get_sign(ae_params_list)
        nnr_sign = f"obs{obs_num}-" + ut.get_sign(nnr_params_list)
        des_path = f"{self.ckp_path}/{ae_sign}/{nnr_sign}.pth"

        W_dim = ae_params_list[-1]
        lr, bs, epochs, cap_str = nnr_params_list
        cap_list = [int(cap) for cap in cap_str.split('_')]
        nnr = NeuralNetworkRegressor(cap_list, W_dim=W_dim)
        if os.path.exists(des_path):
            print(f'nnr model {ae_sign}==>{nnr_sign} found!')
            ckp = torch.load(des_path, map_location=self.device)
            saved_model_state = ckp["model_state"]
            nnr.model.load_state_dict(saved_model_state)
        else:
            nnr_tr_np, nnr_val_np, _ = nnr_data
            loss_tr, loss_val = nnr.fit(nnr_tr_np, nnr_val_np, lr=lr, bs=bs, epochs=epochs, patience=10,
                                              verbose=True, to_plot=True)
            try:
                os.stat(f"{self.ckp_path}/{ae_sign}")
            except:
                os.makedirs(f"{self.ckp_path}/{ae_sign}")
            torch.save({
                "loss_tr": loss_tr,
                "loss_val": loss_val,
                "model_state": nnr.model.state_dict(),
            }, des_path)
        return nnr

    def run_in_one(self, params_all, fine_tune_path=None):
        try:
            ae_params = params_all['ae']
            nnr_params = params_all['nnr']
            ae_params_list = ut.get_ae_params(ae_params)
            nnr_params_list = ut.get_nnr_params(nnr_params)
        except:
            raise Exception('hyperparams invalid')
        ae = self.run_ae(ae_params_list=ae_params_list, fine_tune_path=fine_tune_path)
        wl2wle, nnb_dict, nnb_dict_verbose = self.run_ana_wle(ae, ae_params_list)
        nnr_dict = {}
        nnr_data_dict = {}
        # for obs_num in range(1, self.max_seen_size+1):
        obs_num = self.max_seen_size
        nnr_data = self.prep_data_nnr(wl2wle, nnb_dict_verbose, obs_num=obs_num)
        nnr = self.run_nnr(ae_params_list=ae_params_list, nnr_params_list=nnr_params_list,
                           nnr_data=nnr_data, obs_num=obs_num)
        nnr_data_dict[obs_num] = nnr_data
        nnr_dict[obs_num] = nnr

        self.wl2wle, self.nnb_dict, self.nnb_dict_verbose = wl2wle, nnb_dict, nnb_dict_verbose
        self.ae, self.nnr_data_dict, self.nnr_dict = ae, nnr_data_dict, nnr_dict

    def get_MAPE_helper(self, nnr, A, X, X_nnb, y, wl_list):
        X_torch_, X_nnb_torch_ = [utils.get_tensor(var_np) for var_np in [X, X_nnb]]
        with torch.no_grad():
            y_pred_ = nnr.model.forward(X_torch_).detach().cpu().numpy()
            y_pred_nnb_ = nnr.model.forward(X_nnb_torch_).detach().cpu().numpy()
        mape_ = np.abs(y_pred_ - y) / y
        mape_nnb_ = np.abs(y_pred_nnb_ - y) / y
        mape_dict, mape_nnb_dict = {}, {}
        l_real_dict, l_pred_dict, l_pred_nnb_dict = {}, {}, {}
        for wl in wl_list:
            a = self.meta['wl2aid'][wl]
            inds = np.argwhere(a == A).squeeze()
            mape_dict[wl] = mape_[inds].mean()
            mape_nnb_dict[wl] = mape_nnb_[inds].mean()
            l_real_dict[wl] = y[inds]
            l_pred_dict[wl] = y_pred_[inds]
            l_pred_nnb_dict[wl] = y_pred_nnb_[inds]
        return mape_dict, mape_nnb_dict, l_real_dict, l_pred_dict, l_pred_nnb_dict

    def get_MAPE_dict(self, obs_num, is_te=True):
        nnr_data = self.nnr_data_dict[obs_num]
        nnr = self.nnr_dict[obs_num]
        _, nnr_val_np, nnr_te_np = nnr_data
        nnr_ = nnr_te_np if is_te else nnr_val_np
        wl_list = self.wl_dict['te'] if is_te else self.wl_dict['val']
        A_, X_, X_nnb_, y_ = [nnr_[var] for var in ['A', 'X', 'X_proxy', 'y']]
        m, m_nnb, l_real, l, l_nnb = self.get_MAPE_helper(nnr, A_, X_, X_nnb_, y_, wl_list)
        # As_, Xs_, Xs_nnb_, ys_ = [nnr_[var] for var in ['A_seen', 'X_seen', 'X_seen_proxy', 'y_seen']]
        # ms, ms_nnb, ls_real, ls, ls_nnb = self.get_MAPE_helper(nnr, As_, Xs_, Xs_nnb_, ys_, wl_list)
        return {
            "mape-unseen": m,
            "mape-unseen-nnb": m_nnb,
            "lat-unseen-real": l_real,
            "lat-unseen": l,
            "lat-unseen-nnb": l_nnb,
            # "mape-seen": ms,
            # "mape-seen-nnb": ms_nnb,
            # "lat-seen-real": ls_real,
            # "lat-seen": ls,
            # "lat-seen-nnb": ls_nnb,
        }

    def get_calibrate_MAPE_dict(self, obs_num, is_te=True, ori_stats=None):
        # only consider nnb
        wl_list = self.wl_dict['te'] if is_te else self.wl_dict['val']
        ori_stats = self.get_MAPE_dict(obs_num, is_te=is_te) if ori_stats is None else ori_stats
        l_real, l, l_nnb = [ori_stats[var] for var in ["lat-unseen-real", "lat-unseen", "lat-unseen-nnb"]]
        ls_real, ls, ls_nnb = [ori_stats[var] for var in ["lat-seen-real", "lat-seen", "lat-seen-nnb"]]
        mape_cal_dict, mape_nnb_cal_dict = {}, {}
        l_cal_dict, l_nnb_cal_dict = {}, {}
        for wl in wl_list:
            l_scale = np.mean(ls[wl] / ls_real[wl])
            l_cal = l[wl] / l_scale
            l_cal_dict[wl] = l_cal
            mape_cal_dict[wl] = np.abs(l_cal - l_real[wl]) / l_real[wl]

            l_nnb_scale = np.mean(ls_nnb[wl] / ls_real[wl])
            l_nnb_cal = l_nnb[wl] / l_nnb_scale
            l_nnb_cal_dict[wl] = l_nnb_cal
            mape_nnb_cal_dict[wl] = np.abs(l_nnb_cal - l_real[wl]) / l_real[wl]
        car_stats = {
            "mape-unseen-cal": mape_cal_dict,
            "mape-unseen-nnb-cal": mape_nnb_cal_dict,
            "lat-unseen-cal": l_cal_dict,
            "lat-unseen-nnb-cal": l_nnb_cal_dict
        }
        return {**ori_stats, **car_stats}

    def plot_ae_loss(self, loss_tr, loss_val, ae_sign):
        FS = 16
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
        iters = np.array(range(loss_tr.shape[0]))
        for i in range(4):
            axes[i].plot(iters, loss_tr[:, i], '-', label=f'loss_{i+1}_tr')
            axes[i].plot(iters, loss_val[:, i], '-', label=f'loss_{i+1}_val')
            axes[i].legend(fontsize=FS)
            axes[i].set_xlabel('iterations * 10', fontsize=FS)
            axes[i].set_ylabel('loss value', fontsize=FS)
        fig.suptitle(f'{ae_sign}', fontsize=FS)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.savefig(f'{self.ckp_path}/{ae_sign}/ae_loss.pdf')

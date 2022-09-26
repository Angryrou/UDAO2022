import torch
import numpy as np
import utils as ut
import model.architecture.utils as utils

class AENetNNRNetAdaptor:
    def __init__(self, ae):
        self.dtype = ut.DTYPE
        self.device = ut.DEVICE
        np.random.seed(ut.SEED)
        torch.manual_seed(ut.SEED)
        self.ae = ae
        self.aid2wle = {}
        self.aid2temp = {}

    def get_wle_tr(self, wl_list, data_tr):
        wl2wle_tr = {}
        for wl in wl_list:
            O_ = utils.get_tensor(data_tr[wl]['obses'])
            with torch.no_grad():
                _, _, W_ = self.ae.model.forward(O_)
            wl2wle_tr[wl] = torch.mean(W_, dim=0).detach().cpu().numpy()
        return wl2wle_tr

    def get_wle_te(self, wl_list, data_te, mss):
        wl2wle_te = {}
        for wl in wl_list:
            O_ = utils.get_tensor(data_te[wl]['seen']['obses'])
            with torch.no_grad():
                _, _, W_ = self.ae.model.forward(O_)
            wl2wle_te[wl] = {f'seen_{i}': torch.mean(W_[:i], dim=0).detach().cpu().numpy() for i in range(1, mss+1)}
        return wl2wle_te

    def get_wle(self, data_all, wl_lists, mss):
        # mss is the number of observed confs
        data_tr, data_val, data_te = data_all
        wl_list_tr, wl_list_val, wl_list_te = wl_lists
        wl2wle_tr = self.get_wle_tr(wl_list=wl_list_tr, data_tr=data_tr)

        unseen_wl_list_val = list(set(wl_list_val) - set(wl_list_tr))
        wl2wle_val = self.get_wle_te(wl_list=unseen_wl_list_val, data_te=data_val, mss=mss)

        unseen_wl_list_te = list(set(wl_list_te) - set(wl_list_tr))
        wl2wle_te = self.get_wle_te(wl_list=unseen_wl_list_te, data_te=data_te, mss=mss)

        wl2wle = {**wl2wle_tr, **wl2wle_val, **wl2wle_te}
        return wl2wle

    def get_nnb(self, wle_te, wle_tr, wl_list_tr):
        nearest_ind = np.linalg.norm(wle_te - wle_tr, axis=1).argmin()
        return wl_list_tr[nearest_ind]

    def get_nnb_dict(self, wl2wle, wl_lists, mss):
        nnb_dict = {}
        nnb_dict_verbose = {}
        wl_list_tr, wl_list_val, wl_list_te = wl_lists
        wle_tr = np.concatenate([wl2wle[wl].reshape(1, -1) for wl in wl_list_tr])
        # found the nearest neighbor in training set (if the wl has appeared in the tr set, the nearest one is itself)
        for twl in set(wl_list_val + wl_list_te):
            if twl in wl_list_tr:
                continue
            nnb_wls = [self.get_nnb(wl2wle[twl][f'seen_{s+1}'], wle_tr, wl_list_tr) for s in range(mss)]
            nnb_dict[twl] = nnb_wls
            nnb_dict_verbose[twl] = {f'seen_{s + 1}': nnb_wls[s] for s in range(mss)}
        return nnb_dict, nnb_dict_verbose


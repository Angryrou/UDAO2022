import torch
import utils as ut


#### WL_DIST ####
def get_finer_wl_dist(W, temp):
    W_num, W_dim = W.shape
    mask = temp.reshape(1, -1) == temp.reshape(-1, 1)
    W_dist_maxtrix = torch.norm(W.reshape(1, -1, W_dim) - W.reshape(-1, 1, W_dim), dim=2) ** 2

    max_dist_in_temp = torch.rand(W_num, dtype=ut.DTYPE, device=ut.DEVICE)
    mean_dist_in_temp = torch.rand(W_num, dtype=ut.DTYPE, device=ut.DEVICE)
    min_dist_out_temp = torch.rand(W_num, dtype=ut.DTYPE, device=ut.DEVICE)
    mean_dist_out_temp = torch.rand(W_num, dtype=ut.DTYPE, device=ut.DEVICE)

    for idx, W_i in enumerate(W_dist_maxtrix):
        mask_i = mask[idx]
        max_dist_in_temp[idx] = W_i[mask_i].max()
        mean_dist_in_temp[idx] = W_i[mask_i].mean()
        min_dist_out_temp[idx] = W_i[~mask_i].min()
        mean_dist_out_temp[idx] = W_i[~mask_i].mean()

    return [max_dist_in_temp, mean_dist_in_temp, min_dist_out_temp, mean_dist_out_temp]


#### Loss function ####
def get_loss_l2_list(O, O_pred):
    O_l2 = torch.norm(O - O_pred, dim=1) ** 2
    return O_l2

def get_l2_loss(input, target, mode="sum"):
    l2_list = get_loss_l2_list(target, input)
    if mode == "sum":
        loss = torch.sum(l2_list)
    elif mode == "mean":
        loss = torch.mean(l2_list)
    else:
        raise Exception('mode should be sum or mean')
    return loss

def get_loss_self_dist(A, W):
    # used in encoder
    loss = None
    A_list = A.unique()
    for a in A_list:
        a_ind = torch.where(A == a)[0]
        w_a = W[a_ind]
        w_a_dist_matrix = torch.norm(w_a.unsqueeze(0) - w_a.unsqueeze(1), dim=2) ** 2
        w_a_dist_triu = w_a_dist_matrix.triu()
        if loss is None:
            loss = w_a_dist_triu.sum()
        else:
            loss += w_a_dist_triu.sum()
    return loss

def get_loss_tri(W, temp, margin=1):
    max_dist_in_temp, _, min_dist_out_temp, _ = get_finer_wl_dist(W, temp)
    loss = torch.sum(torch.clamp(max_dist_in_temp - min_dist_out_temp + margin, min=0))
    return loss

def get_tensor(X, dtype=None, device=None, requires_grad=False):
    dtype = ut.DTYPE if dtype is None else dtype
    device = ut.DEVICE if device is None else device
    return torch.tensor(X, dtype=dtype, device=device, requires_grad=requires_grad)

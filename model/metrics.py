# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 05/01/2023

import torch as th
import torch.nn as nn
import torch.nn.functional as F


def get_loss(y, y_hat, loss_type):
    if loss_type == "wmape":
        loss = nn.L1Loss(reduction="sum")(y_hat, y) / y.sum()
    elif loss_type == "msle":
        loss = nn.MSELoss()(th.log(y_hat + 1e-3), th.log(y + 1e-3))
    elif loss_type == "mae":
        loss = nn.L1Loss(reduction="sum")(y_hat, y)
    elif loss_type == "mape":
        loss = th.mean(th.abs(y_hat - y) / (y + 1e-3))
    elif loss_type == "mape+wmape":
        loss = nn.L1Loss(reduction="sum")(y_hat, y) / y.sum() + th.mean(th.abs(y_hat - y) / (y + 1e-3))
    elif loss_type == "mse":
        loss = nn.MSELoss()(y_hat, y)
    elif loss_type == "nll":
        loss = F.nll_loss(y_hat, y)
    else:
        raise Exception(f"loss_type {loss_type} not supported")
    return loss
# Copyright (c) 2020 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: TODO
#
# Created at 28/09/2022
import pickle
import torch as th
import numpy as np

DEFAULT_DEVICE = th.device("cpu")
DEFAULT_DTYPE = th.float32

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def get_bounded(k, lower=0.0, upper=1.0):
    k = np.maximum(k, lower)
    k = np.minimum(k, upper)
    return k

def _get_tensor(x, dtype=None, device=None, requires_grad=False):
    dtype = DEFAULT_DTYPE
    device = DEFAULT_DEVICE if device is None else device

    return th.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
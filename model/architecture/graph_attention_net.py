# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 28/02/2023

import os

from model.architecture.mlp_readout_layer import MLPReadout
os.environ['DGLBACKEND'] = 'pytorch'
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GATv2Conv


class GATv2(nn.Module):

    def __init__(self, net_params):
        super().__init__()

    def forward(self, g, inst_feat):
        ...
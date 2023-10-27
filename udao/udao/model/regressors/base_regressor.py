from abc import ABC, abstractmethod

import torch as th
from torch import nn


class BaseRegressor(nn.Module, ABC):
    @abstractmethod
    def forward(self, embedding: th.Tensor, inst_feat: th.Tensor) -> th.Tensor:
        pass

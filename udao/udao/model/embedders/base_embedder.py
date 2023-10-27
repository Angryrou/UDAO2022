from abc import ABC

from torch import nn


class BaseEmbedder(nn.Module, ABC):
    ...

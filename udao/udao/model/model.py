import torch as th
from torch import nn

from ..utils.interfaces import BaseUdaoInput
from .embedders.base_embedder import BaseEmbedder
from .regressors.base_regressor import BaseRegressor


class UdaoModel(nn.Module):
    def __init__(self, regressor: BaseRegressor, embedder: BaseEmbedder) -> None:
        super().__init__()
        self.regressor = regressor
        self.embedder = embedder

    def forward(self, input_data: BaseUdaoInput) -> th.Tensor:
        embedding = self.embedder(input_data.embedding_input)
        inst_feat = input_data.feature_input
        return self.regressor(embedding, inst_feat)

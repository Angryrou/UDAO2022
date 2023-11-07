from typing import Dict, Type

import torch as th
from torch import nn

from ..utils.interfaces import UdaoInput, UdaoInputShape
from .embedders.base_embedder import BaseEmbedder
from .regressors.base_regressor import BaseRegressor


class UdaoModel(nn.Module):
    @classmethod
    def from_config(
        cls,
        regressor_cls: Type[BaseRegressor],
        embedder_cls: Type[BaseEmbedder],
        iterator_shape: UdaoInputShape,
        regressor_params: Dict,
        embedder_params: Dict,
    ) -> "UdaoModel":
        embedder = embedder_cls.from_iterator_shape(iterator_shape, **embedder_params)
        regressor = regressor_cls(
            regressor_cls.Params(
                input_embedding_dim=embedder.embedding_size,
                input_features_dim=iterator_shape.feature_input_shape,
                output_dim=iterator_shape.output_shape,
                **regressor_params
            ),
        )
        return cls(regressor, embedder)
        pass

    def __init__(self, regressor: BaseRegressor, embedder: BaseEmbedder) -> None:
        super().__init__()
        self.regressor = regressor
        self.embedder = embedder

    def forward(self, input_data: UdaoInput) -> th.Tensor:
        embedding = self.embedder(input_data.embedding_input)
        inst_feat = input_data.feature_input
        return self.regressor(embedding, inst_feat)

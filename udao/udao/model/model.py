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
        embedder_cls: Type[BaseEmbedder],
        regressor_cls: Type[BaseRegressor],
        iterator_shape: UdaoInputShape,
        regressor_params: Dict,
        embedder_params: Dict,
    ) -> "UdaoModel":
        embedder = embedder_cls.from_iterator_shape(iterator_shape, **embedder_params)
        regressor = regressor_cls(
            regressor_cls.Params(
                input_embedding_dim=embedder.embedding_size,
                input_features_dim=len(iterator_shape.feature_input_names),
                output_dim=iterator_shape.output_shape,
                **regressor_params
            ),
        )
        return cls(embedder, regressor)
        pass

    def __init__(self, embedder: BaseEmbedder, regressor: BaseRegressor) -> None:
        super().__init__()
        self.embedder = embedder
        self.regressor = regressor

    def forward(self, input_data: UdaoInput) -> th.Tensor:
        embedding = self.embedder(input_data.embedding_input)
        inst_feat = input_data.feature_input
        return self.regressor(embedding, inst_feat)

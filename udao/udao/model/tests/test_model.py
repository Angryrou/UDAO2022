from typing import Any

import torch as th
from attr import dataclass

from ...utils.interfaces import UdaoInput, UdaoInputShape
from ..embedders import BaseEmbedder
from ..model import UdaoModel
from ..regressors import BaseRegressor


class DummyEmbedder(BaseEmbedder):
    @dataclass
    class Params(BaseEmbedder.Params):
        output_size: int

    @classmethod
    def from_iterator_shape(
        cls, iterator_shape: UdaoInputShape, **kwargs: Any
    ) -> "DummyEmbedder":
        return cls(cls.Params(**kwargs))

    def forward(self, input: th.Tensor) -> th.Tensor:
        return input


class DummyRegressor(BaseRegressor):
    def forward(self, embedding: th.Tensor, inst_feat: th.Tensor) -> th.Tensor:
        return embedding + inst_feat


class TestUdaoModel:
    def test_from_config(self) -> None:
        iterator_shape = UdaoInputShape(
            embedding_input_shape=1,
            feature_input_names=["a"],
            output_shape=1,
        )
        model = UdaoModel.from_config(
            regressor_cls=DummyRegressor,
            embedder_cls=DummyEmbedder,
            iterator_shape=iterator_shape,
            regressor_params={},
            embedder_params={"output_size": 1},
        )
        assert model.regressor.input_dim == 2
        assert model.embedder.embedding_size == 1
        assert model.regressor.output_dim == 1

    def test_forward(self) -> None:
        iterator_shape = UdaoInputShape(
            embedding_input_shape=1,
            feature_input_names=["a"],
            output_shape=1,
        )
        model = UdaoModel.from_config(
            regressor_cls=DummyRegressor,
            embedder_cls=DummyEmbedder,
            iterator_shape=iterator_shape,
            regressor_params={},
            embedder_params={"output_size": 1},
        )
        embedding = th.tensor([1.0])
        inst_feat = th.tensor([2.0])
        out = model.forward(
            UdaoInput(embedding_input=embedding, feature_input=inst_feat)
        )
        assert out == th.tensor([3.0])

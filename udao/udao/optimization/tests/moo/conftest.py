import pytest
import torch as th
from torch import nn

from ....data.containers.tabular_container import TabularContainer
from ....data.extractors.tabular_extractor import TabularFeatureExtractor
from ....data.handler.data_processor import DataProcessor
from ....data.preprocessors.base_preprocessor import StaticFeaturePreprocessor
from ....data.tests.iterators.dummy_udao_iterator import DummyUdaoIterator
from ....utils.interfaces import UdaoInput


class ObjModel1(nn.Module):
    def forward(self, x: UdaoInput) -> th.Tensor:
        return th.reshape(x.feature_input[:, 0] ** 2, (-1, 1))


class ObjModel2(nn.Module):
    def forward(self, x: UdaoInput) -> th.Tensor:
        return th.reshape(x.feature_input[:, 1] ** 2, (-1, 1))


class ComplexObj1(nn.Module):
    def forward(self, x: UdaoInput) -> th.Tensor:
        return th.reshape(
            x.feature_input[:, 0] ** 2 - x.feature_input[:, 1] ** 2, (-1, 1)
        )


class ComplexObj2(nn.Module):
    def forward(self, x: UdaoInput) -> th.Tensor:
        return th.reshape(
            x.feature_input[:, 0] ** 2 + x.feature_input[:, 1] ** 2, (-1, 1)
        )


class TabularFeaturePreprocessor(StaticFeaturePreprocessor):
    def preprocess(self, tabular_feature: TabularContainer) -> TabularContainer:
        tabular_feature.data.loc[:, "v1"] = tabular_feature.data["v1"] / 1
        tabular_feature.data.loc[:, "v2"] = (tabular_feature.data["v2"] - 1) / 6
        return tabular_feature

    def inverse_transform(self, tabular_feature: TabularContainer) -> TabularContainer:
        tabular_feature.data.loc[:, "v1"] = tabular_feature.data["v1"] * 1
        tabular_feature.data.loc[:, "v2"] = tabular_feature.data["v2"] * 6 + 1
        return tabular_feature


@pytest.fixture()
def data_processor() -> DataProcessor:
    return DataProcessor(
        iterator_cls=DummyUdaoIterator,
        feature_extractors={
            "embedding_features": TabularFeatureExtractor(columns=["embedding_input"]),
            "tabular_features": TabularFeatureExtractor(
                columns=["v1", "v2"],
            ),
            "objectives": TabularFeatureExtractor(columns=["objective_input"]),
        },
        feature_preprocessors={"tabular_features": [TabularFeaturePreprocessor()]},
    )

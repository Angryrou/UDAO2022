from typing import Sequence, Tuple

import torch as th

from ....data.containers.tabular_container import TabularContainer
from ....data.iterators.base_iterator import FeatureIterator
from ....utils.interfaces import (
    FeatureInput,
    FeatureInputShape,
    UdaoInput,
    UdaoInputShape,
)


class DummyFeatureIterator(FeatureIterator[FeatureInput, FeatureInputShape]):
    def __init__(
        self,
        keys: Sequence[str],
        tabular_features: TabularContainer,
        objectives: TabularContainer,
    ) -> None:
        super().__init__(keys, tabular_features=tabular_features, objectives=objectives)

    def _getitem(self, idx: int) -> Tuple[FeatureInput, th.Tensor]:
        key = self.keys[idx]
        return (
            FeatureInput(
                th.tensor(self.tabular_features.get(key), dtype=self.tensors_dtype)
            ),
            th.tensor(self.objectives.get(key), dtype=self.tensors_dtype),
        )

    @property
    def shape(self) -> FeatureInputShape:
        return FeatureInputShape(
            feature_input_names=list(self.tabular_features.data.columns),
            output_names=list(self.objectives.data.columns),
        )

    @staticmethod
    def collate(
        items: Sequence[Tuple[FeatureInput, th.Tensor]]
    ) -> Tuple[FeatureInput, th.Tensor]:
        features = FeatureInput(th.vstack([item[0].feature_input for item in items]))
        objectives = th.vstack([item[1] for item in items])
        return features, objectives


class DummyUdaoIterator(FeatureIterator[UdaoInput, UdaoInputShape]):
    def __init__(
        self,
        keys: Sequence[str],
        tabular_features: TabularContainer,
        objectives: TabularContainer,
        embedding_features: TabularContainer,
    ) -> None:
        self.embedding_features = embedding_features
        super().__init__(keys, tabular_features=tabular_features, objectives=objectives)

    def _getitem(self, idx: int) -> Tuple[UdaoInput, th.Tensor]:
        key = self.keys[idx]
        return (
            UdaoInput(
                embedding_input=th.tensor(
                    self.embedding_features.get(key), dtype=self.tensors_dtype
                ),
                feature_input=th.tensor(
                    self.tabular_features.get(key), dtype=self.tensors_dtype
                ),
            ),
            th.tensor(self.objectives.get(key), dtype=self.tensors_dtype),
        )

    @property
    def shape(self) -> UdaoInputShape:
        return UdaoInputShape(
            embedding_input_shape=self.embedding_features.data.shape[1],
            feature_input_names=list(self.tabular_features.data.columns),
            output_names=list(self.objectives.data.columns),
        )

    @staticmethod
    def collate(
        items: Sequence[Tuple[UdaoInput, th.Tensor]]
    ) -> Tuple[UdaoInput, th.Tensor]:
        embedding_input = th.vstack([item[0].embedding_input for item in items])
        features = th.vstack([item[0].feature_input for item in items])
        objectives = th.vstack([item[1] for item in items])
        return (
            UdaoInput(embedding_input=embedding_input, feature_input=features),
            objectives,
        )

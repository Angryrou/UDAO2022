from typing import Sequence, Tuple

import torch as th

from ....data.containers.tabular_container import TabularContainer
from ....data.iterators.base_iterator import UdaoIterator
from ....utils.interfaces import UdaoInput, UdaoInputShape


class DummyUdaoIterator(UdaoIterator):
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
            output_shape=self.objectives.data.shape[1],
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

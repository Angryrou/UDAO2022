from typing import Sequence, Tuple

import torch as th

from ....data.containers.tabular_container import TabularContainer
from ....data.iterators.base_iterator import UdaoIterator
from ....utils.interfaces import UdaoInput, UdaoInputShape


class DummyUdaoIterator(UdaoIterator):
    def __init__(
        self,
        keys: Sequence[str],
        feature: TabularContainer,
        embedding: TabularContainer,
        objective: TabularContainer,
    ) -> None:
        super().__init__(keys)

        self.feature_container = feature
        self.embedding_container = embedding
        self.objective_container = objective

    def __getitem__(self, idx: int) -> Tuple[UdaoInput, th.Tensor]:
        key = self.keys[idx]
        return (
            UdaoInput(
                embedding_input=th.tensor(
                    self.embedding_container.get(key), dtype=self.tensors_dtype
                ),
                feature_input=th.tensor(
                    self.feature_container.get(key), dtype=self.tensors_dtype
                ),
            ),
            th.tensor(self.objective_container.get(key), dtype=self.tensors_dtype),
        )

    def get_iterator_shape(self) -> UdaoInputShape:
        return UdaoInputShape(
            embedding_input_shape=self.embedding_container.data.shape[1],
            feature_input_names=list(self.feature_container.data.columns),
            output_shape=self.objective_container.data.shape[1],
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

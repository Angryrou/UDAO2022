from typing import Any, Dict, Sequence

import torch as th

from ..containers import TabularContainer
from .base_iterator import BaseIterator


class TabularIterator(BaseIterator[th.Tensor, Dict[str, Any]]):
    """Iterator on tabular data.

    Parameters
    ----------
    keys : Sequence[str]
        Keys of the dataset, used for accessing all features
    table : TabularContainer
        Container for the tabular data
    """

    def __init__(
        self,
        keys: Sequence[str],
        tabular_feature: TabularContainer,
    ):
        super().__init__(keys)
        self.tabular_feature = tabular_feature

    def __len__(self) -> int:
        return len(self.keys)

    def _getitem(self, idx: int) -> th.Tensor:
        key = self.keys[idx]
        return th.tensor(self.tabular_feature.get(key), dtype=self.tensors_dtype)

    def get_iterator_shape(self) -> Any:
        sample_input = self._get_sample()
        return {"input_shape": sample_input.shape}

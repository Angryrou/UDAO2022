from typing import Any, Sequence

from ..containers import TabularContainer
from .base_iterator import BaseDatasetIterator


class TabularIterator(BaseDatasetIterator[TabularContainer]):
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
        self.keys = keys
        self.tabular_feature = tabular_feature

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Any:
        key = self.keys[idx]
        return self.tabular_feature.get(key)

    def get_iterator_shape(self) -> Any:
        """Returns the shape of the iterator."""
        return self.tabular_feature.data.shape[-1]

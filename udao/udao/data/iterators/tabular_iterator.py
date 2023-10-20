from typing import Any, Sequence

from udao.data.containers import DataFrameContainer

from .base_iterator import BaseDatasetIterator


class TabularIterator(BaseDatasetIterator):
    def __init__(
        self,
        keys: Sequence[str],
        dataframe_container: DataFrameContainer,
    ):
        self.keys = keys
        self.dataframe_container = dataframe_container

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Any:
        key = self.keys[idx]
        return self.dataframe_container.get(key)

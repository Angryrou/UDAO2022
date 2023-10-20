from typing import Any, Sequence

import pandas as pd

from .base_iterator import BaseDatasetIterator


class TabularIterator(BaseDatasetIterator):
    def __init__(
        self,
        keys: Sequence[str],
        feature_frame: pd.DataFrame,
    ):
        self.keys = keys
        self.feature_frame = feature_frame

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Any:
        key = self.keys[idx]
        return self.feature_frame.loc[key].values

from typing import Any, Callable, List

import pandas as pd

from ..containers import TabularContainer
from .base_extractors import StaticFeatureExtractor


class TabularFeatureExtractor(StaticFeatureExtractor[TabularContainer]):
    def __init__(self, feature_func: Callable, **kwargs: Any) -> None:
        self.feature_func = feature_func
        self.func_kwargs = kwargs

    def extract_features(self, df: pd.DataFrame) -> TabularContainer:
        return TabularContainer(self.feature_func(df, **self.func_kwargs))


## Tabular utils functions ##
def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """select specific columns from a dataframe"""
    return df[columns]

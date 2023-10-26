from typing import Callable

import pandas as pd

from ..containers import TabularContainer
from .base_extractors import StaticFeatureExtractor


class TabularFeatureExtractor(StaticFeatureExtractor[TabularContainer]):
    def __init__(self, feature_func: Callable) -> None:
        self.feature_func = feature_func

    def extract_features(self, df: pd.DataFrame) -> TabularContainer:
        return TabularContainer(self.feature_func(df))

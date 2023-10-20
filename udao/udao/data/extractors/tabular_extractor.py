from typing import Callable

import pandas as pd
from udao.data.containers.query_embedding_container import DataFrameContainer

from .base_extractors import StaticFeatureExtractor


class TabularFeatureExtractor(StaticFeatureExtractor):
    def __init__(self, feature_func: Callable) -> None:
        self.feature_func = feature_func

    def extract_features(self, df: pd.DataFrame) -> DataFrameContainer:
        feature_series = df.apply(self.feature_func, axis=1)
        feature_df = feature_series.to_frame(name="feature")
        return DataFrameContainer(feature_df)

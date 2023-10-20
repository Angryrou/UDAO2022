from typing import Any, Callable, Dict

import pandas as pd

from .base_extractors import StaticFeatureExtractor


class TabularFeatureExtractor(StaticFeatureExtractor):
    def __init__(self, feature_func: Callable) -> None:
        self.feature_func = feature_func

    def extract_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        feature_series = df.apply(self.feature_func, axis=1)
        feature_df = feature_series.to_frame(name="feature")
        return {"feature_frame": feature_df}

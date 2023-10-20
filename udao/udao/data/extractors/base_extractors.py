from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Union

import pandas as pd
from udao.data.utils.utils import DatasetType


class TrainedFeatureExtractor(ABC):
    trained: bool = True

    def __init__(self) -> None:
        pass

    @abstractmethod
    def extract_features(self, df: pd.DataFrame, split: DatasetType) -> Dict[str, Any]:
        pass


class StaticFeatureExtractor(ABC):
    trained: bool = False

    def __init__(self) -> None:
        pass

    @abstractmethod
    def extract_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        pass


FeatureExtractorType = Union[
    Type[TrainedFeatureExtractor], Type[StaticFeatureExtractor]
]

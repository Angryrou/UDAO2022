from abc import ABC, abstractmethod
from typing import Type, Union

import pandas as pd

from ...data.containers import BaseContainer
from ...data.utils.utils import DatasetType


class TrainedFeatureExtractor(ABC):
    trained: bool = True

    def __init__(self) -> None:
        pass

    @abstractmethod
    def extract_features(self, df: pd.DataFrame, split: DatasetType) -> BaseContainer:
        pass


class StaticFeatureExtractor(ABC):
    trained: bool = False

    def __init__(self) -> None:
        pass

    @abstractmethod
    def extract_features(self, df: pd.DataFrame) -> BaseContainer:
        pass


FeatureExtractorType = Union[
    Type[TrainedFeatureExtractor], Type[StaticFeatureExtractor]
]

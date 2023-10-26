from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar, Union

from ..containers.base_container import BaseContainer
from ..utils.utils import DatasetType

T = TypeVar("T", bound=BaseContainer)


class TrainedFeaturePreprocessor(ABC, Generic[T]):
    """Base class for feature processors that require training."""

    trained: bool = True

    def __init__(self) -> None:
        pass

    @abstractmethod
    def preprocess(self, container: T, split: DatasetType) -> T:
        pass


class StaticFeaturePreprocessor(ABC, Generic[T]):
    """Base class for feature processors that do not require training."""

    trained: bool = False

    def __init__(self) -> None:
        pass

    @abstractmethod
    def preprocess(self, container: T) -> T:
        pass


FeaturePreprocessorType = Union[
    Type[TrainedFeaturePreprocessor], Type[StaticFeaturePreprocessor]
]

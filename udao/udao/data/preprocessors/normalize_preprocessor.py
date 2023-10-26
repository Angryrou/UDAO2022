from typing import Any, Protocol, TypeVar

import pandas as pd

from ..containers.base_container import BaseContainer
from ..utils.utils import DatasetType
from .base_preprocessor import TrainedFeaturePreprocessor


# Define a protocol for objects that have fit and transform methods
class FitTransformProtocol(Protocol):
    def fit(self, X: Any, y: Any = None) -> "FitTransformProtocol":
        ...

    def transform(self, X: Any) -> Any:
        ...


T = TypeVar("T", bound=BaseContainer)


class NormalizePreprocessor(TrainedFeaturePreprocessor[T]):
    """Normalize the data using a normalizer that
    implements the fit and transform methods, e.g. MinMaxScaler.

    Parameters
    ----------
    normalizer : FitTransformProtocol
        A normalizer that implements the fit and transform methods
        (e.g. sklearn.MinMaxScaler)
    df_key : str
        The key of the dataframe in the container.
    """

    def __init__(
        self, normalizer: FitTransformProtocol, data_key: str = "data"
    ) -> None:
        self.normalizer = normalizer
        self.df_key = data_key

    def preprocess(self, container: T, split: DatasetType) -> T:
        """Normalize the data in the container.

        Parameters
        ----------
        container : T
            Child of BaseContainer
        split : DatasetType
            Train or other (val, test).
        Returns
        -------
        T
            Child of BaseContainer with the normalized data.
        """
        container = container.copy()
        df: pd.DataFrame = container.__getattribute__(self.df_key)
        if split == "train":
            self.normalizer.fit(df)
        # assumes the normalizer returns an array.
        transformed_data = self.normalizer.transform(df)
        container.__setattr__(
            self.df_key,
            pd.DataFrame(transformed_data, index=df.index, columns=df.columns),
        )
        return container

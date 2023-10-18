from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd
from attr import dataclass
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils.logging import logger
from .utils.utils import (
    DatasetType,
    FeatureExtractorType,
    StaticFeatureExtractor,
    TrainedFeatureExtractor,
    train_test_val_split_on_column,
)

tqdm.pandas()


class BaseDatasetIterator(Dataset):
    """Dummy Base class for all dataset iterators.
    Inherits from torch.utils.data.Dataset.
    Defined to allow for type hinting and having an __init__ method.
    """

    @abstractmethod
    def __init__(self, keys, *args, **kwargs):  # type: ignore
        pass


class DataHandler:
    """
    DataHandler class to handle data loading, splitting, feature extraction and
    dataset iterator creation.
    """

    def __init__(
        self,
        path: str,
        index_column: str,
        feature_extractors: List[Tuple[FeatureExtractorType, Any]],
        target_Iterator: Type[BaseDatasetIterator],
        dryrun: bool = False,
    ) -> None:
        self.dryrun = dryrun
        self.index_column = index_column
        self.full_df = self._load_data(path)
        self.index_splits: Dict[DatasetType, List[str]] = {}
        self.features: Dict[DatasetType, Dict] = defaultdict(dict)
        self.feature_extractors = feature_extractors
        self.target_Iterator = target_Iterator

    def _load_data(self, path: str, index_column: str = "id") -> pd.DataFrame:
        full_df = pd.read_csv(path)
        if self.dryrun:
            full_df = full_df.sample(5000)
        full_df.set_index(self.index_column, inplace=True, drop=False)

        return full_df

    def split_data(
        self,
        stratify_on: Optional[str] = None,
        val_frac: float = 0.2,
        test_frac: float = 0.1,
    ) -> "DataHandler":
        if self.full_df is None:
            raise ValueError("Data not loaded yet.")
        if stratify_on is None:
            stratify_on = self.index_column
        df_splits = train_test_val_split_on_column(
            self.full_df, stratify_on, val_frac=val_frac, test_frac=test_frac
        )
        self.index_splits = {
            split: df.index.to_list() for split, df in df_splits.items()
        }
        return self

    def get_iterators(self) -> Dict[DatasetType, BaseDatasetIterator]:
        """Get iterators for the different splits of the data.

        Returns
        -------
        Dict[DatasetType, BaseDatasetIterator]
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        if self.full_df is None:
            raise ValueError("Data not loaded yet.")
        for Extractor, args in self.feature_extractors:
            if issubclass(Extractor, StaticFeatureExtractor):
                logger.info(
                    f"Extracting features for static extractor {Extractor.__name__}"
                )

                for split, keys in self.index_splits.items():
                    self.features[split] = {
                        **self.features[split],
                        **Extractor(*args).extract_features(self.full_df.loc[keys]),
                    }

            elif issubclass(Extractor, TrainedFeatureExtractor):
                logger.info(
                    f"Extracting features for trained extractor {Extractor.__name__}"
                )
                for split, keys in self.index_splits.items():
                    self.features[split] = {
                        **self.features[split],
                        **Extractor(*args).extract_features(
                            self.full_df.loc[keys], split=split
                        ),
                    }
            else:
                raise ValueError(
                    f"Extractor {Extractor.__name__} not supported. Should implement"
                    " either StaticFeatureExtractor or TrainedFeatureExtractor."
                )

        return {
            split: self.target_Iterator(
                self.index_splits[split], **self.features[split]
            )
            for split in self.index_splits
        }


@dataclass
class DataHandlerParams:
    index_column: str
    feature_extractors: List[Tuple[FeatureExtractorType, Any]]
    target_Iterator: Type[BaseDatasetIterator]
    path: str
    stratify_on: Optional[str]
    val_frac: float = 0.2
    test_frac: float = 0.1
    dryrun: bool = False

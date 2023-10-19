from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd
from attr import dataclass
from torch.utils.data import Dataset
from udao.data.utils.utils import (
    DatasetType,
    FeatureExtractorType,
    StaticFeatureExtractor,
    TrainedFeatureExtractor,
    train_test_val_split_on_column,
)
from udao.utils.logging import logger


class BaseDatasetIterator(Dataset):
    """Dummy Base class for all dataset iterators.
    Inherits from torch.utils.data.Dataset.
    Defined to allow for type hinting and having an __init__ method.
    """

    @abstractmethod
    def __init__(self, keys, *args, **kwargs):  # type: ignore
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


@dataclass
class DataHandlerParams:
    """DataHandlerParams class to store the parameters of the DataHandler.

    Parameters
    ----------
    index_column : str
        Column that should be used as index (unique identifier)
    feature_extractors : List[Tuple[FeatureExtractorType, Any]]
        List of tuples of the form (Extractor, args) where Extractor
        implements FeatureExtractor and args are the arguments to be passed
        at initialization.
        If Extractor is a StaticFeatureExtractor, the features are extracted
        independently of the split.
        If Extractor is a TrainedFeatureExtractor, the extractor is first fitted
        on the train split and then applied to the other splits.
    target_Iterator : Type[BaseDatasetIterator]
        Iterator class to be returned after feature extraction.
        It is assumed that the iterator class takes the keys and the features
        extracted by the feature extractors as arguments.
    dryrun : bool, optional
        Dry run mode for fast computation on a large dataset (sampling of a
        small portion), by default False
    stratify_on : Optional[str], optional
        Column on which to stratify the split
        (keeping proportions for each split)
        If None, no stratification is performed
    val_frac : float, optional
        Fraction allotted to the validation set, by default 0.2
    test_frac : float, optional
            Fraction allotted to the test set, by default 0.1
    random_state : Optional[int], optional
        Random state for reproducibility, by default None
    """

    index_column: str
    feature_extractors: List[Tuple[FeatureExtractorType, Any]]
    Iterator: Type[BaseDatasetIterator]
    stratify_on: Optional[str] = None
    val_frac: float = 0.2
    test_frac: float = 0.1
    dryrun: bool = False
    random_state: Optional[int] = None


class DataHandler:
    """
    DataHandler class to handle data loading, splitting, feature extraction and
    dataset iterator creation.
    """

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        params: DataHandlerParams,
    ) -> "DataHandler":
        """Initialize DataHandler from csv.

        Parameters
        ----------
        csv_path : str
            Path to the data file.
        params : DataHandlerParams
        Returns
        -------
        DataHandler
            Initialized DataHandler object.
        """
        return cls(
            pd.read_csv(csv_path),
            params,
        )

    def __init__(
        self,
        data: pd.DataFrame,
        params: DataHandlerParams,
    ) -> None:
        """Initialize DataHandler.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing the data.
        params : DataHandlerParams
            DataHandlerParams object containing the parameters of the DataHandler.
        """
        self.dryrun = params.dryrun
        self.index_column = params.index_column
        self.feature_extractors = params.feature_extractors
        self.Iterator = params.Iterator
        self.stratify_on = params.stratify_on
        self.val_frac = params.val_frac
        self.test_frac = params.test_frac
        self.random_state = params.random_state
        self.full_df = data
        if self.dryrun:
            self.full_df = self.full_df.sample(frac=0.1, random_state=self.random_state)
        self.full_df.set_index(self.index_column, inplace=True, drop=False)

        self.index_splits: Dict[DatasetType, List[str]] = {}
        self.features: Dict[DatasetType, Dict] = defaultdict(dict)

    def split_data(
        self,
    ) -> "DataHandler":
        """Split the data into train, test and validation sets,
        split indices are stored in self.index_splits.

        Returns
        -------
        DataHandler
            set
        """

        df_splits = train_test_val_split_on_column(
            self.full_df,
            self.stratify_on,
            val_frac=self.val_frac,
            test_frac=self.test_frac,
            random_state=self.random_state,
        )
        self.index_splits = {
            split: df.index.to_list() for split, df in df_splits.items()
        }
        return self

    def extract_features(self) -> "DataHandler":
        """Extract features for the different splits of the data.

        Returns
        -------
        DataHandler
            self

        Raises
        ------
        ValueError
            Expects data to be split before extracting features.
        """
        if not self.index_splits:
            raise ValueError("Data not split yet.")
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
        return self

    def get_iterators(self) -> Dict[DatasetType, BaseDatasetIterator]:
        if not self.features:
            logger.warning("No features extracted yet. Extracting features now.")
            self.extract_features()
        return {
            split: self.Iterator(self.index_splits[split], **self.features[split])
            for split in self.index_splits
        }

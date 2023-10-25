from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type

import pandas as pd
from attr import dataclass

from ...data.preprocessors.base_preprocessor import (
    FeaturePreprocessorType,
    StaticFeaturePreprocessor,
    TrainedFeaturePreprocessor,
)
from ...utils.logging import logger
from ..containers import BaseContainer
from ..extractors import (
    FeatureExtractorType,
    StaticFeatureExtractor,
    TrainedFeatureExtractor,
)
from ..iterators import BaseDatasetIterator
from ..utils.utils import DatasetType, train_test_val_split_on_column


@dataclass
class DataHandlerParams:
    index_column: str
    """Column that should be used as index (unique identifier)"""

    feature_extractors: Mapping[str, Tuple[FeatureExtractorType, Any]]
    """Dict that links a feature name to tuples of the form (Extractor, args)
        where Extractor implements FeatureExtractor and args are the arguments
        to be passed at initialization.
        N.B.: Feature names must match the iterator's parameters.

        If Extractor is a StaticFeatureExtractor, the features are extracted
        independently of the split.

        If Extractor is a TrainedFeatureExtractor, the extractor is first fitted
        on the train split and then applied to the other splits."""

    Iterator: Type[BaseDatasetIterator]
    """Iterator class to be returned after feature extraction.
        It is assumed that the iterator class takes the keys and the features
        extracted by the feature extractors as arguments."""

    feature_preprocessors: Optional[
        Mapping[str, List[Tuple[FeaturePreprocessorType, Any]]]
    ] = None
    """Dict that links a feature name to a list of tuples of the form (Processor, args)
        where Processor implements FeatureProcessor and args are the arguments
        to be passed at initialization.
        This allows to apply a series of processors to different features, e.g.
        to normalize the features.
        N.B.: Feature names must match the iterator's parameters.
        If Processor is a StaticFeatureprocessor, the features are processed
        independently of the split.

        If Extractor is a TrainedFeatureProcessor, the processor is first fitted
        on the train split and then applied to the other splits
        (typically for normalization).
        """

    stratify_on: Optional[str] = None
    """Column on which to stratify the split, by default None.
    If None, no stratification is performed."""

    val_frac: float = 0.2
    """Column on which to stratify the split
        (keeping proportions for each split)
        If None, no stratification is performed"""

    test_frac: float = 0.1
    """Fraction allotted to the validation set, by default 0.2"""

    dryrun: bool = False
    """Dry run mode for fast computation on a large dataset (sampling of a
        small portion), by default False"""

    random_state: Optional[int] = None
    """Random state for reproducibility, by default None"""


@dataclass
class FeaturePipeline:
    extractor: Tuple[FeatureExtractorType, Any]
    """Tuple defining the feature extractor and its initialization arguments."""
    preprocessors: Optional[List[Tuple[FeaturePreprocessorType, Any]]] = None
    """List of tuples defining feature preprocessors
    and their initialization arguments."""


def create_data_handler_params(
    iterator_cls: Type[BaseDatasetIterator], *args: str
) -> Callable[..., DataHandlerParams]:
    """
    Creates a DataHandlerParams class dynamically based on
    provided iterator class and additional arguments.

    Parameters
    ----------
    iterator_cls : Type[BaseDatasetIterator]
        Dataset iterator class type.
    args : str
        Additional feature names to be included.

    Returns
    -------
    Type[DataHandlerParams]
        A dynamically generated DataHandlerParams class with arguments
        from the provided iterator class and additional arguments.
    """
    params = iterator_cls.get_parameter_names()
    if args is not None:
        params += args

    def get_data_handler_params(
        index_column: str,
        stratify_on: Optional[str] = None,
        val_frac: float = 0.2,
        test_frac: float = 0.1,
        dryrun: bool = False,
        random_state: Optional[int] = None,
        **kwargs: FeaturePipeline,
    ) -> DataHandlerParams:
        feature_extractors: Dict[str, Tuple[FeatureExtractorType, Any]] = {}
        feature_preprocessors: Dict[str, List[Tuple[FeaturePreprocessorType, Any]]] = {}
        for param in params:
            if param in kwargs:
                feature_extractors[param] = kwargs[param].extractor
                preprocessors = kwargs[param].preprocessors
                if preprocessors is not None:
                    feature_preprocessors[param] = preprocessors
            else:
                raise ValueError(
                    f"Feature pipeline for {param} not specified in kwargs. "
                    f"All iterator features should be provided with an extractor."
                )

        return DataHandlerParams(
            index_column=index_column,
            Iterator=iterator_cls,
            stratify_on=stratify_on,
            val_frac=val_frac,
            test_frac=test_frac,
            dryrun=dryrun,
            random_state=random_state,
            feature_extractors=feature_extractors,
            feature_preprocessors=feature_preprocessors,
        )

    return get_data_handler_params


class DataHandler:
    """
    DataHandler class to handle data loading, splitting, feature extraction and
    dataset iterator creation.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data.
    params : DataHandlerParams
        DataHandlerParams object containing the parameters of the DataHandler.
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
        self.dryrun = params.dryrun
        self.index_column = params.index_column
        self.feature_extractors = params.feature_extractors
        self.feature_processors = params.feature_preprocessors
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
        self.features: Dict[DatasetType, Dict[str, BaseContainer]] = defaultdict(dict)

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
            logger.warning("No data split yet. Splitting data now.")
            self.split_data()
        for name, (Extractor, args) in self.feature_extractors.items():
            if issubclass(Extractor, StaticFeatureExtractor):
                logger.info(
                    f"Extracting features for static extractor {Extractor.__name__}"
                )

                for split, keys in self.index_splits.items():
                    self.features[split][name] = Extractor(*args).extract_features(
                        self.full_df.loc[keys]
                    )

            elif issubclass(Extractor, TrainedFeatureExtractor):
                logger.info(
                    f"Extracting features for trained extractor {Extractor.__name__}"
                )
                extractor = Extractor(*args)
                for split, keys in self.index_splits.items():
                    self.features[split][name] = extractor.extract_features(
                        self.full_df.loc[keys], split=split
                    )

            else:
                raise ValueError(
                    f"Extractor {Extractor.__name__} not supported. Should implement"
                    " either StaticFeatureExtractor or TrainedFeatureExtractor."
                )
        return self

    def process_features(self) -> "DataHandler":
        if not self.feature_processors:
            logger.warning("No feature processors specified. Skipping processing.")
            return self
        for name, processor_list in self.feature_processors.items():
            for Processor, args in processor_list:
                if issubclass(Processor, StaticFeaturePreprocessor):
                    logger.info(
                        f"Processing features for static processor {Processor.__name__}"
                    )
                    for split, features in self.features.items():
                        self.features[split][name] = Processor(*args).preprocess(
                            features[name]
                        )
                elif issubclass(Processor, TrainedFeaturePreprocessor):
                    logger.info(
                        "Processing features for trained processor"
                        f"{Processor.__name__}"
                    )
                    processor = Processor(*args)
                    for split, features in self.features.items():
                        self.features[split][name] = processor.preprocess(
                            features[name], split=split
                        )
        return self

    def get_iterators(self) -> Dict[DatasetType, BaseDatasetIterator]:
        """Return a dictionary of iterators for the different splits of the data.

        Returns
        -------
        Dict[DatasetType, BaseDatasetIterator]
            Dictionary of iterators for the different splits of the data.
        """
        if not self.features:
            logger.warning("No features extracted yet. Extracting features now.")
            self.extract_features().process_features()
        return {
            split: self.Iterator(self.index_splits[split], **self.features[split])
            for split in self.index_splits
        }

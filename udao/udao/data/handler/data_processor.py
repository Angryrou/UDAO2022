from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch as th
from pandas import DataFrame

from ...utils.interfaces import UdaoInputShape
from ..containers.base_container import BaseContainer
from ..extractors import FeatureExtractor
from ..iterators import BaseIterator
from ..preprocessors.base_preprocessor import FeaturePreprocessor
from ..utils.utils import DatasetType


@dataclass
class FeaturePipeline:
    extractor: FeatureExtractor
    """Tuple defining the feature extractor and its initialization arguments."""
    preprocessors: Optional[List[FeaturePreprocessor]] = None
    """List of tuples defining feature preprocessors
    and their initialization arguments."""


class DataProcessor:
    """
    Parameters
    ----------
    iterator_cls: Type[BaseDatasetIterator]
        Dataset iterator class type.

    feature_extractors: Mapping[str, Tuple[FeatureExtractorType, Any]]
        Dict that links a feature name to tuples of the form (Extractor, args)
        where Extractor implements FeatureExtractor and args are the arguments
        to be passed at initialization.
        N.B.: Feature names must match the iterator's parameters.

        If Extractor is a StaticFeatureExtractor, the features are extracted
        independently of the split.

        If Extractor is a TrainedFeatureExtractor, the extractor is first fitted
        on the train split and then applied to the other splits.

    feature_preprocessors: Optional[Mapping[str, List[FeaturePreprocessor]]]
        Dict that links a feature name to a list of tuples of the form (Processor, args)
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

    tensors_dtype: Optional[th.dtype]
        Data type of the tensors returned by the iterator, by default None
    """

    def __init__(
        self,
        iterator_cls: Type[BaseIterator],
        feature_extractors: Dict[str, FeatureExtractor],
        feature_preprocessors: Optional[
            Mapping[
                str,
                Sequence[FeaturePreprocessor],
            ]
        ] = None,
        tensors_dtype: Optional[th.dtype] = None,
    ) -> None:
        self.iterator_cls = iterator_cls
        self.feature_extractors = feature_extractors
        self.feature_processors = feature_preprocessors or {}

    def _apply_processing_function(
        self,
        function: Callable[..., BaseContainer],
        data: Union[DataFrame, BaseContainer],
        split: DatasetType,
        is_trained: bool,
    ) -> BaseContainer:
        if is_trained:
            features = function(data, split=split)
        else:
            features = function(data)

        return features

    def extract_features(
        self, data: DataFrame, split: DatasetType
    ) -> Dict[str, BaseContainer]:
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
        features: Dict[str, BaseContainer] = {}
        for name, extractor in self.feature_extractors.items():
            features[name] = self._apply_processing_function(
                extractor.extract_features,
                data,
                split=split,
                is_trained=extractor.trained,
            )
            for preprocessor in self.feature_processors.get(name, []):
                features[name] = self._apply_processing_function(
                    preprocessor.preprocess,
                    features[name],
                    split=split,
                    is_trained=preprocessor.trained,
                )

        return features

    def make_iterator(
        self, data: DataFrame, keys: Sequence, split: DatasetType
    ) -> BaseIterator:
        return self.iterator_cls(keys, **self.extract_features(data, split=split))

    def derive_batch_input(
        self,
        input_non_decision: Dict[str, Any],
        input_variables: Dict[str, list],
    ) -> Tuple[Any, UdaoInputShape]:
        """Derive the batch input from the input dict

        Parameters
        ----------
        input_non_decision : Dict[str, Any]
            The fixed values for the non-decision inputs

        input_variables : Dict[str, list]
            The values for the variables inputs

        Returns
        -------
        Any
            The batch input for the model
        """
        n_items = len(input_variables[list(input_variables.keys())[0]])
        keys = [f"{i}" for i in range(n_items)]
        pd_input = DataFrame.from_dict(
            {
                **{k: [v] * n_items for k, v in input_non_decision.items()},
                **input_variables,
                "id": keys,
            }
        )
        pd_input.set_index("id", inplace=True)
        iterator = self.make_iterator(pd_input, keys, split="test")
        dataloader = iterator.get_dataloader(batch_size=n_items)
        batch_input, _ = next(iter(dataloader))
        return batch_input, iterator.get_iterator_shape()


def create_data_processor(
    iterator_cls: Type[BaseIterator], *args: str
) -> Callable[..., DataProcessor]:
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
    params_getter: Type[DataHandlerParams]
        A dynamically generated DataHandlerParams class
        with arguments derived from the provided iterator class,
        in addition to other specified arguments.

    Notes
    -----
    The returned function has the following signature:
        >>> def get_processor(
        >>>    **kwargs: FeaturePipeline,
        >>> ) -> DataProcessor:

        where kwargs are the feature names and their corresponding feature
    """
    params = iterator_cls.get_parameter_names()
    if args is not None:
        params += args

    def get_processor(
        tensors_dtype: Optional[th.dtype] = None,
        **kwargs: FeaturePipeline,
    ) -> DataProcessor:
        feature_extractors: Dict[str, FeatureExtractor] = {}
        feature_preprocessors: Dict[str, Sequence[FeaturePreprocessor]] = {}
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

        return DataProcessor(
            iterator_cls=iterator_cls,
            tensors_dtype=tensors_dtype,
            feature_extractors=feature_extractors,
            feature_preprocessors=feature_preprocessors,
        )

    return get_processor

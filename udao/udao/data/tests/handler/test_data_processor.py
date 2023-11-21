from typing import Tuple, cast

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from ...containers.tabular_container import TabularContainer
from ...extractors.tabular_extractor import TabularFeatureExtractor, select_columns
from ...handler.data_processor import (
    DataProcessor,
    FeaturePipeline,
    create_data_processor,
)
from ...iterators.tabular_iterator import TabularIterator
from ...preprocessors.normalize_preprocessor import NormalizePreprocessor


@pytest.fixture
def df_fixture() -> Tuple[pd.DataFrame, DataProcessor]:
    n = 1000
    ids = list(range(1, n + 1))
    tids = [1 for _ in range(n - 10)] + [2 for _ in range(10)]
    values = list(range(n))

    df = pd.DataFrame.from_dict({"id": ids, "tid": tids, "value": values})
    df.set_index("id", inplace=True)

    data_processor = DataProcessor(
        iterator_cls=TabularIterator,
        feature_extractors={
            "tabular_feature": TabularFeatureExtractor(
                select_columns, columns=["value"]
            )
        },
        feature_preprocessors={
            "tabular_feature": [NormalizePreprocessor(MinMaxScaler())]
        },
    )

    return (df, data_processor)


def test_create_data_processor() -> None:
    # Create the dynamic DataHandlerParams class
    data_processor_getter = create_data_processor(TabularIterator)

    def df_func(df: DataFrame) -> DataFrame:
        return df[["col1", "col2"]]

    scaler = MinMaxScaler()
    # Instantiate the dynamic class
    params_instance = data_processor_getter(
        tabular_feature=FeaturePipeline(
            extractor=TabularFeatureExtractor(df_func),
            preprocessors=[NormalizePreprocessor(scaler)],
        ),
    )

    # Test if the provided parameters exist and are set correctly
    assert params_instance.iterator_cls == TabularIterator
    assert isinstance(
        params_instance.feature_extractors["tabular_feature"], TabularFeatureExtractor
    )
    assert params_instance.feature_extractors["tabular_feature"].feature_func == df_func

    assert params_instance.feature_processors is not None
    assert len(params_instance.feature_processors) == 1
    assert isinstance(
        params_instance.feature_processors["tabular_feature"][0], NormalizePreprocessor
    )


class TestDataProcessor:
    def test_extract_features_applies_normalization(
        self, df_fixture: Tuple[pd.DataFrame, DataProcessor]
    ) -> None:
        df, data_processor = df_fixture
        features = data_processor.extract_features(df, "train")

        assert set(features.keys()) == {"tabular_feature"}
        assert cast(TabularContainer, features["tabular_feature"]).data.shape == (
            len(df),
            1,
        )
        np.testing.assert_array_almost_equal(
            cast(TabularContainer, features["tabular_feature"]).data.values,
            np.linspace(0, 1, len(df)).reshape(-1, 1),
        )

    def test_make_iterators(
        self, df_fixture: Tuple[pd.DataFrame, DataProcessor]
    ) -> None:
        df, data_processor = df_fixture

        iterator = data_processor.make_iterator(
            keys=list(df.index), data=df, split="train"
        )
        assert iterator[0] == 0

import random
import string
from typing import Dict, List, Tuple, cast
from unittest.mock import patch

import pandas as pd
import pytest
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from ...containers.tabular_container import TabularContainer
from ...extractors import TabularFeatureExtractor
from ...handler import DataHandler
from ...handler.data_handler import (
    DataHandlerParams,
    FeaturePipeline,
    create_data_handler_params,
)
from ...iterators import TabularIterator
from ...preprocessors.normalize_preprocessor import NormalizePreprocessor
from ...utils.utils import DatasetType


def random_string(length: int) -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


@pytest.fixture
def df_fixture() -> Tuple[pd.DataFrame, DataHandlerParams]:
    n = 1000
    ids = list(range(1, n + 1))
    tids = [1 for _ in range(n - 10)] + [2 for _ in range(10)]
    random_strings = [random_string(5) for _ in range(n)]

    df = pd.DataFrame.from_dict({"id": ids, "tid": tids, "plan": random_strings})

    def df_func(df: DataFrame) -> DataFrame:
        df = df.copy()
        df = df["plan"].apply(lambda x: len(x)).to_frame("feature")
        df["id"] = df.index
        df.set_index("id", inplace=True)
        return df

    params = DataHandlerParams(
        index_column="id",
        feature_extractors={"tabular_feature": (TabularFeatureExtractor, [df_func])},
        Iterator=TabularIterator,
        stratify_on=None,
        test_frac=0.1,
        val_frac=0.3,
        random_state=1,
    )
    return df, params


def test_create_data_handler_params() -> None:
    # Create the dynamic DataHandlerParams class
    params_getter = create_data_handler_params(TabularIterator)

    def df_func(df: DataFrame) -> DataFrame:
        return df[["col1", "col2"]]

    scaler = MinMaxScaler()
    # Instantiate the dynamic class
    params_instance = params_getter(
        index_column="index",
        tabular_feature=FeaturePipeline(
            extractor=(TabularFeatureExtractor, [df_func]),
            preprocessors=[(NormalizePreprocessor, [scaler])],
        ),
    )

    # Test if the provided parameters exist and are set correctly
    assert params_instance.index_column == "index"
    assert params_instance.Iterator, TabularIterator
    assert params_instance.feature_extractors["tabular_feature"] == (
        TabularFeatureExtractor,
        [df_func],
    )
    if params_instance.feature_preprocessors is None:
        raise ValueError("feature_preprocessors should not be None")
    assert params_instance.feature_preprocessors["tabular_feature"] == [
        (NormalizePreprocessor, [scaler])
    ]


def test_create_data_handler_params_raises_error() -> None:
    params_getter = create_data_handler_params(TabularIterator)
    with pytest.raises(ValueError):
        params_getter(
            index_column="index",
        )


class TestDataHandler:
    def test_split_applies_stratification(
        self, df_fixture: Tuple[pd.DataFrame, DataHandlerParams]
    ) -> None:
        df, params = df_fixture
        params.stratify_on = "tid"
        dh = DataHandler(df, params)
        dh.split_data()
        for split, keys in dh.index_splits.items():
            df_split = df.loc[keys]
            assert len(df_split[df_split["tid"] == 1]) == 99 * len(
                df_split[df_split["tid"] == 2]
            )

    def test_split_no_stratification(
        self, df_fixture: Tuple[pd.DataFrame, DataHandlerParams]
    ) -> None:
        """Check that the split is done correctly
        when no stratification is applied.
        The proportions are correct and there is
        no intersection between the splits."""
        df, params = df_fixture
        params.stratify_on = None
        params.random_state = 1
        dh = DataHandler(df, params)
        dh.split_data()
        assert len(dh.index_splits["test"]) / len(dh.full_df) == dh.test_frac
        assert len(dh.index_splits["val"]) / len(dh.full_df) == dh.val_frac
        assert len(dh.index_splits["train"]) / len(dh.full_df) == 1 - (
            dh.val_frac + dh.test_frac
        )
        assert not set(dh.index_splits["test"]) & set(dh.index_splits["val"])
        assert not set(dh.index_splits["test"]) & set(dh.index_splits["train"])
        assert not set(dh.index_splits["val"]) & set(dh.index_splits["train"])

        df_split = df.loc[dh.index_splits["test"]]
        # results are deterministic because of random_state
        # another random_state would give different results
        assert len(df_split[df_split["tid"] == 1]) == 98
        assert len(df_split[df_split["tid"] == 2]) == 2

    def test_extract_features_has_right_shape(
        self, df_fixture: Tuple[pd.DataFrame, DataHandlerParams]
    ) -> None:
        df, params = df_fixture
        dh = DataHandler(df, params)
        dh.split_data().extract_features()
        df_features_dict: Dict[DatasetType, pd.DataFrame] = {
            s: cast(TabularContainer, dh.features[s]["tabular_feature"]).data
            for s in dh.features
        }
        for split in dh.features:
            assert set(df_features_dict[split].index) == set(dh.index_splits[split])

        expected_lengths = {
            "train": len(df) * (1 - params.val_frac - params.test_frac),
            "val": len(df) * params.val_frac,
            "test": len(df) * params.test_frac,
        }
        for df_feature, length in zip(
            df_features_dict.values(), expected_lengths.values()
        ):
            assert len(df_feature) == length

    def test_extract_feature_calls_split_data(
        self, df_fixture: Tuple[pd.DataFrame, DataHandlerParams]
    ) -> None:
        df, params = df_fixture
        dh = DataHandler(df, params)
        with patch.object(dh, "split_data") as mock_split_data:
            dh.extract_features()
            mock_split_data.assert_called_once()

    def test_postprocess(
        self, df_fixture: Tuple[pd.DataFrame, DataHandlerParams]
    ) -> None:
        _, params = df_fixture
        df = pd.DataFrame.from_dict(
            {"id": ["0", "1", "2", "3", "4"], "plan": ["h", "hello", "word", "!", ""]}
        )
        df.set_index("id", inplace=True, drop=False)
        params.feature_preprocessors = {
            "tabular_feature": [(NormalizePreprocessor, [MinMaxScaler()])]
        }
        dh = DataHandler(df, params)
        # manual split to have a deterministic test
        dh.index_splits = {"train": ["0", "1"], "val": ["2", "3"], "test": ["4"]}
        dh.extract_features().process_features()
        train_results: Dict[DatasetType, List[List[float]]] = {
            "train": [[0], [1]],
            "val": [
                [0.75],
                [0],
            ],
            "test": [
                [-0.25],
            ],
        }
        for split, expected_results in train_results.items():
            container = cast(TabularContainer, dh.features[split]["tabular_feature"])
            assert container.data.values.tolist() == expected_results

    def test_get_iterators(
        self, df_fixture: Tuple[pd.DataFrame, DataHandlerParams]
    ) -> None:
        df, params = df_fixture
        dh = DataHandler(df, params)
        dh.split_data().extract_features()
        iterators = dh.get_iterators()
        assert len(iterators) == 3
        assert all(isinstance(it, params.Iterator) for it in iterators.values())
        for split, it in iterators.items():
            features_container = cast(
                TabularContainer, dh.features[split]["tabular_feature"]
            )
            assert len(it) == len(features_container.data)

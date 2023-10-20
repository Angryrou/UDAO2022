import random
import string
from typing import Tuple

import pandas as pd
import pytest

from ...data.dataset import DataHandler, DataHandlerParams
from ...data.tabular_dataset import TabularIterator
from ...data.utils.utils import TabularFeatureExtractor


def random_string(length: int) -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


@pytest.fixture
def df_fixture() -> Tuple[pd.DataFrame, DataHandlerParams]:
    n = 1000
    ids = list(range(1, n + 1))
    tids = [1 for _ in range(n - 10)] + [2 for _ in range(10)]
    random_strings = [random_string(5) for _ in range(n)]

    df = pd.DataFrame.from_dict({"id": ids, "tid": tids, "plan": random_strings})
    params = DataHandlerParams(
        index_column="id",
        feature_extractors=[(TabularFeatureExtractor, [lambda r: r["plan"][0]])],
        Iterator=TabularIterator,
        stratify_on=None,
        test_frac=0.1,
        val_frac=0.3,
        random_state=1,
    )
    return df, params


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
        print(dh.features["train"]["feature_frame"].index)
        for split in dh.features:
            assert set(dh.features[split]["feature_frame"].index) == set(
                dh.index_splits[split]
            )
        assert len(dh.features["train"]["feature_frame"]) == len(df) * (
            1 - params.val_frac - params.test_frac
        )
        assert len(dh.features["val"]["feature_frame"]) == len(df) * params.val_frac
        assert len(dh.features["test"]["feature_frame"]) == len(df) * params.test_frac

    def test_extract_feature_raises_error(
        self, df_fixture: Tuple[pd.DataFrame, DataHandlerParams]
    ) -> None:
        with pytest.raises(
            ValueError,
        ):
            df, params = df_fixture
            dh = DataHandler(df, params)
            dh.extract_features()

    def test_get_iterators(
        self, df_fixture: Tuple[pd.DataFrame, DataHandlerParams]
    ) -> None:
        df, params = df_fixture
        dh = DataHandler(df, params)
        dh.split_data().extract_features()
        iterators = dh.get_iterators()
        assert len(iterators) == 3
        assert all(isinstance(it, params.Iterator) for it in iterators.values())
        assert all(
            len(it) == len(dh.features[split]["feature_frame"])
            for split, it in iterators.items()
        )

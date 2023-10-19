import random
import string

import pandas as pd
import pytest
from udao.data.dataset import DataHandler, DataHandlerParams
from udao.data.tabular_dataset import TabularIterator
from udao.data.utils.utils import TabularFeatureExtractor


def random_string(length: int) -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


@pytest.fixture
def df_fixture() -> pd.DataFrame:
    n = 1000
    ids = list(range(1, n + 1))
    tids = [1 for _ in range(n - 10)] + [2 for _ in range(10)]
    random_strings = [random_string(5) for _ in range(n)]

    df = pd.DataFrame.from_dict({"id": ids, "tid": tids, "plan": random_strings})
    return df


class TestDataHandler:
    def test_split_applies_stratification(self, df_fixture: pd.DataFrame) -> None:
        params = DataHandlerParams(
            index_column="id",
            feature_extractors=[(TabularFeatureExtractor, lambda r: r["plan"][0])],
            Iterator=TabularIterator,
            stratify_on="tid",
            test_frac=0.1,
            val_frac=0.1,
        )
        dh = DataHandler(df_fixture, params)
        dh.split_data()
        for split, keys in dh.index_splits.items():
            df_split = df_fixture.loc[keys]
            assert len(df_split[df_split["tid"] == 1]) == 99 * len(
                df_split[df_split["tid"] == 2]
            )

    def test_split_no_stratification(self, df_fixture: pd.DataFrame) -> None:
        params = DataHandlerParams(
            index_column="id",
            feature_extractors=[(TabularFeatureExtractor, lambda r: r["plan"][0])],
            Iterator=TabularIterator,
            stratify_on=None,
            test_frac=0.1,
            val_frac=0.1,
            random_state=1,
        )
        dh = DataHandler(df_fixture, params)
        dh.split_data()
        df_split = df_fixture.loc[dh.index_splits["test"]]
        # results are deterministic because of random_state
        # another random_state would give different results
        assert len(df_split[df_split["tid"] == 1]) == 97
        assert len(df_split[df_split["tid"] == 2]) == 3

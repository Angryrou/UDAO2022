import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ...containers.tabular_container import TabularContainer
from ...preprocessors.normalize_preprocessor import NormalizePreprocessor


class TestNormalizePreprocessor:
    def test_fit_and_transform(self) -> None:
        df_train = pd.DataFrame.from_dict({"index": [1, 2], "col": [3, 4]})
        df_val = pd.DataFrame.from_dict({"index": [3, 4], "col": [5, 6]})
        df_train.set_index("index", inplace=True)
        df_val.set_index("index", inplace=True)
        train_container = TabularContainer(df_train)
        processor: NormalizePreprocessor[TabularContainer] = NormalizePreprocessor(
            MinMaxScaler()
        )
        train_container_result = processor.preprocess(train_container, "train")
        assert train_container_result.data.values.tolist() == [[0.0], [1.0]]
        assert train_container.data.values.tolist() == [[3.0], [4.0]]
        val_container = TabularContainer(df_val)
        val_container_result = processor.preprocess(val_container, "val")
        assert val_container_result.data.values.tolist() == [[2.0], [3.0]]
        assert val_container.data.values.tolist() == [[5.0], [6.0]]

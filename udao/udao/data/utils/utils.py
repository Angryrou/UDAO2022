from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Literal, Optional, Type, Union

import pandas as pd
from sklearn.model_selection import train_test_split

PandasTypes = {float: "float64", int: "int64", str: "object"}

DatasetType = Literal["train", "val", "test"]


class TrainedFeatureExtractor(ABC):
    trained: bool = True

    def __init__(self) -> None:
        pass

    @abstractmethod
    def extract_features(self, df: pd.DataFrame, split: DatasetType) -> Dict[str, Any]:
        pass


class StaticFeatureExtractor(ABC):
    trained: bool = False

    def __init__(self) -> None:
        pass

    @abstractmethod
    def extract_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        pass


FeatureExtractorType = Union[
    Type[TrainedFeatureExtractor], Type[StaticFeatureExtractor]
]


class TabularFeatureExtractor(StaticFeatureExtractor):
    def __init__(self, feature_func: Callable) -> None:
        self.feature_func = feature_func

    def extract_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        feature_series = df.apply(self.feature_func, axis=1)
        feature_df = feature_series.to_frame(name="feature")
        return {"feature_frame": feature_df}


def train_test_val_split_on_column(
    df: pd.DataFrame,
    groupby_col: Optional[str],
    *,
    val_frac: float,
    test_frac: float,
    random_state: Optional[int] = None
) -> Dict[DatasetType, pd.DataFrame]:
    """return tr_mask, val_mask, te_mask"""
    train_df, non_train_df = train_test_split(
        df,
        test_size=val_frac + test_frac,
        stratify=df[groupby_col] if groupby_col else None,
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        non_train_df,
        test_size=test_frac / (val_frac + test_frac),
        stratify=non_train_df[groupby_col] if groupby_col else None,
        random_state=random_state,
    )
    df_dict: Dict[DatasetType, pd.DataFrame] = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    return df_dict

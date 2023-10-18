from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Type, Union

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


def train_test_val_split_on_column(
    df: pd.DataFrame, groupby_col: str, *, val_frac: float, test_frac: float
) -> Dict[DatasetType, pd.DataFrame]:
    """return tr_mask, val_mask, te_mask"""
    train_df, non_train_df = train_test_split(
        df, test_size=val_frac + test_frac, stratify=df[groupby_col]
    )
    test_df, val_df = train_test_split(
        non_train_df,
        test_size=test_frac / (val_frac + test_frac),
        stratify=non_train_df[groupby_col],
    )
    df_dict: Dict[DatasetType, pd.DataFrame] = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    return df_dict

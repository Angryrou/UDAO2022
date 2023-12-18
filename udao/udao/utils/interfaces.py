from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

import torch as th

T = TypeVar("T")

ST = TypeVar("ST")


@dataclass
class FeatureInput:
    feature_input: th.Tensor


@dataclass
class FeatureInputShape:
    feature_input_names: list[str]
    output_names: list[str]


@dataclass
class UdaoInput(Generic[T], FeatureInput):
    embedding_input: T


@dataclass
class UdaoInputShape(Generic[ST], FeatureInputShape):
    embedding_input_shape: ST


class VarTypes(Enum):
    INT = "int"
    BOOL = "bool"
    CATEGORY = "category"
    FLOAT = "float"

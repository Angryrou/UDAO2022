from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

import torch as th

T = TypeVar("T")

ST = TypeVar("ST")


@dataclass
class UdaoInput(Generic[T]):
    embedding_input: T
    feature_input: th.Tensor


@dataclass
class UdaoInputShape(Generic[ST]):
    embedding_input_shape: ST
    feature_input_names: list[str]
    output_shape: int


class VarTypes(Enum):
    INT = "int"
    BOOL = "bool"
    CATEGORY = "category"
    FLOAT = "float"

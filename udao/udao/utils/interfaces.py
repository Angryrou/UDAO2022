from dataclasses import dataclass
from typing import Generic, TypeVar

import torch as th

T = TypeVar("T")


@dataclass
class BaseUdaoInput(Generic[T]):
    embedding_input: T
    feature_input: th.Tensor

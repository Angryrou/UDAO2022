from dataclasses import dataclass
from typing import Generic, TypeVar

import torch as th

T = TypeVar("T")


@dataclass
class BaseUdaoInput(Generic[T]):  # To do: move to an interface between data and model
    embedding_input: T
    feature_input: th.Tensor

from dataclasses import dataclass
from typing import Any, Callable, Literal, Union

import torch as th

from ...data.handler.data_processor import DataProcessor
from .utils import ModelComponent

ConstraintType = Union[Literal["=="], Literal["<="], Literal[">="]]


@dataclass
class Constraint:
    lower: float
    upper: float
    function: Callable[..., th.Tensor]
    stress: float = 0.0

    def __call__(self, *args: Any, **kwargs: Any) -> th.Tensor:
        return self.function(*args, **kwargs)


class ModelConstraint(ModelComponent, Constraint):
    def __init__(
        self,
        lower: float,
        upper: float,
        data_processor: DataProcessor,
        model: th.nn.Module,
    ) -> None:
        ModelComponent.__init__(self, data_processor, model)

        def function(*args: Any, **kwargs: Any) -> th.Tensor:
            return self.apply_model(*args, **kwargs)

        Constraint.__init__(self, lower, upper, function)

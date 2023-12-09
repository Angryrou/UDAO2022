from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

import torch as th

from ...data.handler.data_processor import DataProcessor
from .utils import ModelComponent

ConstraintType = Union[Literal["=="], Literal["<="], Literal[">="]]


@dataclass
class Constraint:
    function: Callable[..., th.Tensor]
    lower: Optional[float] = None
    upper: Optional[float] = None
    stress: float = 0.0

    def __call__(self, *args: Any, **kwargs: Any) -> th.Tensor:
        return self.function(*args, **kwargs)


class ModelConstraint(ModelComponent, Constraint):
    def __init__(
        self,
        data_processor: DataProcessor,
        model: th.nn.Module,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        stress: float = 0.0,
    ) -> None:
        ModelComponent.__init__(self, data_processor, model)

        def function(*args: Any, **kwargs: Any) -> th.Tensor:
            return self.apply_model(*args, **kwargs)

        Constraint.__init__(
            self, function=function, lower=lower, upper=upper, stress=stress
        )

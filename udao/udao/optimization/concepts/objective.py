from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

import torch as th

from ...data.handler.data_processor import DataProcessor
from ...utils.interfaces import VarTypes
from .utils import ModelComponent

ObjectiveDirection = Union[Literal["MIN"], Literal["MAX"]]


@dataclass
class Objective:
    name: str
    direction_type: ObjectiveDirection
    function: Callable[..., th.Tensor]
    upper: Optional[float] = None
    lower: Optional[float] = None
    type: Optional[VarTypes] = None

    @property
    def direction(self) -> int:
        """Get gradient direction from optimization type"""
        if self.direction_type == "MIN":
            return 1
        else:
            return -1

    def __call__(self, *args: Any, **kwargs: Any) -> th.Tensor:
        return self.function(*args, **kwargs)


class ModelObjective(ModelComponent, Objective):
    def __init__(
        self,
        name: str,
        direction_type: ObjectiveDirection,
        data_processor: DataProcessor,
        model: th.nn.Module,
        type: Optional[VarTypes] = None,
    ) -> None:
        ModelComponent.__init__(self, data_processor, model)

        def function(*args: Any, **kwargs: Any) -> th.Tensor:
            return self.apply_model(*args, **kwargs)

        Objective.__init__(self, name, direction_type, function, type)

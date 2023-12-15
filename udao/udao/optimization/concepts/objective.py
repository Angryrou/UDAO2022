from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import torch as th

from ...utils.interfaces import VarTypes
from .utils import UdaoFunction

ObjectiveDirection = Union[Literal["MIN"], Literal["MAX"]]


@dataclass
class Objective:
    name: str
    direction_type: ObjectiveDirection
    function: UdaoFunction
    lower: Optional[float] = None
    upper: Optional[float] = None
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

    def __repr__(self) -> str:
        return (
            f"Objective(name={self.name}, direction={self.direction_type}, "
            f"lower={self.lower}, upper={self.upper})"
        )

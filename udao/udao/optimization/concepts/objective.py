from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

import torch as th

from ...utils.interfaces import VarTypes

ObjectiveDirection = Union[Literal["MIN"], Literal["MAX"]]


@dataclass
class Objective:
    name: str
    direction_type: ObjectiveDirection
    function: Callable[..., th.Tensor]
    type: Optional[VarTypes] = None

    @property
    def direction(self) -> int:
        """Get gradient direction from optimization type"""
        if self.direction_type == "MIN":
            return 1
        else:
            return -1

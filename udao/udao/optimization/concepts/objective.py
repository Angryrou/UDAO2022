from dataclasses import dataclass
from typing import Callable, Literal, Union

import torch as th

ObjectiveType = Union[Literal["MIN"], Literal["MAX"]]


@dataclass
class Objective:
    name: str
    type: ObjectiveType
    function: Callable[..., th.Tensor]

    @property
    def direction(self) -> int:
        """Get gradient direction from optimization type"""
        if self.type == "MIN":
            return 1
        else:
            return -1

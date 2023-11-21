from dataclasses import dataclass
from typing import Callable, Literal, Union

import numpy as np

ObjectiveType = Union[Literal["MIN"], Literal["MAX"]]


@dataclass
class Objective:
    type: ObjectiveType
    function: Callable[..., np.ndarray]

    @property
    def direction(self) -> int:
        """Get gradient direction from optimization type"""
        if self.type == "MIN":
            return 1
        else:
            return -1

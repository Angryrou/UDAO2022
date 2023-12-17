from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import torch as th

from ...utils.interfaces import VarTypes
from .utils import UdaoFunction

ObjectiveDirection = Union[Literal["MIN"], Literal["MAX"]]


@dataclass
class Objective:
    """Objective to optimize."""

    name: str
    """Name of the objective."""
    direction_type: ObjectiveDirection
    """Direction of the objective: MIN or MAX."""
    function: UdaoFunction
    """Objective function."""
    lower: Optional[float] = None
    """Lower bound of the objective."""
    upper: Optional[float] = None
    """Upper bound of the objective."""
    type: Optional[VarTypes] = None
    """Type of the objective.
    If int, the optimization can behave differently."""

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

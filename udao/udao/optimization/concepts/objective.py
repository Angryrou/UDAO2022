from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

import torch as th

from ...utils.interfaces import VarTypes
from ..concepts.utils import UdaoFunction

ObjectiveDirection = Union[Literal["MIN"], Literal["MAX"]]


@dataclass
class Objective:
    """Objective to optimize."""

    name: str
    """Name of the objective."""
    direction_type: ObjectiveDirection
    """Direction of the objective: MIN or MAX."""
    function: Union[UdaoFunction, th.nn.Module, Callable[..., th.Tensor]]
    """Objective function.
    The choice of the type depends on whether a DataProcessor is specified
    for the problem:
    - if no DataProcessor is provided: UdaoFunction, it is a callable
    that takes input_variables and input_parameters
    - else, th.nn.Module or other Callable returning a tensor.
    """
    lower: Optional[float] = None
    """Lower bound of the objective."""
    upper: Optional[float] = None
    """Upper bound of the objective."""
    type: VarTypes = VarTypes.FLOAT
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

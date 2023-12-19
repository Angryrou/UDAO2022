from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

import torch as th

from ..concepts.utils import UdaoFunction

ConstraintType = Union[Literal["=="], Literal["<="], Literal[">="]]


@dataclass
class Constraint:
    """Constraint for optimization."""

    function: Union[UdaoFunction, th.nn.Module, Callable[..., th.Tensor]]
    """Objective function.
    The choice of the type depends on whether a DataProcessor is specified
    for the problem:
    - if no DataProcessor is provided: UdaoFunction, it is a callable
    that takes input_variables and input_parameters
    - else, th.nn.Module or other Callable returning a tensor.
    """
    lower: Optional[float] = None
    """lower bound of the constraint."""
    upper: Optional[float] = None
    """upper bound of the constraint."""
    stress: float = 0.0
    """stress to be applied when a loss is computed"""

    def __call__(self, *args: Any, **kwargs: Any) -> th.Tensor:
        return self.function(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"Constraint(lower={self.lower}, upper={self.upper}, stress={self.stress})"
        )

from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import torch as th

from .utils import UdaoFunction

ConstraintType = Union[Literal["=="], Literal["<="], Literal[">="]]


@dataclass
class Constraint:
    """Constraint for optimization."""

    function: UdaoFunction
    """Constraint function."""
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

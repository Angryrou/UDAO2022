from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import torch as th

from .utils import UdaoFunction

ConstraintType = Union[Literal["=="], Literal["<="], Literal[">="]]


@dataclass
class Constraint:
    function: UdaoFunction
    lower: Optional[float] = None
    upper: Optional[float] = None
    stress: float = 0.0

    def __call__(self, *args: Any, **kwargs: Any) -> th.Tensor:
        return self.function(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"Constraint(lower={self.lower}, upper={self.upper}, stress={self.stress})"
        )

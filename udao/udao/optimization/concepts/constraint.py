from typing import Callable, Literal, Optional, Union

import torch as th

from .optimization_element import OptimizationElement
from .utils import UdaoFunction

ConstraintType = Union[Literal["=="], Literal["<="], Literal[">="]]


class Constraint(OptimizationElement):
    def __init__(
        self,
        function: Union[UdaoFunction, th.nn.Module, Callable[..., th.Tensor]],
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        stress: float = 0.0,
    ):
        super().__init__(function=function, lower=lower, upper=upper)
        self.stress = stress

    def __repr__(self) -> str:
        return (
            f"Constraint(lower={self.lower}, upper={self.upper}, stress={self.stress})"
        )

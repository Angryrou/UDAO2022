from typing import Callable, Literal, Optional, Union

import torch as th

from ...utils.interfaces import VarTypes
from ..concepts.utils import UdaoFunction
from .optimization_element import OptimizationElement

ObjectiveDirection = Union[Literal["MIN"], Literal["MAX"]]


class Objective(OptimizationElement):
    """

    Parameters
    ----------
    name : str
        Name of the objective.
    direction_type : ObjectiveDirection
        Direction of the objective: MIN or MAX.
    type: VarTypes
        Type of the objective, by default VarTypes.FLOAT
    """

    def __init__(
        self,
        name: str,
        direction_type: ObjectiveDirection,
        function: Union[UdaoFunction, th.nn.Module, Callable[..., th.Tensor]],
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        type: VarTypes = VarTypes.FLOAT,
    ):
        super().__init__(function=function, lower=lower, upper=upper)
        self.name = name
        self.direction_type = direction_type
        self.type = type

    @property
    def direction(self) -> int:
        """Get gradient direction from optimization type"""
        if self.direction_type == "MIN":
            return 1
        else:
            return -1

    def __repr__(self) -> str:
        return (
            f"Objective(name={self.name}, direction={self.direction_type}, "
            f"lower={self.lower}, upper={self.upper})"
        )

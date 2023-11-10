from dataclasses import dataclass, field
from typing import Union


@dataclass
class Variable:
    pass


@dataclass
class NumericVariable(Variable):
    lower: Union[int, float]
    upper: Union[int, float]

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(
                f"ERROR: the lower bound of variable {self}"
                " is greater than its upper bound!"
            )


@dataclass
class IntegerVariable(NumericVariable):
    lower: int
    upper: int


@dataclass
class FloatVariable(NumericVariable):
    lower: float
    upper: float


@dataclass
class BoolVariable(IntegerVariable):
    lower: int = field(default=0, init=False)
    upper: int = field(default=1, init=False)


@dataclass
class EnumVariable(Variable):
    values: list

from dataclasses import dataclass, field
from typing import Union
import numpy as np


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


def random_variable(var: Variable, n_samples: int) -> np.ndarray:
    if isinstance(var, FloatVariable):
        return np.random.uniform(var.lower, var.upper, n_samples)
    elif isinstance(var, IntegerVariable):
        return np.random.randint(var.lower, var.upper + 1, size=n_samples)
    elif isinstance(var, EnumVariable):
        inds = np.random.randint(0, len(var.values), size=n_samples)
        return np.array(var.values)[inds]
    else:
        raise NotImplementedError(f"ERROR: variable type {type(var)} is not supported!")

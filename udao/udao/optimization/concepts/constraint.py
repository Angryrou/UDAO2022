from dataclasses import dataclass
from typing import Callable, Literal, Union

import numpy as np

ConstraintType = Union[Literal["=="], Literal["<="], Literal[">="]]


@dataclass
class Constraint:
    type: ConstraintType
    function: Callable[..., np.ndarray]

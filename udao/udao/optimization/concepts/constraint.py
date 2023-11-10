from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

ConstraintType = Literal["=="] | Literal["<="] | Literal[">="]


@dataclass
class Constraint:
    type: ConstraintType
    function: Callable[..., np.ndarray]

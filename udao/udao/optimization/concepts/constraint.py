from dataclasses import dataclass
from typing import Callable, Literal, Union

import numpy as np
import torch as th

ConstraintType = Union[Literal["=="], Literal["<="], Literal[">="]]


@dataclass
class Constraint:
    type: ConstraintType
    function: Callable[..., Union[np.ndarray, th.Tensor]]

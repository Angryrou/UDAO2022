# Author(s): chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 9/14/22


from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..concepts.variable import Variable


class BaseSolver(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def _get_input(self, variables: List[Variable]) -> np.ndarray:
        pass

    # @abstractmethod
    # def solve(self, ):
    #     ...

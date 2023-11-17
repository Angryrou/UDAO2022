# Author(s): chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 9/14/22


from abc import ABC, abstractmethod
from typing import List

from ..concepts import Constraint, Objective, Variable
from ..utils.moo_utils import Point


class BaseSolver(ABC):
    @abstractmethod
    def solve(
        self,
        objective: Objective,
        constraints: List[Constraint],
        variables: List[Variable],
        wl_id: str | None,
    ) -> Point:
        pass

    # @abstractmethod
    # def solve(self, ):
    #     ...

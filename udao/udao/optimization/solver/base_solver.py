from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..concepts import Constraint, Objective, Variable
from ..utils.moo_utils import Point


class BaseSolver(ABC):
    @abstractmethod
    def solve(
        self,
        objective: Objective,
        variables: Dict[str, Variable],
        constraints: Optional[List[Constraint]] = None,
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> Point:
        ...

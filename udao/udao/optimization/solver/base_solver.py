from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

from ..concepts import Constraint, Objective, Variable


class BaseSolver(ABC):
    @abstractmethod
    def solve(
        self,
        objective: Objective,
        variables: Dict[str, Variable],
        constraints: Optional[Sequence[Constraint]] = None,
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        ...

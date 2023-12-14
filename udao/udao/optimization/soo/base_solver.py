from abc import ABC, abstractmethod
from typing import Dict, Tuple

from ..concepts import SOProblem


class SOSolver(ABC):
    @abstractmethod
    def solve(
        self,
        problem: SOProblem,
    ) -> Tuple[float, Dict[str, float]]:
        ...

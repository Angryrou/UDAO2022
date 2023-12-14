from abc import ABC, abstractmethod
from typing import Any

from ..concepts.problem import MOProblem


class BaseMOO(ABC):
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def solve(self, problem: MOProblem) -> Any:
        ...

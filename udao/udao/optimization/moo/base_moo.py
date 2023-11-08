from abc import ABC, abstractmethod
from typing import Any


class BaseMOO(ABC):
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> Any:
        ...

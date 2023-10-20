from abc import ABC, abstractmethod


class BaseContainer(ABC):
    @abstractmethod
    def get(self, key: str):  # type: ignore
        pass

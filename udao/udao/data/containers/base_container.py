from abc import ABC, abstractmethod


class BaseContainer(ABC):
    """Base class for containers.
    Containers are used to store and retrieve data
    from a dataset, based on a key."""

    @abstractmethod
    def get(self, key: str):  # type: ignore
        pass

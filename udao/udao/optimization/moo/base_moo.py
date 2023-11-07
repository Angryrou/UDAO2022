from abc import ABCMeta, abstractmethod


class BaseMOO(object, metaclass=ABCMeta):
    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def solve(self, *args):
        ...

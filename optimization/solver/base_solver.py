# Author(s): chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 9/14/22

from abc import ABCMeta, abstractmethod

class BaseSolver(object, metaclass=ABCMeta):
    def __init__(self, ):
        pass

    @abstractmethod
    def solve(self, ):
        ...
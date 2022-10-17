# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: APIs for predictive models
#
# Created at 14/10/2022

from abc import ABCMeta, abstractmethod

class BaseModel(object, metaclass=ABCMeta):
    def __init__(self, ):
        pass

    @abstractmethod
    def initialize(self, *args):
        ...

    @abstractmethod
    def fit(self, *args):
        ...

    @abstractmethod
    def predict(self, *args):
        ...
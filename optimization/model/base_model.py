# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
# Author(s): Chenghao LYU <chenghao at cs dot umass dot edu>
#
# Description: APIs for predictive models
#
# Created at 14/10/2022

from abc import ABCMeta, abstractmethod


class BaseModel(object, metaclass=ABCMeta):
    def __init__(self, target_objs):
        self.target_objs = target_objs

    @abstractmethod
    def initialize(self, *args):
        ...

    @abstractmethod
    def fit(self, *args):
        ...

    @abstractmethod
    def normalize_config(self, config):
        ...

    @abstractmethod
    def internal_prediction(self, obj, config_norm, *args):
        ...

    def predict(self, obj, config, *args):
        assert obj in self.target_objs
        config_norm = self.normalize_config(config)
        obj_val = self.internal_prediction(obj, config_norm, args)
        return obj_val

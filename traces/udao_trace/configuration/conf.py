import math
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Iterable

import numpy as np
import pandas as pd

from . import KnobMeta
from ..utils import JsonHandler, VarTypes, ScaleTypes


class Conf(ABC):

    def __init__(self, meta_file: str):
        knob_meta = JsonHandler.load_json(meta_file)

        self.knob_list = [
            KnobMeta(
                k["id"],
                k["name"],
                k["type"],
                k["construction_type"],
                k["unit"],
                k["min"],
                k["max"],
                k["scale"],
                k["factor"],
                k["base"],
                k["categories"],
                k["default"],
                k["desc"]
            )
            for k in knob_meta
        ]
        self.knob_dict_by_id = {k.id: k for k in self.knob_list}
        self.knob_dict_by_name = {k.name: k for k in self.knob_list}

    @staticmethod
    def _denormalize_knob(k_norm: Union[float, Iterable[float]], k_meta: KnobMeta) -> Union[float, Iterable[float]]:
        if isinstance(k_norm, float):
            assert 0 <= k_norm <= 1, f"normalized value {k_norm} is not in [0, 1]"
        elif isinstance(k_norm, Iterable):
            assert all(0 <= k_norm_ <= 1 for k_norm_ in k_norm), f"normalized value {k_norm} is not in [0, 1]"
        else:
            raise Exception(f"unsupported type {type(k_norm)}")

        k_value = k_meta.min + k_norm * (k_meta.max - k_meta.min)
        if k_meta.ktype in (VarTypes.INT, VarTypes.CATEGORY, VarTypes.BOOL):
            k_value = int(round(k_value))

        return k_value

    @staticmethod
    def _construct_knob(k_denorm: float, k_meta: KnobMeta) -> Union[int, float]:
        if k_meta.scale == ScaleTypes.LINEAR:
            k_value = k_denorm * k_meta.factor
        elif k_meta.scale == ScaleTypes.LOG:
            k_value = math.pow(k_meta.base, k_denorm) * k_meta.factor
        else:
            raise Exception(f"unknown scale type {k_meta.scale}")

        if k_meta.ctype in (VarTypes.INT, VarTypes.CATEGORY, VarTypes.BOOL):
            k_value = int(round(k_value))

        return k_value

    @staticmethod
    def _add_unit(k_value, k_meta: KnobMeta) -> str:
        if k_meta.ctype == VarTypes.BOOL:
            assert k_value in (0, 1), f"bool value {k_value} is not 0 or 1"
            return "true" if k_value == 1 else "false"
        return f"{k_value:g}" if k_meta.unit is None else f"{k_value:g}{k_meta.unit}"

    def get_default_conf(self, to_dict: bool = True) -> Union[List, Dict]:
        if to_dict:
            return {k.name: k.default for k in self.knob_list}
        else:
            return [k.default for k in self.knob_list]

    def normalize(self, conf: Union[Iterable, List[float]]) -> Union[Iterable, List[float]]:
        ...

    def denormalize(self, conf_norm: Union[Iterable, List[float]]) -> Union[Iterable, List[float]]:
        if isinstance(conf_norm, List):
            if isinstance(conf_norm, list):
                assert (len(conf_norm) == len(self.knob_list)), "length of conf_norm as a List should match knob_list"
            else:
                raise Exception(f"unsupported type {type(conf_norm)}")
            return [self._denormalize_knob(k_norm, k_meta) for k_norm, k_meta in zip(conf_norm, self.knob_list)]
        elif isinstance(conf_norm, np.ndarray):
            assert ((len(conf_norm) > 0) and (len(conf_norm[0]) == len(self.knob_list))), \
                "number of columns of conf_norm as a np.ndarray should match knob_list"
            return np.vectorize(lambda c, k: self._denormalize_knob(c, k))(conf_norm, self.knob_list)
        elif isinstance(conf_norm, pd.DataFrame):
            assert (conf_norm.shape[1] == len(self.knob_list)), \
                "number of columns of conf_norm as a DataFrame should match knob_list"
            conf_norm[:] = np.vectorize(lambda c, k: self._denormalize_knob(c, k))(conf_norm, self.knob_list)
            return conf_norm
        else:
            raise Exception(f"unsupported type {type(conf_norm)}")

    @abstractmethod
    def construct_configuration(self, conf_denorm: Union[Iterable, List[float]]) -> Union[Iterable, List[str]]:
        ...

    @abstractmethod
    def deconstruct_configuration(self, conf: Union[pd.DataFrame, List]) -> Union[pd.DataFrame, List]:
        ...

    def construct_configuration_from_norm(self, conf_norm: Union[Iterable, List[float]]) -> Union[Iterable, List[str]]:
        conf_denorm = self.denormalize(conf_norm)
        return self.construct_configuration(conf_denorm)

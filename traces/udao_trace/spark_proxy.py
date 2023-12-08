import math
from typing import Union, List, Dict, Iterable

import pandas as pd
from pandas import DataFrame
import numpy as np

from udao_trace import KnobMeta
from udao_trace.utils import JsonHandler, VarTypes, ScaleTypes


class SparkProxy(object):

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

    def get_default_conf(self, to_dict: bool = True) -> Union[List, Dict]:
        if to_dict:
            return {k.name: k.default for k in self.knob_list}
        else:
            return [k.default for k in self.knob_list]

    def normalize(self, conf: Union[DataFrame, List, Dict]) -> Union[DataFrame, List, Dict]:
        ...

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
        elif isinstance(conf_norm, DataFrame):
            assert (conf_norm.shape[1] == len(self.knob_list)), \
                "number of columns of conf_norm as a DataFrame should match knob_list"
            conf_norm[:] = np.vectorize(lambda c, k: self._denormalize_knob(c, k))(conf_norm, self.knob_list)
            return conf_norm
        else:
            raise Exception(f"unsupported type {type(conf_norm)}")

    def deconstruct_configuration(self, conf: Union[DataFrame, List]) -> Union[DataFrame, List]:
        # convert from str to numeric values, with unit removed and knob dependency resolved.
        ...

    @staticmethod
    def _add_unit(k_value, k_meta: KnobMeta) -> str:
        if k_meta.ctype == VarTypes.BOOL:
            assert k_value in (0, 1), f"bool value {k_value} is not 0 or 1"
            return "true" if k_value == 1 else "false"
        return f"{k_value:g}" if k_meta.unit is None else f"{k_value:g}{k_meta.unit}"

    @staticmethod
    def _compute_k5(dop: int, k5: int) -> int:
        assert (k5 in (0, 1)), f"k5 value {k5} is not 0 or 1"
        return (dop - 1 if dop <= 200 else 200) if k5 == 1 else (dop + 1 if dop > 200 else 200)

    def construct_configuration(self, conf_denorm: Union[Iterable, List]) -> Union[Iterable, List[str]]:
        # convert from numeric values to str parameter values, with dependency resolved and unit added
        if isinstance(conf_denorm, List):
            conf = [self._construct_knob(k_denorm, k_meta) for k_denorm, k_meta in zip(conf_denorm, self.knob_list)]
            assert (len(conf) == len(self.knob_list)), "length of conf as a List should match knob_list"
            # a special transformation for k4 and k6
            # k6: set spark.shuffle.sort.bypassMergeThreshold <= (k2 * k3 * k4) when True, > (k2 * k3 * k4) when False"
            dop = conf[1] * conf[2] * conf[3]
            conf[5] = self._compute_k5(dop, conf[5])
            conf[3] = dop
            return [self._add_unit(k_with_type, k_meta) for k_with_type, k_meta in zip(conf, self.knob_list)]

        elif isinstance(conf_denorm, np.ndarray):
            assert ((len(conf_denorm) > 0) and (len(conf_denorm[0]) == len(self.knob_list))), \
                "number of columns of conf_denorm as a np.ndarray should match knob_list"
            conf = np.vectorize(
                pyfunc=lambda c, k: self._construct_knob(c, k), otypes="f")(conf_denorm, self.knob_list)
            # a special transformation for k4 and k6
            # k6: set spark.shuffle.sort.bypassMergeThreshold <= (k2 * k3 * k4) when True, > (k2 * k3 * k4) when False"
            conf[:, 5] = np.apply_along_axis(
                lambda c: self._compute_k5(dop=c[1] * c[2] * c[3], k5=c[5]), axis=1, arr=conf)
            conf[:, 3] = conf[:, 1] * conf[:, 2] * conf[:, 3]
            return np.vectorize(lambda c, k: self._add_unit(c, k))(conf, self.knob_list)

        elif isinstance(conf_denorm, DataFrame):
            assert (conf_denorm.shape[1] == len(self.knob_list)), \
                "number of columns of conf_denorm as a DataFrame should match knob_list"
            conf_denorm[:] = np.vectorize(
                lambda c, k: self._construct_knob(c, k), otypes="f")(conf_denorm, self.knob_list)
            conf = conf_denorm.copy()
            # a special transformation for k4 and k6
            # k6: set spark.shuffle.sort.bypassMergeThreshold <= (k2 * k3 * k4) when True, > (k2 * k3 * k4) when False"
            conf.iloc[:, 5] = conf.apply(lambda c: self._compute_k5(dop=c[1] * c[2] * c[3], k5=c[5]), axis=1)
            conf.iloc[:, 3] = conf.iloc[:, 1] * conf.iloc[:, 2] * conf.iloc[:, 3]
            conf[:] = np.vectorize(lambda c, k: self._add_unit(c, k))(conf, self.knob_list)
            return conf

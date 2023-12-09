from typing import Union, List, Iterable

import numpy as np
import pandas as pd

from . import Conf


class SparkConf(Conf):

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

        elif isinstance(conf_denorm, pd.DataFrame):
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

        else:
            raise Exception(f"unsupported type {type(conf_denorm)}")

    def deconstruct_configuration(self, conf: Union[pd.DataFrame, List]) -> Union[pd.DataFrame, List]:
        raise NotImplementedError
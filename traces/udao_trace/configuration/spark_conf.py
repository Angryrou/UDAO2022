from typing import Union, List, Iterable

import numpy as np
import pandas as pd
from scipy.stats import qmc

from . import Conf
from ..utils import VarTypes, ScaleTypes
from ..utils.logging import logger


class SparkConf(Conf):

    @staticmethod
    def _compute_k6(dop: int, k6: int) -> int:
        assert (k6 in (0, 1)), f"k6 value {k6} is not 0 or 1"
        return (dop - 1 if dop <= 200 else 200) if k6 == 1 else (dop + 1 if dop > 200 else 200)

    def construct_configuration(self, conf_denorm: Union[Iterable, List]) -> Union[Iterable, List[str]]:
        knob_list = self.knob_list
        log_mask = [k.scale == ScaleTypes.LOG for k in knob_list]
        linear_mask = [k.scale == ScaleTypes.LINEAR for k in knob_list]
        ctype_int_mask = np.array([k.ctype in (VarTypes.INT, VarTypes.CATEGORY, VarTypes.BOOL) for k in knob_list])
        ctype_bool_mask = np.array([k.ctype == VarTypes.BOOL for k in knob_list])
        factors = np.array([k.factor for k in knob_list])
        bases = np.array([k.base for k in knob_list]).astype(float)

        # convert from numeric values to str parameter values, with dependency resolved and unit added
        if isinstance(conf_denorm, List):
            conf = [self._construct_knob(k_denorm, k_meta) for k_denorm, k_meta in zip(conf_denorm, self.knob_list)]
            assert (len(conf) == len(self.knob_list)), "length of conf as a List should match knob_list"
            # a special transformation for k2
            conf[1] = conf[0] * conf[1]
            # a special transformation for k4
            conf[3] = conf[0] * conf[2] * conf[3]
            # a special transformation for k6
            # k6: set spark.shuffle.sort.bypassMergeThreshold <= dop when True, > dop when False"
            conf[5] = self._compute_k6(conf[3], conf[5])
            return [self._add_unit(k_with_type, k_meta) for k_with_type, k_meta in zip(conf, self.knob_list)]

        elif isinstance(conf_denorm, np.ndarray):
            assert ((len(conf_denorm) > 0) and (len(conf_denorm[0]) == len(self.knob_list))), \
                "number of columns of conf_denorm as a np.ndarray should match knob_list"
            conf_denorm[:, log_mask] = np.power(bases[log_mask], conf_denorm[:, log_mask]) * factors[log_mask]
            conf_denorm[:, linear_mask] *= factors[linear_mask]
            conf_denorm[:, ctype_int_mask] = np.round(conf_denorm[:, ctype_int_mask])
            conf = conf_denorm
            # a special transformation for k2
            conf[:, 1] = conf[:, 0] * conf[:, 1]
            # a special transformation for k4
            conf[:, 3] = conf[:, 0] * conf[:, 2] * conf[:, 3]
            # a special transformation for k6
            # k6: set spark.shuffle.sort.bypassMergeThreshold <= dop when True, > dop when False"
            conf[:, 5] = np.apply_along_axis(
                lambda c: self._compute_k6(dop=c[3], k6=c[5]), axis=1, arr=conf)
            # fixme: not handling categorical knobs
            return np.vectorize(lambda c, k: self._add_unit(c, k))(conf, self.knob_list)

        elif isinstance(conf_denorm, pd.DataFrame):
            assert (conf_denorm.shape[1] == len(self.knob_list)), \
                "number of columns of conf_denorm as a DataFrame should match knob_list"
            conf_denorm.iloc[:, log_mask] = np.power(bases[log_mask], conf_denorm.iloc[:, log_mask]) * factors[log_mask]
            conf_denorm.iloc[:, linear_mask] *= factors[linear_mask]
            conf_denorm.iloc[:, ctype_int_mask] = np.round(conf_denorm.iloc[:, ctype_int_mask])
            conf = conf_denorm
            # a special transformation for k2
            conf.iloc[:, 1] = conf.iloc[:, 0] * conf.iloc[:, 1]
            # a special transformation for k4
            conf.iloc[:, 3] = conf.iloc[:, 0] * conf.iloc[:, 2] * conf.iloc[:, 3]
            # a special transformation for k6
            # k6: set spark.shuffle.sort.bypassMergeThreshold <= dop when True, > dop when False"
            conf.iloc[:, 5] = conf.apply(lambda c: self._compute_k6(dop=c[3], k6=c[5]), axis=1)
            conf[:] = np.vectorize(lambda c, k: self._add_unit(c, k))(conf, self.knob_list)
            return conf

        else:
            raise Exception(f"unsupported type {type(conf_denorm)}")

    def deconstruct_configuration(self, conf: Union[pd.DataFrame, List]) -> Union[pd.DataFrame, List]:
        raise NotImplementedError

    def get_lhs_configurations(self, n_samples: int, seed: int) -> pd.DataFrame:
        assert n_samples > 1
        sampler = qmc.LatinHypercube(d=self.knob_num, seed=seed)
        conf_norm = sampler.random(n_samples)
        conf = self.construct_configuration_from_norm(conf_norm)
        unique_rows, counts = np.unique(conf, axis=0, return_counts=True)
        duplicated_rows = unique_rows[counts > 1]
        # Display the result
        if len(duplicated_rows) > 0:
            logger.warn("Duplicated Rows:", duplicated_rows)
        else:
            logger.debug("No duplicated rows.")
        conf_df = pd.DataFrame(unique_rows, columns=self.knob_names)
        conf_df.index = conf_df.apply(lambda x: self.conf2sign(x), axis=1)
        logger.debug(f"finished generated {len(conf_df)} configurations via LHS")
        return conf_df

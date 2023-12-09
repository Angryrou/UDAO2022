from pathlib import Path
from typing import Union, List, Iterable
import numpy as np
import pandas as pd

import pytest

from udao_trace.configuration import SparkConf


@pytest.fixture()
def sp() -> SparkConf:
    base_dir = Path(__file__).parent
    knob_meta_file = str(base_dir / "assets/spark_configuration_aqe_on.json")
    return SparkConf(knob_meta_file)


CONG_NORM1 = [0.0] * 19
CONG_NORM2 = [1.0] * 19
CONG_NORM3 = [0.0, 0.0, 1., 1 / 3, 0.4, 0., 1., 0.4, 0.4, 0.2, 0., 1 / 32, 23 / 48, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2]

CONF_DENORM1 = [2, 1, 4, 1, 0, 0, 0, 50, 0, 1, 0, 0, 2, 0, 20, 0, 0, 5, 1]
CONF_DENORM2 = [12, 5, 16, 4, 5, 1, 1, 75, 5, 6, 32, 32, 50, 4, 80, 4, 4, 35, 6]
CONF_DENORM3 = [2, 1, 16, 2, 2, 0, 1, 60, 2, 2, 0, 1, 25, 2, 50, 2, 2, 20, 2]

CONF1 = ["2g", "1", "4", "4", "12m", "200", "false", "0.5",
         "16MB", "0.1", "0MB", "0MB", "16", "64MB", "2", "32MB", "1MB", "0.05", "512KB"]
CONF2 = ["12g", "5", "16", "320", "384m", "200", "true", "0.75",
         "512MB", "0.6", "320MB", "320MB", "400", "1024MB", "8", "512MB", "16MB", "0.35", "3072KB"]
CONF3 = ["2g", "1", "16", "32", "48m", "200", "true", "0.6",
         "64MB", "0.2", "0MB", "10MB", "200", "256MB", "5", "128MB", "4MB", "0.2", "1024KB"]
HEADER = ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8",
          "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"]

class TestSparkProxy:

    @pytest.mark.parametrize(
        "conf_norm, expected_conf_denorm",
        [
            (CONG_NORM1, CONF_DENORM1),
            (CONG_NORM2, CONF_DENORM2),
            (CONG_NORM3, CONF_DENORM3),
            (
                np.array([CONG_NORM1, CONG_NORM2, CONG_NORM3]),
                np.array([CONF_DENORM1, CONF_DENORM2, CONF_DENORM3])
            ),
            (
                pd.DataFrame([CONG_NORM1, CONG_NORM2, CONG_NORM3], columns=HEADER).astype(float),
                pd.DataFrame([CONF_DENORM1, CONF_DENORM2, CONF_DENORM3], columns=HEADER).astype(float)
            )
        ]
    )
    def test_denormalize(
        self,
        sp: SparkConf,
        conf_norm: Union[List, Iterable],
        expected_conf_denorm: Union[List, Iterable]
    ) -> None:
        if isinstance(conf_norm, List):
            assert (sp.denormalize(conf_norm) == expected_conf_denorm)
        elif isinstance(conf_norm, np.ndarray):
            assert (np.array_equal(sp.denormalize(conf_norm), expected_conf_denorm))
        elif isinstance(conf_norm, pd.DataFrame):
            conf_denorm = sp.denormalize(conf_norm)
            pd.testing.assert_frame_equal(conf_denorm, expected_conf_denorm)
        else:
            raise Exception(f"unsupported type {type(conf_norm)}")

    @pytest.mark.parametrize(
        "conf_denorm, expected_conf",
        [
            (CONF_DENORM1, CONF1),
            (CONF_DENORM2, CONF2),
            (CONF_DENORM3, CONF3),
            (
                np.array([CONF_DENORM1, CONF_DENORM2, CONF_DENORM3]).astype(float),
                np.array([CONF1, CONF2, CONF3]).astype(str)
            ),
            (
                pd.DataFrame([CONF_DENORM1, CONF_DENORM2, CONF_DENORM3], columns=HEADER).astype(float),
                pd.DataFrame([CONF1, CONF2, CONF3], columns=HEADER).astype(str)
            )
        ]
    )
    def test_construct_configuration(
        self,
        sp: SparkConf,
        conf_denorm: Union[List, pd.DataFrame],
        expected_conf: Union[List, pd.DataFrame]
    ) -> None:
        if isinstance(conf_denorm, List):
            assert(sp.construct_configuration(conf_denorm) == expected_conf)
        elif isinstance(conf_denorm, np.ndarray):
            assert (np.array_equal(sp.construct_configuration(conf_denorm), expected_conf))
        elif isinstance(conf_denorm, pd.DataFrame):
            conf = sp.construct_configuration(conf_denorm)
            pd.testing.assert_frame_equal(conf, expected_conf)
        else:
            raise Exception(f"unsupported type {type(conf_denorm)}")

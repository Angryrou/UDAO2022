import json
from pathlib import Path
from typing import Union, List, Dict, Optional
from pandas import DataFrame

import pytest

from .. import SparkProxy, KnobMeta


@pytest.fixture()
def sp() -> SparkProxy:
    base_dir = Path(__file__).parent
    knob_meta_file = str(base_dir / "assets/spark_configuration_aqe_on.json")
    return SparkProxy(knob_meta_file)


class TestSparkProxy:

    @pytest.mark.parametrize(
        "conf_norm, expected_conf_denorm",
        [
            (
                [0.0] * 19,
                [2, 1, 8, 1, 0, 0, 0, 50,
                 0, 1, 0, 0, 2, 0, 20, 0, 0,
                 5, 1]
            ),
            (
                [1.0] * 19,
                [16, 5, 20, 4, 5, 1, 1, 75,
                 5, 6, 32, 32, 50, 4, 80, 4, 4,
                 35, 6]
            ),
            (
                [0.0, 0.0, 1, 1 / 3, 0.4, 0, 1, 0.4,
                 0.4, 0.2, 0, 1 / 32, 23 / 48, 0.5, 0.5, 0.5, 0.5,
                 0.5, 0.2],
                [2, 1, 20, 2, 2, 0, 1, 60,
                 2, 2, 0, 1, 25, 2, 50, 2, 2,
                 20, 2]
            )
        ]
    )
    def test_denormalize(
        self,
        sp: SparkProxy,
        conf_norm: Union[List, DataFrame],
        expected_conf_denorm: Union[List, DataFrame]
    ) -> None:
        assert (sp.denormalize(conf_norm) == expected_conf_denorm)

    @pytest.mark.parametrize(
        "conf_denorm, expected_conf",
        [
            (
                [2, 1, 8, 1, 0, 0, 0, 50,
                 0, 1, 0, 0, 2, 0, 20, 0, 0,
                 5, 1],
                ["2g", "1", "8", "8", "12m", "200", "false", "0.5",
                 "16MB", "0.1", "0MB", "0MB", "16", "64MB", "2", "32MB", "1MB",
                 "0.05", "0.5MB"]
            ),
            (
                [16, 5, 20, 4, 5, 1, 1, 80,
                 5, 6, 32, 32, 50, 4, 75, 4, 4,
                 35, 6],
                ["16g", "5", "20", "400", "384m", "200", "true", "0.75",
                 "512MB", "0.6", "320MB", "320MB", "400", "1024MB", "8", "512MB", "16MB",
                 "0.35", "3MB"]
            ),
            (
                [2, 1, 20, 2, 2, 0, 1, 60,
                 2, 2, 0, 1, 25, 2, 50, 2, 2,
                 20, 2],
                ["2g", "1", "20", "40", "48m", "200", "true", "0.6",
                 "64MB", "0.2", "0MB", "10MB", "200", "256MB", "5.0", "128MB", "4MB",
                 "0.2", "1MB"]

            )
        ]
    )
    def test_construct_configuration(
        self,
        sp: SparkProxy,
        conf_denorm: Union[List, DataFrame],
        expected_conf: Union[List, DataFrame]
    ) -> None:
        assert (sp.construct_configuration(conf_denorm) == expected_conf)

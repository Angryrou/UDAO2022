from typing import Dict, Iterable

import numpy as np
import pytest

from ...concepts.variable import (
    BoolVariable,
    EnumVariable,
    FloatVariable,
    IntegerVariable,
)
from ...solver.random_sampler import RandomSampler


class TestGridSearch:
    @pytest.mark.parametrize(
        "test_data, expected",
        [
            ({"variable": BoolVariable(), "n_samples": 3}, [0, 1, 1]),
            (
                {"variable": IntegerVariable(1, 7), "n_samples": 2},
                [5, 6],
            ),
            (
                {"variable": FloatVariable(2, 4), "n_samples": 5},
                [3.098, 3.430, 3.206, 3.090, 2.847],
            ),
            (
                {
                    "variable": EnumVariable([0, 4, 7, 10]),
                    "n_samples": 2,
                    "range": [0, 4, 7, 10],
                },
                [0, 10],
            ),
        ],
    )
    def test_random_sampler_single_variable(
        self, test_data: Dict, expected: Iterable
    ) -> None:
        solver = RandomSampler(
            RandomSampler.Params(n_samples_per_param=test_data["n_samples"], seed=0)
        )
        output = solver._get_input(
            variables=[test_data["variable"]],
        )
        np.testing.assert_allclose(output, [[e] for e in expected], rtol=1e-2)

    def test_random_sampler_multiple_variables(self) -> None:
        solver = RandomSampler(RandomSampler.Params(n_samples_per_param=3, seed=0))

        output = solver._get_input(
            variables=[BoolVariable(), IntegerVariable(1, 7)],
        )
        np.testing.assert_equal(
            output,
            [
                [0, 1],
                [1, 4],
                [1, 4],
            ],
        )

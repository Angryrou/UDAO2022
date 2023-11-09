from typing import Dict, Iterable

import numpy as np
import pytest

from ...solver.random_sampler import RandomSampler
from ...utils.parameters import VarTypes


class TestGridSearch:
    @pytest.mark.parametrize(
        "test_data, expected",
        [
            ({"type": VarTypes.BOOL, "n_samples": 3, "range": [0, 1]}, [0, 1, 1]),
            (
                {"type": VarTypes.INTEGER, "n_samples": 2, "range": [1, 7]},
                [5, 6],
            ),
            (
                {"type": VarTypes.FLOAT, "n_samples": 5, "range": [2, 4]},
                [3.098, 3.430, 3.206, 3.090, 2.847],
            ),
            (
                {"type": VarTypes.ENUM, "n_samples": 2, "range": [0, 4, 7, 10]},
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
            var_ranges=[test_data["range"]],
            var_types=[test_data["type"]],
        )
        np.testing.assert_allclose(output, [[e] for e in expected], rtol=1e-2)

    def test_random_sampler_multiple_variables(self) -> None:
        solver = RandomSampler(RandomSampler.Params(n_samples_per_param=3, seed=0))

        output = solver._get_input(
            var_ranges=[[0, 1], [1, 7]],
            var_types=[VarTypes.BOOL, VarTypes.INTEGER],
        )
        np.testing.assert_equal(
            output,
            [
                [0, 1],
                [1, 4],
                [1, 4],
            ],
        )

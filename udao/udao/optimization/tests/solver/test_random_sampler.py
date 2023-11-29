from typing import Dict, Iterable

import numpy as np
import pytest
import torch as th

from ...concepts.objective import Objective
from ...concepts.variable import (
    BoolVariable,
    EnumVariable,
    FloatVariable,
    IntegerVariable,
)
from ...solver.random_sampler_solver import RandomSampler
from ...utils.moo_utils import Point


class TestRandomSampler:
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
            variables={"v1": test_data["variable"]},
        )
        np.testing.assert_allclose(
            [[o] for o in output["v1"]], [[e] for e in expected], rtol=1e-2
        )

    def test_random_sampler_multiple_variables(self) -> None:
        solver = RandomSampler(RandomSampler.Params(n_samples_per_param=3, seed=0))

        output = solver._get_input(
            variables={"v1": BoolVariable(), "v2": IntegerVariable(1, 7)},
        )
        expected_array = np.array(
            [
                [0, 1],
                [1, 4],
                [1, 4],
            ]
        )
        np.testing.assert_equal(
            output, {"v1": expected_array[:, 0], "v2": expected_array[:, 1]}
        )

    def test_solve(self) -> None:
        solver = RandomSampler(RandomSampler.Params(n_samples_per_param=30, seed=0))

        def obj1_func(x: Dict, input_parameters: Dict) -> th.Tensor:
            return th.tensor(x["v1"] + x["v2"])

        objective = Objective("obj1", "MAX", obj1_func)
        variables = {"v1": BoolVariable(), "v2": IntegerVariable(1, 7)}
        point = solver.solve(objective=objective, variables=variables)
        assert point == Point(np.array([8]), {"v1": 1, "v2": 7})

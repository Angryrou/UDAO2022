from typing import Dict, Iterable

import numpy as np
import pytest
import torch as th

from ... import concepts as co
from ...solver.random_sampler_solver import RandomSampler


class TestRandomSampler:
    @pytest.mark.parametrize(
        "test_data, expected",
        [
            ({"variable": co.BoolVariable(), "n_samples": 3}, [0, 1, 1]),
            (
                {"variable": co.IntegerVariable(1, 7), "n_samples": 2},
                [5, 6],
            ),
            (
                {"variable": co.FloatVariable(2, 4), "n_samples": 5},
                [3.098, 3.430, 3.206, 3.090, 2.847],
            ),
            (
                {
                    "variable": co.EnumVariable([0, 4, 7, 10]),
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
            variables={"v1": co.BoolVariable(), "v2": co.IntegerVariable(1, 7)},
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

        def obj1_func(
            input_variables: co.InputVariables,
            input_parameters: co.InputParameters = None,
        ) -> th.Tensor:
            return th.tensor(input_variables["v1"] + input_variables["v2"])

        objective = co.Objective("obj1", "MAX", obj1_func)
        variables = {"v1": co.BoolVariable(), "v2": co.IntegerVariable(1, 7)}
        soo_obj, soo_vars = solver.solve(objective=objective, variables=variables)
        assert soo_obj == 8
        assert soo_vars == {"v1": 1, "v2": 7}

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
from ...solver.grid_search_solver import GridSearch
from ...utils.moo_utils import Point


class TestGridSearch:
    @pytest.mark.parametrize(
        "test_data, expected",
        [
            ({"variable": BoolVariable(), "n_grids": 1}, [0]),
            ({"variable": BoolVariable(), "n_grids": 2}, [0, 1]),
            ({"variable": BoolVariable(), "n_grids": 3}, [0, 1]),
            (
                {"variable": IntegerVariable(1, 7), "n_grids": 5},
                [1, 2, 4, 6, 7],
            ),
            (
                {"variable": IntegerVariable(1, 7), "n_grids": 8},
                [1, 2, 3, 4, 5, 6, 7],
            ),
            (
                {"variable": FloatVariable(2, 4), "n_grids": 5},
                [2, 2.5, 3, 3.5, 4],
            ),
            (
                {"variable": EnumVariable([0, 4, 7, 10]), "n_grids": 2},
                [0, 4, 7, 10],
            ),
        ],
    )
    def test_grid_search_get_single_variable(
        self, test_data: Dict, expected: Iterable
    ) -> None:
        solver = GridSearch(GridSearch.Params(n_grids_per_var=[test_data["n_grids"]]))
        output = solver._get_input(
            variables={"variable": test_data["variable"]},
        )
        np.testing.assert_equal(output, {"variable": np.array([e for e in expected])})

    def test_grid_search_get_multiple_variables(self) -> None:
        solver = GridSearch(GridSearch.Params(n_grids_per_var=[2, 7]))
        output = solver._get_input(
            variables={"v1": BoolVariable(), "v2": IntegerVariable(1, 7)},
        )
        expected_array = np.array(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [0, 6],
                [0, 7],
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [1, 6],
                [1, 7],
            ]
        ).T
        np.testing.assert_equal(
            output, {"v1": expected_array[0], "v2": expected_array[1]}
        )

    def test_solve(self) -> None:
        solver = GridSearch(GridSearch.Params(n_grids_per_var=[2, 7]))

        def obj1_func(x: Dict, input_parameters: Dict) -> th.Tensor:
            return th.tensor(x["v1"] + x["v2"])

        objective = Objective("obj1", "MAX", obj1_func)
        variables = {"v1": BoolVariable(), "v2": IntegerVariable(1, 7)}
        point = solver.solve(objective=objective, variables=variables)
        assert point == Point(np.array([8]), {"v1": 1, "v2": 7})

from typing import Dict, Iterable

import numpy as np
import pytest

from ...concepts.variable import (
    BoolVariable,
    EnumVariable,
    FloatVariable,
    IntegerVariable,
)
from ...solver.grid_search import GridSearch


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
    def test_grid_search_single_variable(
        self, test_data: Dict, expected: Iterable
    ) -> None:
        solver = GridSearch(GridSearch.Params(n_grids_per_var=[test_data["n_grids"]]))
        output = solver._get_input(
            variables=[test_data["variable"]],
        )
        np.testing.assert_equal(output, [[e] for e in expected])

    def test_grid_search_multiple_variables(self) -> None:
        solver = GridSearch(GridSearch.Params(n_grids_per_var=[2, 7]))
        output = solver._get_input(
            variables=[BoolVariable(), IntegerVariable(1, 7)],
        )
        np.testing.assert_equal(
            output,
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
            ],
        )

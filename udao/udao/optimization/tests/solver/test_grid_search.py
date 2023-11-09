from typing import Dict, Iterable

import numpy as np
import pytest

from ...solver.grid_search import GridSearch
from ...utils.parameters import VarTypes


class TestGridSearch:
    @pytest.mark.parametrize(
        "test_data, expected",
        [
            ({"type": VarTypes.BOOL, "n_grids": 1, "range": [0, 1]}, [0]),
            ({"type": VarTypes.BOOL, "n_grids": 2, "range": [0, 1]}, [0, 1]),
            ({"type": VarTypes.BOOL, "n_grids": 3, "range": [0, 1]}, [0, 1]),
            (
                {"type": VarTypes.INTEGER, "n_grids": 5, "range": [1, 7]},
                np.linspace(1, 7, num=5, endpoint=True),
            ),
            (
                {"type": VarTypes.INTEGER, "n_grids": 8, "range": [1, 7]},
                [1, 2, 3, 4, 5, 6, 7],
            ),
            (
                {"type": VarTypes.FLOAT, "n_grids": 5, "range": [2, 4]},
                [2, 2.5, 3, 3.5, 4],
            ),
            (
                {"type": VarTypes.ENUM, "n_grids": 2, "range": [0, 4, 7, 10]},
                [0, 4, 7, 10],
            ),
        ],
    )
    def test_grid_search_single_variable(
        self, test_data: Dict, expected: Iterable
    ) -> None:
        solver = GridSearch(GridSearch.Params(n_grids_per_var=[test_data["n_grids"]]))
        output = solver._get_input(
            var_ranges=[test_data["range"]],
            var_types=[test_data["type"]],
        )
        np.testing.assert_equal(output, [[e] for e in expected])

    def test_grid_search_multiple_variables(self) -> None:
        solver = GridSearch(GridSearch.Params(n_grids_per_var=[2, 7]))
        output = solver._get_input(
            var_ranges=[[0, 1], [1, 7]],
            var_types=[VarTypes.BOOL, VarTypes.INTEGER],
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

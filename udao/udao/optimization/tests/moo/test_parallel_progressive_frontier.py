from typing import Optional

import numpy as np
import pytest
import torch as th

from ...concepts import FloatVariable, IntegerVariable, Objective
from ...moo.progressive_frontier import ParallelProgressiveFrontier
from ...utils.moo_utils import Point, Rectangle


def obj1(x: th.Tensor, wl_id: Optional[str]) -> th.Tensor:
    return th.reshape(x[:, 0] ** 2, (-1, 1))


def obj2(x: th.Tensor, wl_id: Optional[str]) -> th.Tensor:
    return th.reshape(x[:, 1] ** 2, (-1, 1))


def complex_obj1(x: th.Tensor, wl_id: Optional[str]) -> th.Tensor:
    return th.reshape(x[:, 0] ** 2 - x[:, 1] ** 2, (-1, 1))


def complex_obj2(x: th.Tensor, wl_id: Optional[str]) -> th.Tensor:
    return th.reshape(x[:, 0] ** 2 + x[:, 1] ** 2, (-1, 1))


@pytest.fixture
def ppf() -> ParallelProgressiveFrontier:
    objectives = [
        Objective("obj1", "MAX", obj1),  # type: ignore
        Objective("obj2", "MIN", obj2),  # type: ignore
    ]
    variables = [FloatVariable(0, 1), IntegerVariable(1, 7)]

    ppf = ParallelProgressiveFrontier(
        variables=variables,
        objectives=objectives,
        solver_params={
            "learning_rate": 0.01,
            "weight_decay": 0,
            "max_iters": 100,
            "patient": 10,
            "multistart": 2,
            "processes": 1,
            "stress": 0.5,
            "seed": 0,
        },
        constraints=[],
        accurate=True,
        std_func=None,
        alpha=0.1,
        precision_list=[2, 2],
    )
    ppf.mogd.device = th.device("cpu")
    return ppf


class TestParallelProgressiveFrontier:
    def test_create_grid_cells(self, ppf: ParallelProgressiveFrontier) -> None:
        utopia = Point(np.array([0, 2, 0]))
        nadir = Point(np.array([4, 10, 1]))
        grid_rectangles = ppf._create_grid_cells(utopia, nadir, 2, 3)

        assert len(grid_rectangles) == 8
        expected = [
            Rectangle(
                utopia=Point(objs=np.array([0.0, 2.0, 0.0])),
                nadir=Point(objs=np.array([2.0, 6.0, 0.5])),
            ),
            Rectangle(
                utopia=Point(objs=np.array([0.0, 2.0, 0.5])),
                nadir=Point(objs=np.array([2.0, 6.0, 1.0])),
            ),
            Rectangle(
                utopia=Point(objs=np.array([0.0, 6.0, 0.0])),
                nadir=Point(objs=np.array([2.0, 10.0, 0.5])),
            ),
            Rectangle(
                utopia=Point(objs=np.array([0.0, 6.0, 0.5])),
                nadir=Point(objs=np.array([2.0, 10.0, 1.0])),
            ),
            Rectangle(
                utopia=Point(objs=np.array([2.0, 2.0, 0.0])),
                nadir=Point(objs=np.array([4.0, 6.0, 0.5])),
            ),
            Rectangle(
                utopia=Point(objs=np.array([2.0, 2.0, 0.5])),
                nadir=Point(objs=np.array([4.0, 6.0, 1.0])),
            ),
            Rectangle(
                utopia=Point(objs=np.array([2.0, 6.0, 0.0])),
                nadir=Point(objs=np.array([4.0, 10.0, 0.5])),
            ),
            Rectangle(
                utopia=Point(objs=np.array([2.0, 6.0, 0.5])),
                nadir=Point(objs=np.array([4.0, 10.0, 1.0])),
            ),
        ]
        for i, rect in enumerate(expected):
            assert rect == grid_rectangles[i]

    def test_solve_with_two_objectives(self, ppf: ParallelProgressiveFrontier) -> None:
        objectives, variables = ppf.solve("1", n_grids=2, max_iters=5)
        assert objectives is not None
        np.testing.assert_array_equal(objectives, [[-1, 0]])
        assert variables is not None
        np.testing.assert_array_equal(variables, [[1, 1]])

    def test_solve_with_three_objectives(
        self, ppf: ParallelProgressiveFrontier
    ) -> None:
        objectives = [
            Objective("obj1", "MAX", obj1),
            Objective("obj2", "MAX", complex_obj1),
            Objective("obj3", "MAX", complex_obj2),
        ]
        ppf.objectives = objectives
        ppf.mogd.objectives = objectives
        obj_values, var_values = ppf.solve("1", n_grids=2, max_iters=2)
        assert obj_values is not None
        np.testing.assert_array_almost_equal(
            obj_values,
            np.array([[-1.0, -0.40966392, -1.59033608], [-1, -1, -1], [-1, 0, -2]]),
        )
        assert var_values is not None
        np.testing.assert_array_equal(
            var_values, np.array([[1.0, 5.61], [1, 1], [1, 7]])
        )

from typing import Dict

import numpy as np
import pytest
import torch as th

from ....data.handler.data_processor import DataProcessor
from ....model.utils.utils import set_deterministic_torch
from ...concepts import FloatVariable, IntegerVariable, Objective, Variable
from ...concepts.utils import ModelComponent
from ...moo.progressive_frontier import ParallelProgressiveFrontier
from ...utils.moo_utils import Point, Rectangle
from .conftest import ComplexObj2, ObjModel1, ObjModel2


@pytest.fixture
def ppf(data_processor: DataProcessor) -> ParallelProgressiveFrontier:
    objectives = [
        Objective("obj1", "MIN", ModelComponent(data_processor, ObjModel1())),
        Objective("obj2", "MIN", ModelComponent(data_processor, ObjModel2())),
    ]
    variables: Dict[str, Variable] = {
        "v1": FloatVariable(0, 1),
        "v2": IntegerVariable(1, 7),
    }

    ppf = ParallelProgressiveFrontier(
        variables=variables,
        objectives=objectives,
        solver_params={
            "learning_rate": 0.1,
            "weight_decay": 0,
            "max_iters": 100,
            "patient": 10,
            "multistart": 5,
            "stress": 10,
            "seed": 0,
        },
        processes=1,
        constraints=[],
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

    def test_solve_with_two_objectives(
        self, ppf: ParallelProgressiveFrontier, data_processor: DataProcessor
    ) -> None:
        set_deterministic_torch()
        objectives, variables = ppf.solve(
            n_grids=2,
            max_iters=4,
            input_parameters={
                "embedding_input": 1,
                "objective_input": 1,
            },
        )
        assert objectives is not None
        np.testing.assert_array_equal(objectives, [[0, 0]])
        assert variables is not None
        assert variables[0] == {"v1": 0.0, "v2": 1.0}

    def test_solve_with_three_objectives(
        self, ppf: ParallelProgressiveFrontier, data_processor: DataProcessor
    ) -> None:
        set_deterministic_torch()
        objectives = [
            Objective("obj1", "MAX", ModelComponent(data_processor, ObjModel1())),
            Objective("obj2", "MAX", ModelComponent(data_processor, ObjModel2())),
            Objective("obj3", "MAX", ModelComponent(data_processor, ComplexObj2())),
        ]
        ppf.objectives = objectives
        obj_values, var_values = ppf.solve(
            n_grids=2,
            max_iters=2,
            input_parameters={
                "embedding_input": 1,
                "objective_input": 1,
            },
        )
        assert obj_values is not None
        np.testing.assert_array_almost_equal(obj_values, [[-1.0, -1.0, -2.0]])
        assert var_values[0] == {"v1": 1.0, "v2": 7.0}

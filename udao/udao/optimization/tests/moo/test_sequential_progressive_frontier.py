from typing import Dict

import numpy as np
import pytest
import torch as th

from ....data.handler.data_processor import DataProcessor
from ....model.utils.utils import set_deterministic_torch
from ...concepts import (
    FloatVariable,
    IntegerVariable,
    ModelComponent,
    Objective,
    Variable,
)
from ...moo.progressive_frontier import SequentialProgressiveFrontier
from ...utils.moo_utils import Point
from .conftest import ObjModel1, ObjModel2


@pytest.fixture
def progressive_frontier(
    data_processor: DataProcessor,
) -> SequentialProgressiveFrontier:
    objectives = [
        Objective(
            "obj1",
            "MIN",
            ModelComponent(data_processor, ObjModel1()),
        ),
        Objective(
            "obj2",
            "MIN",
            ModelComponent(data_processor, ObjModel2()),
        ),
    ]
    variables: Dict[str, Variable] = {
        "v1": FloatVariable(0, 1),
        "v2": IntegerVariable(1, 7),
    }

    spf = SequentialProgressiveFrontier(
        variables=variables,
        objectives=objectives,
        solver_params={
            "learning_rate": 0.1,
            "weight_decay": 0.1,
            "max_iters": 100,
            "patience": 10,
            "multistart": 2,
            "objective_stress": 0.5,
            "seed": 0,
        },
        constraints=[],
    )
    spf.mogd.device = th.device("cpu")
    return spf


class TestProgressiveFrontier:
    def test__get_corner_points(
        self, progressive_frontier: SequentialProgressiveFrontier
    ) -> None:
        utopia = Point(np.array([1, 0.3]))
        nadir = Point(np.array([5, 10]))
        corner_points = progressive_frontier._get_corner_points(utopia, nadir)
        # 1-------3#
        #         #
        # 0-------2#
        expected_points = [
            Point(np.array([1.0, 0.3])),
            Point(np.array([1.0, 10.0])),
            Point(np.array([5.0, 0.3])),
            Point(np.array([5.0, 10.0])),
        ]
        assert all(c == e for c, e in zip(corner_points, expected_points))

    def test__generate_sub_rectangles_bad(
        self, progressive_frontier: SequentialProgressiveFrontier
    ) -> None:
        utopia = Point(np.array([1, 0.3]))
        nadir = Point(np.array([5, 10]))
        middle = Point((utopia.objs + nadir.objs) / 2)

        rectangles = progressive_frontier.generate_sub_rectangles(
            utopia, nadir, middle, successful=False
        )
        ############
        #  0 |  1  #
        ############
        #  - |  -  #
        ############
        assert len(rectangles) == 2
        assert rectangles[0].utopia == Point(np.array([1.0, 5.15]))
        assert rectangles[0].nadir == Point(np.array([3.0, 10]))
        assert rectangles[1].utopia == Point(np.array([3.0, 5.15]))
        assert rectangles[1].nadir == Point(np.array([5.0, 10]))

    def test__generate_sub_rectangles_good(
        self, progressive_frontier: SequentialProgressiveFrontier
    ) -> None:
        utopia = Point(np.array([1, 0.3]))
        nadir = Point(np.array([5, 10]))
        middle = Point((utopia.objs + nadir.objs) / 2)

        rectangles = progressive_frontier.generate_sub_rectangles(utopia, nadir, middle)
        ############
        #  1 |  _  #
        ############
        #  0 |  2  #
        ############
        assert len(rectangles) == 3
        assert rectangles[0].utopia == Point(np.array([1.0, 0.3]))
        assert rectangles[0].nadir == Point(np.array([3.0, 5.15]))
        assert rectangles[1].utopia == Point(np.array([1.0, 5.15]))
        assert rectangles[1].nadir == Point(np.array([3.0, 10.0]))
        assert rectangles[2].utopia == Point(np.array([3.0, 0.3]))
        assert rectangles[2].nadir == Point(np.array([5.0, 5.15]))

    def test_get_utopia_and_nadir(
        self, progressive_frontier: SequentialProgressiveFrontier
    ) -> None:
        points = [
            Point(np.array([1, 5]), {"v1": 0.2, "v2": 1}),
            Point(np.array([3, 10]), {"v1": 0.8, "v2": 6}),
            Point(np.array([5, 0.3]), {"v1": 0.5, "v2": 3}),
        ]
        utopia, nadir = progressive_frontier.get_utopia_and_nadir(points)
        np.testing.assert_array_equal(utopia.objs, np.array([1, 0.3]))
        np.testing.assert_array_equal(nadir.objs, np.array([5, 10]))

    def test_solve(self, progressive_frontier: SequentialProgressiveFrontier) -> None:
        objectives, variables = progressive_frontier.solve(
            n_probes=10,
            input_parameters={
                "embedding_input": 1,
                "objective_input": 1,
            },
        )
        assert objectives is not None
        np.testing.assert_array_equal(objectives, [[0, 0]])
        assert variables[0] == {"v1": 0.0, "v2": 1.0}

    def test_get_utopia_and_nadir_raises_when_no_points(
        self, progressive_frontier: SequentialProgressiveFrontier
    ) -> None:
        with pytest.raises(ValueError):
            progressive_frontier.get_utopia_and_nadir([])

    def test_get_utopia_and_nadir_raises_when_inconsistent_points(
        self, progressive_frontier: SequentialProgressiveFrontier
    ) -> None:
        with pytest.raises(Exception):
            progressive_frontier.get_utopia_and_nadir(
                [
                    Point(np.array([1, 5]), {"v1": 0.2, "v2": 1}),
                    Point(np.array([3, 10]), {"v1": 0.8, "v2": 6}),
                    Point(np.array([5]), {"v1": 0.5, "v2": 3}),
                ]
            )

    def test_get_anchor_points(
        self, progressive_frontier: SequentialProgressiveFrontier
    ) -> None:
        set_deterministic_torch()
        anchor_point = progressive_frontier.get_anchor_point(
            input_parameters={
                "embedding_input": 1,
                "objective_input": 1,
            },
            obj_ind=0,
            anchor_option="2_step",
        )
        # assert anchor_point == Point(np.array([0, 0.6944444]), {"v1": 1, "v2": 6})
        np.testing.assert_array_almost_equal(
            anchor_point.objs, np.array([0.0, 0.6944444])
        )
        assert anchor_point.vars == {"v1": 0.0, "v2": 6.0}
        anchor_point = progressive_frontier.get_anchor_point(
            input_parameters={
                "embedding_input": 1,
                "objective_input": 1,
            },
            obj_ind=1,
            anchor_option="2_step",
        )
        np.testing.assert_array_almost_equal(
            anchor_point.objs, np.array([0.29689768, 0.0])
        )
        assert anchor_point.vars is not None

        assert anchor_point.vars == {"v1": 0.5448831915855408, "v2": 1.0}

import numpy as np
import pytest
import torch as th

from ...concepts import FloatVariable, IntegerVariable, Objective
from ...moo.progressive_frontier import SequentialProgressiveFrontier
from ...utils.moo_utils import Point


@pytest.fixture
def progressive_frontier() -> SequentialProgressiveFrontier:
    objectives = [
        Objective(
            "obj1",
            "MAX",
            lambda x, wl_id: th.reshape(x[:, 0] ** 2, (-1, 1)),  # type: ignore
        ),
        Objective(
            "obj2",
            "MIN",
            lambda x, wl_id: th.reshape(x[:, 1] ** 2, (-1, 1)),  # type: ignore
        ),
    ]
    variables = [FloatVariable(0, 1), IntegerVariable(1, 7)]

    return SequentialProgressiveFrontier(
        variables=variables,
        objectives=objectives,
        solver_params={
            "learning_rate": 0.01,
            "weight_decay": 0.1,
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
            Point(np.array([1, 5]), np.array([0.2, 1])),
            Point(np.array([3, 10]), np.array([0.8, 6])),
            Point(np.array([5, 0.3]), np.array([0.5, 3])),
        ]
        utopia, nadir = progressive_frontier.get_utopia_and_nadir(points)
        np.testing.assert_array_equal(utopia.objs, np.array([1, 0.3]))
        np.testing.assert_array_equal(nadir.objs, np.array([5, 10]))

    def test_solve(self, progressive_frontier: SequentialProgressiveFrontier) -> None:
        objectives, variables = progressive_frontier.solve("1", n_probes=10)
        assert objectives is not None
        np.testing.assert_array_equal(objectives, [[-1, 0]])
        assert variables is not None
        np.testing.assert_array_equal(variables, [[1, 1]])

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
                    Point(np.array([1, 5]), np.array([0.2, 1])),
                    Point(np.array([3]), np.array([0.8, 6])),
                    Point(np.array([5, 0.3]), np.array([0.5, 3])),
                ]
            )

    def test_get_anchor_points(
        self, progressive_frontier: SequentialProgressiveFrontier
    ) -> None:
        anchor_point = progressive_frontier.get_anchor_point(
            wl_id="1", obj_ind=0, anchor_option="2_step"
        )
        assert anchor_point == Point(np.array([-1, 0]), np.array([1, 1]))
        anchor_point = progressive_frontier.get_anchor_point(
            wl_id="1", obj_ind=1, anchor_option="2_step"
        )
        np.testing.assert_array_almost_equal(
            anchor_point.objs, np.array([-0.82810003, 0.0])
        )
        assert anchor_point.vars is not None
        np.testing.assert_array_almost_equal(
            anchor_point.vars,
            np.array([0.91, 1.0]),
        )

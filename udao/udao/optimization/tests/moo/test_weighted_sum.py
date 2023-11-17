from typing import Any, Optional

import numpy as np
import pytest
import torch as th

from ...concepts import BoolVariable, Constraint, IntegerVariable, Objective
from ...moo.weighted_sum import WeightedSum
from ...solver.base_solver import BaseSolver
from ...solver.grid_search import GridSearch
from ...solver.random_sampler import RandomSampler
from ...utils.exceptions import NoSolutionError


class TestWeightedSum:
    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearch(GridSearch.Params(n_grids_per_var=[2, 7])),
            RandomSampler(RandomSampler.Params(n_samples_per_param=30, seed=0)),
        ],
    )
    def test_solve_without_wl_id(self, inner_solver: BaseSolver) -> None:
        """solve a dummy minimization problem with 2 objectives and 1 constraint"""
        ws_pairs = np.array([[0.3, 0.7], [0.6, 0.4]])
        objectives = [
            Objective(
                "obj1",
                function=lambda x, **kw: th.tensor(x[:, 0]),
                direction_type="MIN",
            ),
            Objective(
                "obj2",
                function=lambda x, **kw: th.tensor((x[:, 0] + x[:, 1]) / 10),
                direction_type="MIN",
            ),
        ]
        constraints = [
            Constraint(function=lambda x, **kw: x[:, 0] + x[:, 1] - 2, type=">=")
        ]
        ws_algo = WeightedSum(
            inner_solver=inner_solver,
            ws_pairs=ws_pairs,
            objectives=objectives,
            constraints=constraints,
        )
        po_objs, po_vars = ws_algo.solve(
            wl_id=None, variables=[BoolVariable(), IntegerVariable(1, 7)]
        )
        np.testing.assert_equal(po_objs, np.array([[0, 0.2]]))
        np.testing.assert_equal(po_vars, np.array([[0, 2]]))

    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearch(GridSearch.Params(n_grids_per_var=[2, 7])),
            RandomSampler(RandomSampler.Params(n_samples_per_param=30, seed=0)),
        ],
    )
    def test_solve_with_wl_id(self, inner_solver: BaseSolver) -> None:
        """solve a dummy minimization problem with 2 objectives and 1 constraint"""
        ws_pairs = np.array([[0.3, 0.7], [0.6, 0.4]])

        def obj_func1(x: Any, wl_id: Optional[str] = None) -> th.Tensor:
            if not wl_id:
                return th.tensor(x[:, 0])
            else:
                return th.tensor(x[:, 0] + 1)

        def obj_func2(x: np.ndarray, wl_id: Optional[str] = None) -> th.Tensor:
            if not wl_id:
                return th.tensor((x[:, 0] + x[:, 1]) / 10)
            else:
                return th.tensor((x[:, 0] + x[:, 1]) / 10 + 1)

        objectives = [
            Objective(
                "obj1",
                function=obj_func1,
                direction_type="MIN",
            ),
            Objective("obj2", function=obj_func2, direction_type="MIN"),
        ]
        constraints = [
            Constraint(
                function=lambda x, wl_id: x[:, 0] + x[:, 1] - 3
                if wl_id
                else x[:, 0] + x[:, 1] - 2,
                type=">=",
            )
        ]

        ws_algo = WeightedSum(
            inner_solver=inner_solver,
            ws_pairs=ws_pairs,
            objectives=objectives,
            constraints=constraints,
        )
        po_objs, po_vars = ws_algo.solve(
            wl_id="1", variables=[BoolVariable(), IntegerVariable(1, 7)]
        )

        np.testing.assert_equal(po_objs, np.array([[1, 1.3]]))
        np.testing.assert_equal(po_vars, np.array([[0, 3]]))

    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearch(GridSearch.Params(n_grids_per_var=[2, 7])),
            RandomSampler(RandomSampler.Params(n_samples_per_param=10, seed=0)),
        ],
    )
    def test_ws_raises_no_solution(self, inner_solver: BaseSolver) -> None:
        ws_pairs = np.array([[0.3, 0.7], [0.6, 0.4]])
        objectives = [
            Objective(
                "obj1",
                function=lambda x, **kw: th.tensor(x[:, 0]),
                direction_type="MIN",
            ),
            Objective(
                "obj2",
                function=lambda x, **kw: th.tensor((x[:, 0] + x[:, 1]) / 10),
                direction_type="MIN",
            ),
        ]
        constraints = [
            Constraint(function=lambda x, **kw: x[:, 0] + x[:, 1] - 10, type=">=")
        ]
        ws_algo = WeightedSum(
            inner_solver=inner_solver,
            ws_pairs=ws_pairs,
            objectives=objectives,
            constraints=constraints,
        )
        with pytest.raises(NoSolutionError):
            ws_algo.solve(wl_id=None, variables=[BoolVariable(), IntegerVariable(1, 7)])

    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearch(GridSearch.Params(n_grids_per_var=[2, 7])),
            RandomSampler(RandomSampler.Params(n_samples_per_param=30, seed=0)),
        ],
    )
    def test_works_with_three_objectives(self, inner_solver: BaseSolver) -> None:
        ws_pairs = np.array([[0.3, 0.5, 0.2], [0.6, 0.3, 0.1]])
        objectives = [
            Objective(
                "obj1",
                function=lambda x, **kw: th.tensor(x[:, 0]),
                direction_type="MIN",
            ),
            Objective(
                "obj2",
                function=lambda x, **kw: th.tensor(x[:, 1]),
                direction_type="MIN",
            ),
            Objective(
                "obj3",
                function=lambda x, **kw: th.tensor((x[:, 0] + x[:, 1]) / 10),
                direction_type="MIN",
            ),
        ]
        constraints = [
            Constraint(function=lambda x, **kw: x[:, 0] + x[:, 1] - 3, type=">=")
        ]
        ws_algo = WeightedSum(
            inner_solver=inner_solver,
            ws_pairs=ws_pairs,
            objectives=objectives,
            constraints=constraints,
        )
        po_objs, po_vars = ws_algo.solve(
            wl_id="1", variables=[BoolVariable(), IntegerVariable(1, 7)]
        )

        np.testing.assert_equal(po_objs, np.array([[0, 3, 0.3]]))
        np.testing.assert_equal(po_vars, np.array([[0, 3]]))

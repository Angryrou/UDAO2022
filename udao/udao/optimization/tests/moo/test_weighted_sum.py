from typing import Any, Dict, Optional

import numpy as np
import pytest
import torch as th

from ...concepts import BoolVariable, Constraint, IntegerVariable, Objective
from ...moo.weighted_sum import WeightedSum
from ...solver.base_solver import BaseSolver
from ...solver.grid_search_solver import GridSearch
from ...solver.random_sampler_solver import RandomSampler
from ...utils.exceptions import NoSolutionError


class TestWeightedSum:
    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearch(GridSearch.Params(n_grids_per_var=[2, 7])),
            RandomSampler(RandomSampler.Params(n_samples_per_param=30, seed=0)),
        ],
    )
    def test_solve_without_input_parameters(self, inner_solver: BaseSolver) -> None:
        """solve a dummy minimization problem with 2 objectives and 1 constraint"""
        ws_pairs = np.array([[0.3, 0.7], [0.6, 0.4]])
        objectives = [
            Objective(
                "obj1",
                function=lambda x, **kw: th.tensor(x["v1"]),
                direction_type="MIN",
            ),
            Objective(
                "obj2",
                function=lambda x, **kw: th.tensor((x["v1"] + x["v2"]) / 10),
                direction_type="MIN",
            ),
        ]
        constraints = [
            Constraint(function=lambda x, **kw: x["v1"] + x["v2"] - 2, type=">=")
        ]

        ws_algo = WeightedSum(
            inner_solver=inner_solver,
            ws_pairs=ws_pairs,
            objectives=objectives,
            constraints=constraints,
        )
        po_objs, po_vars = ws_algo.solve(
            variables={"v1": BoolVariable(), "v2": IntegerVariable(1, 7)}
        )
        np.testing.assert_equal(po_objs, np.array([[0, 0.2]]))
        np.testing.assert_equal(po_vars, np.array({"v1": 0, "v2": 2}))

    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearch(GridSearch.Params(n_grids_per_var=[2, 7])),
            RandomSampler(RandomSampler.Params(n_samples_per_param=30, seed=0)),
        ],
    )
    def test_solve_with_input_parameters(self, inner_solver: BaseSolver) -> None:
        """solve a dummy minimization problem with 2 objectives and 1 constraint"""
        ws_pairs = np.array([[0.3, 0.7], [0.6, 0.4]])

        def obj_func1(x: Any, input_parameters: Optional[Dict] = None) -> th.Tensor:
            if not input_parameters:
                return th.tensor(x["v1"])
            else:
                return th.tensor(x["v1"] + input_parameters["count"])

        def obj_func2(
            x: np.ndarray, input_parameters: Optional[Dict] = None
        ) -> th.Tensor:
            if not input_parameters:
                return th.tensor((x["v1"] + x["v2"]) / 10)
            else:
                return th.tensor((x["v1"] + x["v2"]) / 10 + input_parameters["count"])

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
                function=lambda x, input_parameters: x["v1"] + x["v2"] - 3
                if input_parameters
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
            input_parameters={"count": 1},
            variables={"v1": BoolVariable(), "v2": IntegerVariable(1, 7)},
        )

        np.testing.assert_equal(po_objs, np.array([[1, 1.3]]))
        np.testing.assert_equal(po_vars, np.array([{"v1": 0, "v2": 3}]))

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
                function=lambda x, **kw: th.tensor(x["v1"]),
                direction_type="MIN",
            ),
            Objective(
                "obj2",
                function=lambda x, **kw: th.tensor((x["v1"] + x["v2"]) / 10),
                direction_type="MIN",
            ),
        ]
        constraints = [
            Constraint(function=lambda x, **kw: x["v1"] + x["v2"] - 10, type=">=")
        ]
        ws_algo = WeightedSum(
            inner_solver=inner_solver,
            ws_pairs=ws_pairs,
            objectives=objectives,
            constraints=constraints,
        )
        with pytest.raises(NoSolutionError):
            ws_algo.solve(variables={"v1": BoolVariable(), "v2": IntegerVariable(1, 7)})

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
                function=lambda x, **kw: th.tensor(x["v1"]),
                direction_type="MIN",
            ),
            Objective(
                "obj2",
                function=lambda x, **kw: th.tensor(x["v2"]),
                direction_type="MIN",
            ),
            Objective(
                "obj3",
                function=lambda x, **kw: th.tensor((x["v1"] + x["v2"]) / 10),
                direction_type="MIN",
            ),
        ]
        constraints = [
            Constraint(function=lambda x, **kw: x["v1"] + x["v2"] - 3, type=">=")
        ]
        ws_algo = WeightedSum(
            inner_solver=inner_solver,
            ws_pairs=ws_pairs,
            objectives=objectives,
            constraints=constraints,
        )
        po_objs, po_vars = ws_algo.solve(
            input_parameters={"count": 1},
            variables={"v1": BoolVariable(), "v2": IntegerVariable(1, 7)},
        )

        np.testing.assert_equal(po_objs, np.array([[0, 3, 0.3]]))
        np.testing.assert_equal(po_vars, np.array([{"v1": 0, "v2": 3}]))

from typing import Dict, Optional

import numpy as np
import pytest
import torch as th

from ...concepts import BoolVariable, Constraint, IntegerVariable, Objective
from ...concepts.problem import MOProblem
from ...concepts.utils import InputParameters, InputVariables
from ...moo.weighted_sum import WeightedSum
from ...soo.grid_search_solver import GridSearch
from ...soo.random_sampler_solver import RandomSampler
from ...soo.so_solver import SOSolver
from ...utils.exceptions import NoSolutionError


@pytest.fixture
def simple_problem() -> MOProblem:
    def obj_func1(
        input_variables: InputVariables, input_parameters: InputParameters = None
    ) -> th.Tensor:
        return input_variables["v1"] + (input_parameters or {}).get("count", 0)

    def obj_func2(
        input_variables: InputVariables, input_parameters: InputParameters = None
    ) -> th.Tensor:
        return (input_variables["v1"] + input_variables["v2"]) / 10 + (
            input_parameters or {}
        ).get("count", 0)

    objectives = [
        Objective(
            "obj1",
            function=obj_func1,
            minimize=True,
        ),
        Objective("obj2", function=obj_func2, minimize=True),
    ]

    def constraint_func(
        input_variables: InputVariables, input_parameters: InputParameters = None
    ) -> th.Tensor:
        return (
            (input_variables["v1"] + input_variables["v2"])
            - 2
            - (input_parameters or {}).get("count", 0)
        )

    constraints = [Constraint(function=constraint_func, lower=0)]
    return MOProblem(
        objectives=objectives,
        constraints=constraints,
        variables={"v1": BoolVariable(), "v2": IntegerVariable(1, 7)},
        input_parameters={"count": 1},
    )


class TestWeightedSum:
    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearch(GridSearch.Params(n_grids_per_var=[2, 7])),
            RandomSampler(RandomSampler.Params(n_samples_per_param=30)),
        ],
    )
    def test_solve_without_input_parameters(
        self, inner_solver: SOSolver, simple_problem: MOProblem
    ) -> None:
        """solve a dummy minimization problem with 2 objectives and 1 constraint"""
        ws_pairs = np.array([[0.3, 0.7], [0.6, 0.4]])
        simple_problem.input_parameters = None

        ws_algo = WeightedSum(
            so_solver=inner_solver,
            ws_pairs=ws_pairs,
        )
        po_objs, po_vars = ws_algo.solve(problem=simple_problem, seed=0)
        np.testing.assert_array_almost_equal(po_objs, np.array([[0, 0.2]]))
        np.testing.assert_equal(po_vars, np.array({"v1": 0, "v2": 2}))

    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearch(GridSearch.Params(n_grids_per_var=[2, 7])),
            RandomSampler(RandomSampler.Params(n_samples_per_param=30)),
        ],
    )
    def test_solve_with_input_parameters(
        self, inner_solver: SOSolver, simple_problem: MOProblem
    ) -> None:
        """solve a dummy minimization problem with 2 objectives and 1 constraint"""
        ws_pairs = np.array([[0.3, 0.7], [0.6, 0.4]])

        ws_algo = WeightedSum(
            so_solver=inner_solver,
            ws_pairs=ws_pairs,
        )
        po_objs, po_vars = ws_algo.solve(problem=simple_problem, seed=0)

        np.testing.assert_almost_equal(po_objs, np.array([[1, 1.3]]))
        np.testing.assert_equal(po_vars, np.array([{"v1": 0, "v2": 3}]))

    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearch(GridSearch.Params(n_grids_per_var=[2, 7])),
            RandomSampler(RandomSampler.Params(n_samples_per_param=10)),
        ],
    )
    def test_ws_raises_no_solution(
        self, inner_solver: SOSolver, simple_problem: MOProblem
    ) -> None:
        ws_pairs = np.array([[0.3, 0.7], [0.6, 0.4]])

        def f3(
            input_variables: Dict[str, th.Tensor],
            input_parameters: Optional[Dict[str, th.Tensor]] = None,
        ) -> th.Tensor:
            return input_variables["v1"] + input_variables["v2"] - 10

        simple_problem.constraints = [Constraint(function=f3, lower=0)]
        simple_problem.input_parameters = None
        ws_algo = WeightedSum(
            so_solver=inner_solver,
            ws_pairs=ws_pairs,
        )
        with pytest.raises(NoSolutionError):
            ws_algo.solve(problem=simple_problem, seed=0)

    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearch(GridSearch.Params(n_grids_per_var=[2, 7])),
            RandomSampler(RandomSampler.Params(n_samples_per_param=30)),
        ],
    )
    def test_works_with_three_objectives(
        self, inner_solver: SOSolver, simple_problem: MOProblem
    ) -> None:
        ws_pairs = np.array([[0.3, 0.5, 0.2], [0.6, 0.3, 0.1]])

        def f2(
            input_variables: Dict[str, th.Tensor],
            input_parameters: Optional[Dict[str, th.Tensor]] = None,
        ) -> th.Tensor:
            return input_variables["v2"]

        objectives = list(simple_problem.objectives)
        objectives.insert(1, Objective("obj3", function=f2, minimize=True))
        simple_problem.objectives = objectives
        simple_problem.input_parameters = None

        def constraint_f(
            input_variables: Dict[str, th.Tensor],
            input_parameters: Optional[Dict[str, th.Tensor]] = None,
        ) -> th.Tensor:
            return input_variables["v1"] + input_variables["v2"] - 3

        simple_problem.constraints = [Constraint(function=constraint_f, lower=0)]
        ws_algo = WeightedSum(
            so_solver=inner_solver,
            ws_pairs=ws_pairs,
        )
        po_objs, po_vars = ws_algo.solve(problem=simple_problem, seed=0)

        np.testing.assert_almost_equal(po_objs, np.array([[0, 3, 0.3]]))
        np.testing.assert_equal(po_vars, np.array([{"v1": 0, "v2": 3}]))

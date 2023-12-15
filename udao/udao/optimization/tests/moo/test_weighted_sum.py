import numpy as np
import pytest
import torch as th

from ...concepts import BoolVariable, Constraint, IntegerVariable, Objective
from ...concepts.utils import InputParameters, InputVariables
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

        def f1(
            input_variables: InputVariables, input_parameters: InputParameters = None
        ) -> th.Tensor:
            return th.tensor(input_variables["v1"])

        def f2(
            input_variables: InputVariables, input_parameters: InputParameters = None
        ) -> th.Tensor:
            return th.tensor((input_variables["v1"] + input_variables["v2"]) / 10)

        def f3(
            input_variables: InputVariables, input_parameters: InputParameters = None
        ) -> th.Tensor:
            return th.tensor((input_variables["v1"] + input_variables["v2"]) - 2)

        objectives = [
            Objective(
                "obj1",
                function=f1,
                direction_type="MIN",
            ),
            Objective(
                "obj2",
                function=f2,
                direction_type="MIN",
            ),
        ]
        constraints = [
            Constraint(
                function=f3,
                lower=0,
            )
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

        def obj_func1(
            input_variables: InputVariables, input_parameters: InputParameters = None
        ) -> th.Tensor:
            if not input_parameters:
                return th.tensor(input_variables["v1"])
            else:
                return th.tensor(input_variables["v1"] + input_parameters["count"])

        def obj_func2(
            input_variables: InputVariables, input_parameters: InputParameters = None
        ) -> th.Tensor:
            if not input_parameters:
                return th.tensor((input_variables["v1"] + input_variables["v2"]) / 10)
            else:
                return th.tensor(
                    (input_variables["v1"] + input_variables["v2"]) / 10
                    + input_parameters["count"]
                )

        objectives = [
            Objective(
                "obj1",
                function=obj_func1,
                direction_type="MIN",
            ),
            Objective("obj2", function=obj_func2, direction_type="MIN"),
        ]

        def constraint_func(
            input_variables: InputVariables, input_parameters: InputParameters = None
        ) -> th.Tensor:
            if not input_parameters:
                return th.tensor((input_variables["v1"] + input_variables["v2"]) - 2)
            else:
                return th.tensor(
                    (input_variables["v1"] + input_variables["v2"])
                    - 2
                    - input_parameters["count"]
                )

        constraints = [Constraint(function=constraint_func, lower=0)]

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

        def f1(
            input_variables: InputVariables, input_parameters: InputParameters = None
        ) -> th.Tensor:
            return th.tensor(input_variables["v1"])

        def f2(
            input_variables: InputVariables, input_parameters: InputParameters = None
        ) -> th.Tensor:
            return th.tensor((input_variables["v1"] + input_variables["v2"]) / 10)

        def constraint_f(
            input_variables: InputVariables, input_parameters: InputParameters = None
        ) -> th.Tensor:
            return th.tensor((input_variables["v1"] + input_variables["v2"]) - 10)

        objectives = [
            Objective(
                "obj1",
                function=f1,
                direction_type="MIN",
            ),
            Objective(
                "obj2",
                function=f2,
                direction_type="MIN",
            ),
        ]
        constraints = [Constraint(function=constraint_f, lower=0)]
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

        def f1(
            input_variables: InputVariables, input_parameters: InputParameters = None
        ) -> th.Tensor:
            return th.tensor(input_variables["v1"])

        def f2(
            input_variables: InputVariables, input_parameters: InputParameters = None
        ) -> th.Tensor:
            return th.tensor(input_variables["v2"])

        def f3(
            input_variables: InputVariables, input_parameters: InputParameters = None
        ) -> th.Tensor:
            return th.tensor((input_variables["v1"] + input_variables["v2"]) / 10)

        def constraint_f(
            input_variables: InputVariables, input_parameters: InputParameters = None
        ) -> th.Tensor:
            return th.tensor(input_variables["v1"] + input_variables["v2"] - 3)

        objectives = [
            Objective(
                "obj1",
                function=f1,
                direction_type="MIN",
            ),
            Objective(
                "obj2",
                function=f2,
                direction_type="MIN",
            ),
            Objective(
                "obj3",
                function=f3,
                direction_type="MIN",
            ),
        ]
        constraints = [Constraint(function=constraint_f, lower=0)]
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

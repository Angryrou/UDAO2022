import numpy as np
import pytest
import torch as th

from udao.optimization.concepts import BoolVariable, Constraint, IntegerVariable, FloatVariable, Objective
from udao.optimization.concepts.problem import MOProblem
from udao.optimization.concepts.utils import InputParameters, InputVariables
from udao.optimization.moo.weighted_sum import WeightedSum
from udao.optimization.soo.base_solver import SOSolver
from udao.optimization.soo.grid_search_solver import GridSearch
from udao.optimization.soo.random_sampler_solver import RandomSampler
from udao.optimization.utils.exceptions import NoSolutionError

# generate even weights for 2d and 3D
def even_weights(stepsize, m):
    if m == 2:
        w1 = np.hstack([np.arange(0, 1, stepsize), 1])
        w2 = 1 - w1
        ws_pairs = [[w1, w2] for w1, w2 in zip(w1, w2)]

    elif m == 3:
        w_steps = np.linspace(0, 1, num=int(1 / stepsize) + 1, endpoint=True)
        for i, w in enumerate(w_steps):
            # use round to avoid case of floating point limitations in Python
            # the limitation: 1- 0.9 = 0.09999999999998 rather than 0.1
            other_ws_range = round((1 - w), 10)
            w2 = np.linspace(0, other_ws_range, num=round(other_ws_range/stepsize + 1), endpoint=True)
            w3 = other_ws_range - w2
            num = w2.shape[0]
            w1 = np.array([w] * num)
            ws = np.hstack([w1.reshape([num, 1]), w2.reshape([num, 1]), w3.reshape([num, 1])])
            if i == 0:
                ws_pairs = ws
            else:
                ws_pairs = np.vstack([ws_pairs, ws])

    assert all(np.round(np.sum(ws_pairs, axis=1), 10) == 1)
    return ws_pairs

class TestWeightedSumHcf:
    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearch(GridSearch.Params(n_grids_per_var=[100, 100])),
            # RandomSampler(RandomSampler.Params(n_samples_per_param=10000, seed=0)),
        ],
    )
    def test_solve_without_input_parameters(
        self, inner_solver: SOSolver, hcf_problem: MOProblem
    ) -> None:
        """solve a dummy minimization problem with 2 objectives and 2 constraint"""
        # ws_pairs = np.array([[0.3, 0.7], [0.6, 0.4]])
        ws_steps = 1 / 5
        ws_pairs = even_weights(ws_steps, 2)
        hcf_problem.input_parameters = None

        ws_algo = WeightedSum(
            so_solver=inner_solver,
            ws_pairs=ws_pairs,
        )
        po_objs, po_vars = ws_algo.solve(problem=hcf_problem)
        np.testing.assert_equal(np.round(po_objs, 5), np.array([[136, 4],
                                                                [91.87185, 5.59423],
                                                                [55.39067, 11.22141],
                                                                [21.82185, 22.42516],
                                                                [4.77870, 35.74013],
                                                                [0, 50]]))
        # np.testing.assert_equal(po_vars, np.array([{"v1": 5, "v2": 3},
        #                                            {"v1": 3.74, "v2": 3},
        #                                            {"v1": 2.63, "v2": 2.63},
        #                                            {"v1": 1.67, "v2": 1.64},
        #                                            {"v1": 0.76, "v2": 0.79},
        #                                            {"v1": 0, "v2": 0}]))


class TestWeightedSumModel:
    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearch(GridSearch.Params(n_grids_per_var=[100, 100])),
            RandomSampler(RandomSampler.Params(n_samples_per_param=10000, seed=0)),
        ],
    )
    def test_solve(
        self, inner_solver: SOSolver, simple_nn_problem: MOProblem
    ) -> None:
        """solve a dummy minimization problem with 2 objectives and 2 constraint"""
        # ws_pairs = np.array([[0.3, 0.7], [0.6, 0.4]])
        ws_steps = 1 / 5
        ws_pairs = even_weights(ws_steps, 2)
        simple_nn_problem.input_parameters = None

        ws_algo = WeightedSum(
            so_solver=inner_solver,
            ws_pairs=ws_pairs,
        )
        po_objs, po_vars = ws_algo.solve(problem=simple_nn_problem)
        # np.testing.assert_equal(po_objs, np.array([[136, 4], [0, 50]]))
        # np.testing.assert_equal(po_vars, np.array([{"v1": 5, "v2": 3}, {"v1": 0, "v2": 0}]))
from typing import Dict

import numpy as np
import torch as th

from udao.optimization.concepts import Constraint, FloatVariable, Objective, Variable
from udao.optimization.concepts.problem import MOProblem
from udao.optimization.concepts.utils import InputParameters, InputVariables
from udao.optimization.moo.weighted_sum import WeightedSum
from udao.optimization.moo.progressive_frontier import SequentialProgressiveFrontier
from udao.optimization.moo.progressive_frontier import ParallelProgressiveFrontier
from udao.optimization.soo.grid_search_solver import GridSearch
from udao.optimization.soo.mogd import MOGD
from udao.utils.logging import logger

#     An example: 2D
#     https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization_problems
#     Binh and Korn function:
#     # minimize:
#     #          f1(x1, x2) = 4 * x_1 * x_1 + 4 * x_2 * x_2
#     #          f2(x1, x2) = (x_1 - 5) * (x_1 - 5) + (x_2 - 5) * (x_2 - 5)
#     # subject to:
#     #          g1(x_1, x_2) = (x_1 - 5) * (x_1 - 5) + x_2 * x_2 <= 25
#     #          g2(x_1, x_2) = (x_1 - 8) * (x_1 - 8) + (x_2 + 3) * (x_2 + 3) >= 7.7
#     #          x_1 in [0, 5], x_2 in [0, 3]
#     """

logger.setLevel("INFO")


if __name__ == "__main__":
    def Obj1(input_variables: InputVariables, input_parameters: InputParameters = None) -> th.Tensor:
        y = 4 * input_variables["v1"] ** 2 + 4 * input_variables["v2"] ** 2
        return th.reshape(y, (-1, 1))


    def Obj2(input_variables: InputVariables, input_parameters: InputParameters = None) -> th.Tensor:
        y = (input_variables["v1"] - 5) ** 2 + (input_variables["v2"] - 5) ** 2
        return th.reshape(y, (-1, 1))


    def Const1(input_variables: InputVariables, input_parameters: InputParameters = None) -> th.Tensor:
        y = (input_variables["v1"] - 5) ** 2 + input_variables["v2"] ** 2
        return th.reshape(y, (-1, 1))


    def Const2(input_variables: InputVariables, input_parameters: InputParameters = None) -> th.Tensor:
        y = (input_variables["v1"] - 8) ** 2 + (input_variables["v2"] + 3) ** 2
        return th.reshape(y, (-1, 1))

    objectives = [
        Objective("obj1", minimize=True, function=Obj1),
        Objective("obj2", minimize=True, function=Obj2),
    ]
    variables: Dict[str, Variable] = {
        "v1": FloatVariable(0, 5),
        "v2": FloatVariable(0, 3),
    }
    constraints = [Constraint(function=Const1, upper=25),
                   Constraint(function=Const2, lower=7.7)]

    problem = MOProblem(
        objectives=objectives,
        variables=variables,
        constraints=constraints,
        input_parameters=None,
    )

    # Single-objective optimization solvers
    so_mogd = MOGD(
        MOGD.Params(
            learning_rate=0.1,
            max_iters=100,
            patience=20,
            multistart=1,
            objective_stress=10,
            device=th.device("cpu"),
        )
    )
    so_grid = GridSearch(GridSearch.Params(n_grids_per_var=[100, 100]))

    # WS
    w1 = np.linspace(0, 1, num=11, endpoint=True)
    w2 = 1 - w1
    ws_pairs = np.vstack((w1, w2)).T
    ws_algo = WeightedSum(
        so_solver=so_grid,
        ws_pairs=ws_pairs,
    )
    ws_objs, ws_vars = ws_algo.solve(problem=problem)
    logger.info(f"Found PF-AS solutions of closed form: {ws_objs}, {ws_vars}")

    # PF-AS
    spf = SequentialProgressiveFrontier(
        params=SequentialProgressiveFrontier.Params(n_probes=11),
        solver=so_mogd,
    )
    spf_objs, spf_vars = spf.solve(
        problem=problem,
        seed=0
    )
    logger.info(f"Found PF-AS solutions of closed form: {spf_objs}, {spf_vars}")

    # PF-AP
    ppf = ParallelProgressiveFrontier(
        params=ParallelProgressiveFrontier.Params(
            processes=1,
            n_grids=2,
            max_iters=4,
        ),
        solver=so_mogd,
    )
    ppf_objs, ppf_vars = ppf.solve(
        problem=problem,
        seed=0,
    )
    logger.info(f"Found PF-AP solutions of closed form: {ppf_objs}, {ppf_vars}")


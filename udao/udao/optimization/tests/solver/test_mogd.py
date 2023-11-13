import numpy as np
import pytest
import torch as th
import torch.nn as nn

from ....model.utils.utils import set_deterministic_torch
from ...concepts import FloatVariable, IntegerVariable, Objective
from ...solver.mogd import MOGD


class SimpleModel1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = th.nn.Linear(2, 2)
        self.fc2 = th.nn.Linear(2, 1)

    def forward(self, x: th.Tensor, wl_id: None = None) -> th.Tensor:
        return self.fc2(self.fc1(x))


class SimpleModel2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = th.nn.Linear(2, 1)

    def forward(self, x: th.Tensor, wl_id: None = None) -> th.Tensor:
        return self.fc1(x)


@pytest.fixture()
def mogd() -> MOGD:
    set_deterministic_torch(42)

    params = MOGD.Params(
        learning_rate=0.1,
        weight_decay=0.1,
        max_iters=10,
        patient=10,
        seed=0,
        multistart=10,
        processes=1,
        stress=0.1,
    )
    mogd = MOGD(params)
    model_1 = SimpleModel1()
    model_2 = SimpleModel2()
    mogd._problem(
        variables=[FloatVariable(0, 1), IntegerVariable(2, 3)],
        accurate=True,
        std_func=None,
        objectives=[
            Objective(
                "obj1",
                "MAX",
                model_1,  # type: ignore
            ),
            Objective(
                "obj2",
                "MIN",
                model_2,
            ),
        ],
        constraints=[],
    )

    return mogd


class TestMOGD:
    def test_constraint_so_opt(self, mogd: MOGD) -> None:
        optimal_obj, optimal_vars = mogd.constraint_so_opt(
            wl_id="1",
            obj="obj1",
            opt_obj_ind=0,
            obj_bounds_dict={"obj1": th.tensor((0, 2)), "obj2": th.tensor((0, 1))},
            precision_list=[2, 2],
            verbose=False,
        )
        assert optimal_obj is not None
        np.testing.assert_array_almost_equal(
            optimal_obj, np.array([0.7501509189605713, 0.261410653591156])
        )
        assert optimal_vars is not None
        np.testing.assert_array_equal(optimal_vars, np.array([[0.0, 2.21]]))

    def test_constraint_parallel(self, mogd: MOGD) -> None:
        set_deterministic_torch(42)
        res_list = mogd.constraint_so_parallel(
            wl_id="1",
            obj="obj1",
            opt_obj_ind=0,
            precision_list=[2, 2],
            cell_list=[
                {"obj1": th.tensor((0, 2)), "obj2": th.tensor((0, 1))},
                {"obj1": th.tensor((0.05, 2.1)), "obj2": th.tensor((0.1, 1))},
            ],
        )
        obj_optimal_1, var_optimal_1 = res_list[0]
        obj_optimal_2, var_optimal_2 = res_list[1]
        assert obj_optimal_1 is not None
        np.testing.assert_array_almost_equal(
            obj_optimal_1, [0.7501509189605713, 0.261410653591156]
        )  # type: ignore
        assert var_optimal_1 is not None
        np.testing.assert_array_equal(var_optimal_1, [[0.0, 2.21]])
        assert obj_optimal_2 is not None
        np.testing.assert_array_almost_equal(
            obj_optimal_2, [0.7494739890098572, 0.2552645206451416]
        )
        assert var_optimal_2 is not None
        np.testing.assert_array_equal(var_optimal_2, [[0.0, 2.2]])

    def test_single_objective_opt(self, mogd: MOGD) -> None:
        mogd.objectives = [
            Objective(
                "obj1",
                "MAX",
                lambda x, wl_id: th.reshape(x[:, 0] ** 2 + x[:, 1] ** 2, (-1, 1)),  # type: ignore
            ),
            Objective(
                "obj2",
                "MIN",
                lambda x, wl_id: th.reshape(
                    (x[:, 0] - 1) ** 2 + x[:, 1] ** 2, (-1, 1)
                ),  # type: ignore
            ),
        ]

        optimal_obj, optimal_vars = mogd.single_objective_opt(
            wl_id="1",
            obj="obj1",
            opt_obj_ind=0,
            precision_list=[2, 2],
            verbose=False,
        )

        assert optimal_obj is not None
        np.testing.assert_array_equal(optimal_obj, np.array([2]))
        assert optimal_vars is not None
        np.testing.assert_array_equal(optimal_vars, np.array([[1, 3]]))

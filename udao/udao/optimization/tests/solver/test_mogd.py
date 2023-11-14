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
    mogd.problem_setup(
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
        precision_list=[2, 2],
    )

    return mogd


class TestMOGD:
    def test_constraint_so_opt(self, mogd: MOGD) -> None:
        optimal_obj, optimal_vars = mogd.optimize_constrained_so(
            wl_id="1",
            objective_name="obj1",
            obj_bounds_dict={"obj1": th.tensor((0, 2)), "obj2": th.tensor((0, 1))},
        )
        assert optimal_obj is not None
        np.testing.assert_array_almost_equal(
            optimal_obj, np.array([0.7501509189605713, 0.261410653591156])
        )
        assert optimal_vars is not None
        np.testing.assert_array_equal(optimal_vars, np.array([[0, 2.21]]))

    def test_constraint_parallel(self, mogd: MOGD) -> None:
        res_list = mogd.optimize_constrained_so_parallel(
            wl_id="1",
            objective_name="obj1",
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

    def test_constraint_single_objective_opt(self, mogd: MOGD) -> None:
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

        optimal_obj, optimal_vars = mogd.optimize_constrained_so(
            wl_id="1",
            objective_name="obj1",
            obj_bounds_dict=None,
            batch_size=16,
        )

        assert optimal_obj is not None
        np.testing.assert_array_equal(optimal_obj, np.array([2, 1]))
        assert optimal_vars is not None
        np.testing.assert_array_equal(optimal_vars, np.array([[1, 3]]))

    def test__soo_loss_no_batch(self, mogd: MOGD) -> None:
        vars = th.rand(2)
        obj_bounds_dict = {"obj1": th.tensor((0, 2)), "obj2": th.tensor((0, 1))}
        objs_pred_dict = {
            cst_obj: mogd._get_tensor_obj_pred("1", vars, ob_ind)
            for ob_ind, cst_obj in enumerate(obj_bounds_dict)
        }
        loss, loss_idx = mogd._soo_loss(
            "1", vars, objs_pred_dict, obj_bounds_dict, "obj1", 0
        )
        assert th.allclose(loss, th.tensor(-0.2764), rtol=1e-3)
        assert th.equal(loss_idx, th.tensor(0))

    def test__soo_loss_with_batch(self, mogd: MOGD) -> None:
        vars = th.rand(3, 2)
        obj_bounds_dict = {"obj1": th.tensor((0, 2)), "obj2": th.tensor((0, 1))}
        objs_pred_dict = {
            cst_obj: mogd._get_tensor_obj_pred("1", vars, ob_ind)
            for ob_ind, cst_obj in enumerate(obj_bounds_dict)
        }
        loss, loss_idx = mogd._soo_loss(
            "1", vars, objs_pred_dict, obj_bounds_dict, "obj1", 0
        )
        assert th.allclose(loss, th.tensor(-0.2880), rtol=1e-3)
        assert th.equal(loss_idx, th.tensor(1))

    def test__unbounded_soo_loss(self, mogd: MOGD) -> None:
        vars = th.rand(3, 2)
        objs_pred_dict = {
            cst_obj.name: mogd._get_tensor_obj_pred("1", vars, ob_ind)
            for ob_ind, cst_obj in enumerate(mogd.objectives)
        }
        loss, loss_idx = mogd._unbounded_soo_loss("1", 0, objs_pred_dict, vars)
        assert th.allclose(loss, th.tensor(-0.3319), rtol=1e-3)
        assert th.equal(loss_idx, th.tensor(1))

from typing import Any

import pytest
import torch as th
from torch import nn

from ..module import UdaoModule
from ..utils.utils import set_deterministic_torch


@pytest.fixture
def sample_module() -> UdaoModule:
    set_deterministic_torch(0)

    model = nn.Linear(2, 2)
    objectives = ["obj1", "obj2"]
    loss = nn.L1Loss()
    module = UdaoModule(model, objectives, loss)
    return module


class TestUdaoModule:
    def test_initialize(self, sample_module: UdaoModule) -> None:
        assert sample_module.loss_weights == {"obj1": 1.0, "obj2": 1.0}
        assert isinstance(sample_module.optimizer, th.optim.AdamW)

    def test_compute_loss(self, sample_module: UdaoModule) -> None:
        y = th.tensor([[1, 1], [2, 2]], dtype=th.float32)
        y_hat = th.tensor([[2, 2], [1, 1]], dtype=th.float32)
        loss, loss_dict = sample_module.compute_loss(y, y_hat)
        assert loss == 2.0
        assert loss_dict == {"obj1": 1.0, "obj2": 1.0}

    def test_training_step(self, sample_module: UdaoModule, mocker: Any) -> None:
        batch = (
            th.tensor([[1, 1], [2, 2]], dtype=th.float32),
            th.tensor([[2, 2], [1, 1]], dtype=th.float32),
        )
        mocked_log = mocker.patch.object(sample_module, "log")
        returned = sample_module.training_step(batch, 0)
        assert mocked_log.called
        assert th.allclose(returned, th.tensor(4.1752))
        assert len(sample_module.step_outputs["train"]) == 1
        assert len(sample_module.step_outputs["train"]) == 1

    def test__shared_estimate(self, sample_module: UdaoModule, mocker: Any) -> None:
        batch = (
            th.tensor([[0.5, 2], [0.5, 2]], dtype=th.float32),
            th.tensor([[1, 1], [1, 1]], dtype=th.float32),
        )
        mocked_log = mocker.spy(sample_module, "log")
        sample_module.step_outputs["train"].append(batch[0])
        sample_module.step_targets["train"].append(batch[1])
        sample_module._shared_estimate("train")
        assert mocked_log.call_count == 3
        assert mocked_log.call_args_list[0][0] == ("train_loss_epoch", 1.5)
        assert mocked_log.call_args_list[1][0] == ("wmape_obj1", 0.5)
        assert mocked_log.call_args_list[2][0] == ("wmape_obj2", 1)

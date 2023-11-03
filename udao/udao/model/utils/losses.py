import torch as th
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class WMAPELoss(_Loss):
    def __init__(self) -> None:
        super().__init__(reduction="sum")

    def forward(self, input: th.Tensor, target: th.Tensor) -> th.Tensor:
        return th.abs(input - target).sum() / target.sum()


class MSLELoss(_Loss):
    def forward(self, input: th.Tensor, target: th.Tensor) -> th.Tensor:
        return F.mse_loss(
            th.log(input + 1e-3), th.log(target + 1e-3), reduction=self.reduction
        )


class MAPELoss(_Loss):
    def forward(self, input: th.Tensor, target: th.Tensor) -> th.Tensor:
        return th.mean(th.abs(input - target) / (target + 1e-3))

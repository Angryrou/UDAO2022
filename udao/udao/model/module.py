from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import torch as th
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.nn.modules.loss import _Loss
from torchmetrics import Metric, MetricCollection

from .utils.losses import WMAPELoss


@dataclass
class LearningParams:
    init_lr: float = 1e-3
    """Initial learning rate."""
    weight_decay: float = 1e-2
    """Weight decay."""
    min_lr: float = 1e-5
    """Minimum learning rate."""


class UdaoModule(pl.LightningModule):
    """Pytorch Lightning module for UDAO.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    metrics: Optional[List[Metric]], optional
        A list of metrics - from torchmetrics - to compute,
        by default None
    objectives : List[str]
        The list of objectives to train on.
        Should be a subset of the objectives in the dataset, i.e.
        the size of the output of the model should be equal or
        larger than the size of the list of objectives.
    loss : Optional[_Loss], optional
        A ptorch loss function to apply, by default None
        If None, the WMAPELoss is used.
    loss_weights : Optional[Dict[str, float]], optional
        Loss weights to apply in sum of different
        objective losses, by default None
    learning_params : Optional[LearningParams], optional
        Learning parameters, by default None
        Default values are used if None.
    """

    def __init__(
        self,
        model: nn.Module,
        objectives: List[str],
        loss: Optional[_Loss] = None,
        metrics: Optional[List[Metric]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        learning_params: Optional[LearningParams] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss: _Loss = WMAPELoss()
        if loss is not None:
            self.loss = loss

        self.objectives = objectives
        self.lr_scheduler: Optional[th.optim.lr_scheduler.LRScheduler] = None
        if learning_params is None:
            self.learning_params = LearningParams()
        else:
            self.learning_params = learning_params
        self.optimizer: th.optim.Optimizer = th.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_params.init_lr,
            weight_decay=self.learning_params.weight_decay,
        )
        self.loss_weight: Dict[str, float]
        if loss_weights is None:
            self.loss_weights = {m: 1.0 for m in self.objectives}
        else:
            self.loss_weights = loss_weights
        self.metrics: Dict[str, Dict[str, MetricCollection]] = defaultdict(dict)
        if metrics:
            metric_collection = MetricCollection(metrics)
            for split in ["train", "val", "test"]:
                for objective in self.objectives:
                    self.metrics[split][objective] = metric_collection.clone(
                        prefix=f"{split}_{objective}_"
                    )

    def compute_loss(
        self,
        y: th.Tensor,
        y_hat: th.Tensor,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """Compute the loss for different objectives
        and sum them with given weights, if more than one."""
        loss = th.tensor(0)
        loss_dict: Dict[str, th.Tensor] = {
            m: self.loss(y_hat[:, i], y[:, i]) for i, m in enumerate(self.objectives)
        }

        if len(loss_dict) == 1:
            loss = loss_dict[self.objectives[0]]
        else:
            loss = th.sum(
                th.tensor(
                    [self.loss_weights[k] * loss for k, loss in loss_dict.items()]
                )
            )
        return loss, loss_dict

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer

    def _shared_step(
        self, batch: Tuple[Any, th.Tensor], split: str
    ) -> Tuple[th.Tensor, th.Tensor]:
        features, y = batch
        y_hat = self.model(features)
        for i, objective in enumerate(self.objectives):
            self.metrics[split][objective].update(y_hat[:, i], y[:, i])
        return y_hat, y

    def _shared_epoch_end(self, split: str) -> None:
        for objective in self.objectives:
            metric = self.metrics[split][objective]
            output = metric.compute()
            self.log_dict(output, on_epoch=True, prog_bar=True, logger=True)
            metric.reset()

    def training_step(self, batch: Tuple[Any, th.Tensor], batch_idx: int) -> th.Tensor:
        y_hat, y = self._shared_step(batch, "train")
        loss, _ = self.compute_loss(y, y_hat)
        if th.isnan(loss):
            raise ValueError("get a nan loss in train")
        elif th.isinf(loss):
            raise ValueError("get an inf loss in train")
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self._shared_epoch_end("train")

    def validation_step(self, batch: Tuple[Any, th.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._shared_epoch_end("val")

    def test_step(self, batch: Tuple[Any, th.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        self._shared_epoch_end("test")

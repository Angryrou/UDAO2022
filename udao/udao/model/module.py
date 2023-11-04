from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch as th
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from scipy.stats import pearsonr
from torch import nn
from torch.nn.modules.loss import _Loss

from .utils.losses import WMAPELoss


def compute_metrics(y_objective: th.Tensor, y_hat_objective: th.Tensor) -> Dict:
    """Compute a dictionary of metrics for a given objective."""
    y_err = np.abs(y_objective - y_hat_objective)
    wmape = (y_err.sum() / y_objective.sum()).item()
    y_err_rate = y_err / (y_objective + np.finfo(np.float32).eps)
    mape = y_err_rate.mean()
    err_50, err_90, err_95, err_99 = np.percentile(y_err_rate, [50, 90, 95, 99])
    glb_err = np.abs(y_objective.sum() - y_hat_objective.sum()) / y_hat_objective.sum()
    corr, _ = pearsonr(y_objective, y_hat_objective)
    q_errs = np.maximum(
        (y_objective + np.finfo(np.float32).eps)
        / (y_hat_objective + np.finfo(np.float32).eps),
        (y_hat_objective + np.finfo(np.float32).eps)
        / (y_objective + np.finfo(np.float32).eps),
    )
    q_err_mean = np.mean(q_errs)
    q_err_50, q_err_90, q_err_95, q_err_99 = np.percentile(q_errs, [50, 90, 95, 99])
    return {
        "wmape": wmape,
        "mape": mape,
        "err_50": err_50,
        "err_90": err_90,
        "err_95": err_95,
        "err_99": err_99,
        "q_err_mean": q_err_mean,
        "q_err_50": q_err_50,
        "q_err_90": q_err_90,
        "q_err_95": q_err_95,
        "q_err_99": q_err_99,
        "glb_err": glb_err,
        "corr": corr,
    }


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
    learning_params : Optional[Dict[str, Any]], optional
        _description_, by default None
    """

    def __init__(
        self,
        model: nn.Module,
        objectives: List[str],
        loss: Optional[_Loss] = None,
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

        # value lists for end of epoch computations
        self.step_outputs: Dict[str, List[th.Tensor]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self.step_targets: Dict[str, List[th.Tensor]] = {
            "train": [],
            "val": [],
            "test": [],
        }

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

    def training_step(self, batch: Tuple[Any, th.Tensor], batch_idx: int) -> th.Tensor:
        # training_step defines the train loop.
        features, y = batch
        y_hat = self.model(features)
        loss, _ = self.compute_loss(y, y_hat)
        if th.isnan(loss):
            raise ValueError("get a nan loss in train")
        elif th.isinf(loss):
            raise ValueError("get an inf loss in train")
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.step_outputs["train"].append(y_hat)
        self.step_targets["train"].append(y)
        return loss

    def on_train_epoch_end(self) -> None:
        self._shared_estimate("train")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer

    def validation_step(self, batch: Tuple[Any, th.Tensor], batch_idx: int) -> None:
        features, y = batch
        y_hat = self.model(features)
        self.step_outputs["val"].append(y_hat)
        self.step_targets["val"].append(y)

    def _shared_estimate(self, split: str) -> None:
        all_preds = th.cat(self.step_outputs[split], dim=0)
        all_targets = th.cat(self.step_targets[split], dim=0)

        loss, _ = self.compute_loss(all_targets, all_preds)
        self.log(f"{split}_loss_epoch", loss, prog_bar=True, logger=True)

        for i, objective in enumerate(self.objectives):
            metrics = compute_metrics(
                all_targets[:, i].detach().cpu().numpy(),
                all_preds[:, i].detach().cpu().numpy(),
            )
            self.log(
                f"wmape_{objective}",
                metrics["wmape"],
                prog_bar=True,
                logger=True,
            )
        self.step_outputs[split].clear()
        self.step_targets[split].clear()

    def on_validation_epoch_end(self) -> None:
        self._shared_estimate("val")

    def test_step(self, batch: Tuple[Any, th.Tensor], batch_idx: int) -> None:
        features, y = batch
        y_hat = self.model(features)
        self.step_outputs["test"].append(y_hat)
        self.step_targets["test"].append(y)

    def on_test_epoch_end(self) -> None:
        self._shared_estimate("val")

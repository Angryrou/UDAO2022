from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import pytorch_warmup as warmup
import torch as th
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion, OptimizerLRScheduler
from scipy.stats import pearsonr
from torch import nn


def loss_compute(
    y: th.Tensor,
    y_hat: th.Tensor,
    loss_type: str,
    objectives: List[str],
    loss_weights: Dict[str, float],
) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
    loss = th.tensor(0)
    loss_dict: Dict[str, th.Tensor] = {
        m: get_loss(y[:, i], y_hat[:, i], loss_type) for i, m in enumerate(objectives)
    }

    if len(loss_dict) == 1:
        loss = loss_dict[objectives[0]]
    else:
        loss = th.sum(
            th.tensor([loss_weights[k] * loss for k, loss in loss_dict.items()])
        )
    return loss, loss_dict


def compute_metrics(y_objective: th.Tensor, y_hat_objective: th.Tensor) -> Dict:
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


def get_loss(y: th.Tensor, y_hat: th.Tensor, loss_type: str) -> th.Tensor:
    loss = th.tensor(0)
    if loss_type == "wmape":
        loss = nn.L1Loss(reduction="sum")(y_hat, y) / y.sum()
    elif loss_type == "msle":
        loss = nn.MSELoss()(th.log(y_hat + 1e-3), th.log(y + 1e-3))
    elif loss_type == "mae":
        loss = nn.L1Loss(reduction="sum")(y_hat, y)
    elif loss_type == "mape":
        loss = th.mean(th.abs(y_hat - y) / (y + 1e-3))
    elif loss_type == "mape+wmape":
        loss = nn.L1Loss(reduction="sum")(y_hat, y) / y.sum() + th.mean(
            th.abs(y_hat - y) / (y + 1e-3)
        )
    elif loss_type == "mse":
        loss = nn.MSELoss()(y_hat, y)
    elif loss_type == "nll":
        loss = nn.functional.nll_loss(y_hat, y)
    else:
        raise Exception(f"loss_type {loss_type} not supported")
    return loss


class UdaoModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        objectives: List[str],
        loss_type: str = "wmape",
        loss_weights: Optional[Dict[str, float]] = None,
        learning_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.training_step_outputs: List[th.Tensor] = []
        self.training_step_targets: List[th.Tensor] = []
        self.validation_step_outputs: List[th.Tensor] = []
        self.validation_step_targets: List[th.Tensor] = []
        self.loss_type = loss_type
        self.objectives = objectives
        self.lr_scheduler: th.optim.lr_scheduler.LRScheduler
        if learning_params is None:
            self.learning_params = {
                "init_lr": 1e-3,
                "min_lr": 1e-5,
                "weight_decay": 1e-5,
                "epochs": 100,
            }
        else:
            self.learning_params = learning_params
        self.optimizer: th.optim.Optimizer = th.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_params["init_lr"],
            weight_decay=self.learning_params["weight_decay"],
        )
        self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)
        self.loss_weight: Dict[str, float]
        if loss_weights is None:
            self.loss_weights = {m: 1.0 for m in self.objectives}
        else:
            self.loss_weights = loss_weights

    def training_step(self, batch: Tuple[Any, th.Tensor], batch_idx: int) -> th.Tensor:
        # training_step defines the train loop.
        features, y = batch
        y_hat = self.model(features)
        loss, _ = loss_compute(
            y, y_hat, self.loss_type, self.objectives, self.loss_weights
        )
        if th.isnan(loss):
            raise ValueError("get a nan loss in train")
        elif th.isinf(loss):
            raise ValueError("get an inf loss in train")
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.training_step_outputs.append(y_hat)
        self.training_step_targets.append(y)
        return loss

    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]
    ) -> None:
        if self.warmup_scheduler is not None:
            with self.warmup_scheduler.dampening():
                scheduler.step(metric)
        else:
            scheduler.step(metric)

    def on_train_epoch_end(self) -> None:
        all_preds = th.stack(self.training_step_outputs)
        all_targets = th.stack(self.training_step_targets)

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
        self.training_step_outputs.clear()
        self.training_step_targets.clear()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}

    def on_train_start(self) -> None:
        self.lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(
                self.trainer.num_training_batches * self.learning_params["epochs"]
            ),
            eta_min=self.learning_params["min_lr"],
        )
        self.configure_optimizers()  # test that this works

    def validation_step(self, batch: Tuple[Any, th.Tensor], batch_idx: int) -> None:
        features, y = batch
        y_hat = self.model(features)
        self.validation_step_outputs.append(y_hat)
        self.validation_step_targets.append(y)

    def on_validation_epoch_end(self) -> None:
        all_preds = th.stack(self.validation_step_outputs)
        all_targets = th.stack(self.validation_step_outputs)

        loss, _ = loss_compute(
            all_targets, all_preds, self.loss_type, self.objectives, self.loss_weights
        )
        self.log("val_loss", loss, prog_bar=True, logger=True)

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
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()

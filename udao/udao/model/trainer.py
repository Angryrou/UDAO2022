from time import time
from typing import Dict, List, Tuple
import torch as th
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from pytorch_warmup import BaseWarmup
from utils.logging import logger
import pytorch_lightning as pl
import numpy as np
from scipy.stats import pearsonr
from torch import nn


def loss_compute(
    y, y_hat, loss_type, objectives: List[str], loss_weights
) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
    loss = th.tensor(0)
    loss_dict: Dict[str, th.Tensor] = {
        m: get_loss(y[:, i], y_hat[:, i], loss_type) for i, m in enumerate(objectives)
    }

    if len(loss_dict) == 1:
        loss = loss_dict[objectives[0]]
    else:
        loss = th.sum(th.tensor([loss_weights[k] * l for k, l in loss_dict.items()]))
    return loss, loss_dict


def get_eval_metrics(y_list, y_hat_list, loss_type, objectives, loss_weights):
    y = th.vstack(y_list)
    y_hat = th.vstack(y_hat_list)
    loss_total, loss_dict = loss_compute(y, y_hat, loss_type, objectives, loss_weights)
    metrics_dict = {}
    for i, objective in enumerate(objectives):
        loss = loss_dict[objective].item()
        y_i, y_hat_i = (
            y[:, i].detach().cpu().numpy(),
            y_hat[:, i].detach().cpu().numpy(),
        )
        y_err = np.abs(y_i - y_hat_i)
        wmape = (y_err.sum() / y_i.sum()).item()
        y_err_rate = y_err / (y_i + np.finfo(np.float32).eps)
        mape = y_err_rate.mean()
        err_50, err_90, err_95, err_99 = np.percentile(y_err_rate, [50, 90, 95, 99])
        glb_err = np.abs(y_i.sum() - y_hat_i.sum()) / y_hat_i.sum()
        corr, _ = pearsonr(y_i, y_hat_i)
        q_errs = np.maximum(
            (y_i + np.finfo(np.float32).eps) / (y_hat_i + np.finfo(np.float32).eps),
            (y_hat_i + np.finfo(np.float32).eps) / (y_i + np.finfo(np.float32).eps),
        )
        q_err_mean = np.mean(q_errs)
        q_err_50, q_err_90, q_err_95, q_err_99 = np.percentile(q_errs, [50, 90, 95, 99])
        metrics_dict[objective] = {
            "loss": loss,
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

    return loss_total, metrics_dict


def get_loss(y, y_hat, loss_type) -> th.Tensor:
    loss = 0
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


class TrainStatsTrace:  # Use ModelCheckpoint in pytorch-lightning instead
    def __init__(self, weights_pth_signature):
        self.weights_pth_signature = weights_pth_signature
        self.best_epoch = -1
        self.best_batch = -1
        self.best_loss = float("inf")

    def update(self, model, cur_epoch, cur_batch, cur_loss):
        if cur_loss < self.best_loss:
            self.best_loss = cur_loss
            self.best_epoch = cur_epoch
            self.best_batch = cur_batch
            th.save(
                {
                    "model": model.state_dict(),
                    "best_epoch": self.best_epoch,
                    "best_batch": self.best_batch,
                },
                self.weights_pth_signature,
            )

    def pop_model_dict(self, device):
        try:
            ckp_model = th.load(self.weights_pth_signature, map_location=device)
            return ckp_model["model"]
        except:
            raise Exception(f"{self.weights_pth_signature} has not got desire ckp.")


class UdaoModule(pl.LightningModule):
    def __init__(self, embedder, regressor, objectives: List[str]):
        super().__init__()
        self.embedder = embedder
        self.regressor = regressor
        self.validation_step_outputs = []
        self.validation_step_targets = []
        self.loss_type = "wmape"
        self.objectives = objectives
        self.loss_weights = th.ones(len(objectives))

    def training_step(self, batch):
        # training_step defines the train loop.
        embedding_input, other_input, y = batch
        embedding = self.embedder(embedding_input)
        y_hat = self.regressor(embedding, other_input)
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
        return loss

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=0.001)
        return {
            "optimizer": optimizer,
            "gradient_clip_val": 0.5,  # doesn't seem to work
            "gradient_clip_algorithm": "norm",
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        embedding_input, other_input, y = batch
        embedding = self.embedder(embedding_input)
        y_hat = self.regressor(embedding, other_input)
        self.validation_step_outputs.append(y_hat)
        self.validation_step_targets.append(y)

    def on_validation_epoch_end(self):
        all_preds = th.stack(self.validation_step_outputs)
        all_targets = th.stack(self.validation_step_outputs)
        # do something with all preds
        loss, metrics = get_eval_metrics(
            all_targets,
            all_preds,
            self.loss_type,
            self.objectives,
            self.loss_weights,
        )
        self.log("val_loss", loss, prog_bar=True, logger=True)
        # free memory
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear() 


def train(
    device: th.device,
    model: th.nn.Module,
    epochs: int,
    optimizer: th.optim.Optimizer,
    data_loader: DataLoader,
    loss_type: str,
    obj: str,
    loss_ws: th.Tensor,
    warmup_scheduler: BaseWarmup,
    lr_scheduler: th.optim.lr_scheduler.LambdaLR,
    in_feat_minmax: th.Tensor,
    obj_minmax: th.Tensor,
    val_loader: DataLoader,
    writer: SummaryWriter,
    nbatches: int,
):
    ckp_start = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        for batch_idx, x in enumerate(data_loader):
            model.train()
            optimizer.zero_grad()
            batch_y, batch_y_hat = model.forward(x, device, mode="train")
            loss = loss_compute(batch_y, batch_y_hat, loss_type, obj, loss_ws)
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)  # clip the gradient
            optimizer.step()

            with warmup_scheduler.dampening():
                lr_scheduler.step()

            if th.isnan(loss):
                raise ValueError("get a nan loss in train")
            elif th.isinf(loss):
                raise ValueError("get an inf loss in train")

            if batch_idx == (nbatches - 1):
                with th.no_grad():
                    wmape_tr_dict = {
                        m: (
                            th.abs(batch_y[:, i] - batch_y_hat[:, i]).sum()
                            / batch_y[:, i].sum()
                        )
                        .detach()
                        .item()
                        for i, m in enumerate(OBJ_MAP[obj])
                    }
                    batch_time_tr = (time.time() - ckp_start) / nbatches

                t1 = time.time()
                loss_val, m_dict_val = evaluate_model(
                    model,
                    val_loader,
                    device,
                    in_feat_minmax,
                    obj_minmax,
                    loss_type,
                    obj,
                    loss_ws,
                    if_y=False,
                )
                eval_time = time.time() - t1

                tst.update(model, epoch, batch_idx, loss_val)
                cur_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar(
                    "train/_loss",
                    loss.detach().item(),
                    epoch * nbatches + batch_idx,
                )
                writer.add_scalar("val/_loss", loss_val, epoch * nbatches + batch_idx)
                for m in OBJ_MAP[obj]:
                    writer.add_scalar(
                        f"train/_wmape_{m}",
                        wmape_tr_dict[m],
                        epoch * nbatches + batch_idx,
                    )
                    writer.add_scalar(
                        f"val/_wmape_{m}",
                        m_dict_val[m]["wmape"],
                        epoch * nbatches + batch_idx,
                    )
                writer.add_scalar("learning_rate", cur_lr, epoch * nbatches + batch_idx) # replace with learning rate monitor callback in pytorch-lightning
                writer.add_scalar(
                    "batch_time", batch_time_tr, epoch * nbatches + batch_idx
                ) # replace with timing callback in pytorch-lightning

                logger.info(
                    "Epoch {:03d} | Batch {:06d} | LR: {:.8f} | TR Loss {:.6f} | VAL Loss {:.6f} | "
                    "s/ba {:.3f} | s/eval {:.3f} ".format(
                        epoch,
                        batch_idx,
                        cur_lr,
                        loss.detach().item(),
                        loss_val,
                        batch_time_tr,
                        eval_time,
                    )
                )
                logger.info(
                    " \n ".join(
                        [
                            "[{}] TR WMAPE {:.6f} | VAL WAMPE {:.6f} | VAL QErrMean {:.6f} | CORR {:.6f}".format(
                                m,
                                wmape_tr_dict[m],
                                m_dict_val[m]["wmape"],
                                m_dict_val[m]["q_err_mean"],
                                m_dict_val[m]["corr"],
                            )
                            for m in OBJ_MAP[obj]
                        ]
                    )
                )

                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch} cost {epoch_time} s.")
                ckp_start = time.time()


def evaluate(
    ckp_path: str,
    hp_params: dict,
    hp_prefix_sign: str,
    ts: str,
    results_pth_sign: str,
):
    total_time = time.time() - t0
    model.load_state_dict(tst.pop_model_dict(device))
    loss_val, m_dict_val = evaluate_model(
        model, val_loader, device, in_feat_minmax, obj_minmax, loss_type, obj, loss_ws
    )
    loss_te, m_dict_te, y_te, y_hat_te = evaluate_model(
        model,
        te_loader,
        device,
        in_feat_minmax,
        obj_minmax,
        loss_type,
        obj,
        loss_ws,
        if_y=True,
    )

    results = {
        "hp_params": hp_params,
        "Epoch": tst.best_epoch,
        "Batch": tst.best_batch,
        "Total_time": total_time,
        "timestamp": ts,
        "loss_val": loss_val,
        "loss_te": loss_te,
        "metric_val": m_dict_val,
        "metric_te": m_dict_te,
        "y_te": y_te,
        "y_te_hat": y_hat_te,
    }
    th.save(results, results_pth_sign)
    plot_error_rate(y_te, y_hat_te, ckp_path)
    show_results(results, obj)
    return model, results, hp_params, hp_prefix_sign

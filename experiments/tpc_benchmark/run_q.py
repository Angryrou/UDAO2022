from typing import cast

import lightning.pytorch as pl
import pytorch_warmup as warmup
import torch as th
from args import get_graph_avg_args
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import WeightedMeanAbsolutePercentageError

from udao.data import DataHandler, QueryPlanIterator
from udao.model.embedders.graph_averager import GraphAverager
from udao.model.model import UdaoModel
from udao.model.module import LearningParams, UdaoModule
from udao.model.regressors.mlp import MLP
from udao.model.utils.losses import WMAPELoss
from udao.model.utils.schedulers import UdaoLRScheduler, setup_cosine_annealing_lr
from udao.utils.logging import logger
from utils import magic_extract, tensor_dtypes

logger.setLevel("INFO")
if __name__ == "__main__":
    args = get_graph_avg_args()
    device = "gpu" if th.cuda.is_available() else "cpu"
    model_sign = "graph_avg"
    objectives = ["latency_s", "io_mb"]
    th.set_default_dtype(tensor_dtypes)  # type: ignore

    # Data-related arguments
    benchmark = args.benchmark
    q_type = args.q_type
    seed = args.seed
    debug = args.debug

    # Embedder parameters
    lpe_size = args.lpe_size
    output_size = args.output_size
    op_groups = args.op_groups
    type_embedding_dim = args.type_embedding_dim
    vec_size = args.vec_size
    embedding_normalizer = args.embedding_normalizer

    # Regressor parameters
    n_layers = args.n_layers
    hidden_dim = args.hidden_dim
    dropout = args.dropout

    # Learning parameters
    init_lr = args.init_lr
    min_lr = args.min_lr
    weight_decay = args.weight_decay
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Data definition
    data_processor, df, index_splits = magic_extract(
        benchmark=benchmark,
        debug=debug,
        seed=seed,
        q_type=q_type,
        op_groups=op_groups,
        objectives=objectives,
        vec_size=vec_size,
    )

    data_handler = DataHandler(
        df.reset_index(),
        DataHandler.Params(
            index_column="id",
            stratify_on="tid",
            val_frac=0.2 if debug else 0.1,
            test_frac=0.2 if debug else 0.1,
            dryrun=False,
            data_processor=data_processor,
            random_state=seed,
        ),
    )
    data_handler.index_splits = index_splits
    split_iterators = data_handler.get_iterators()

    # Model definition and training

    model = UdaoModel.from_config(
        embedder_cls=GraphAverager,
        regressor_cls=MLP,
        iterator_shape=split_iterators["train"].shape,
        embedder_params={
            "output_size": output_size,  # 32
            "op_groups": op_groups,  # ["type", "cbo", "op_enc"]
            "type_embedding_dim": type_embedding_dim,  # 8
            "embedding_normalizer": embedding_normalizer,  # None
        },
        regressor_params={
            "n_layers": n_layers,  # 3
            "hidden_dim": hidden_dim,  # 512
            "dropout": dropout,  # 0.1
        },
    )
    module = UdaoModule(
        model,
        objectives,
        loss=WMAPELoss(),
        learning_params=LearningParams(
            init_lr=init_lr,  # 1e-3
            min_lr=min_lr,  # 1e-5
            weight_decay=weight_decay,  # 1e-2
        ),
        metrics=[WeightedMeanAbsolutePercentageError],
    )
    tb_logger = TensorBoardLogger("tb_logs")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{benchmark}_{q_type}_{model_sign}",
        filename="{epoch}"
        "-val_lat_WMAPE={val_latency_s_WeightedMeanAbsolutePercentageError:.3f}"
        "-val_io_WMAPE={val_io_mb_WeightedMeanAbsolutePercentageError:.3f}",
        auto_insert_metric_name=False,
    )
    train_iterator = cast(QueryPlanIterator, split_iterators["train"])
    scheduler = UdaoLRScheduler(setup_cosine_annealing_lr, warmup.UntunedLinearWarmup)
    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=epochs,
        logger=tb_logger,
        callbacks=[scheduler, checkpoint_callback],
    )
    trainer.fit(
        model=module,
        train_dataloaders=split_iterators["train"].get_dataloader(
            batch_size, num_workers=0 if debug else num_workers, shuffle=True
        ),
        val_dataloaders=split_iterators["val"].get_dataloader(
            batch_size, num_workers=0 if debug else num_workers, shuffle=False
        ),
    )

    test_results = trainer.test(
        model=module,
        dataloaders=split_iterators["test"].get_dataloader(
            batch_size, num_workers=0 if debug else num_workers, shuffle=False
        ),
    )
    print(test_results)

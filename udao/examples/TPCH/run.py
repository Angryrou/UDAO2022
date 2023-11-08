from pathlib import Path

import lightning as pl
import pandas as pd
import pytorch_warmup as warmup
import torch as th
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torchmetrics import WeightedMeanAbsolutePercentageError
from udao.data.extractors import PredicateEmbeddingExtractor, QueryStructureExtractor
from udao.data.extractors.tabular_extractor import TabularFeatureExtractor
from udao.data.handler.data_handler import (
    DataHandler,
    FeaturePipeline,
    create_data_handler_params,
)
from udao.data.iterators import QueryPlanIterator
from udao.data.predicate_embedders import Word2VecEmbedder
from udao.data.preprocessors.normalize_preprocessor import NormalizePreprocessor
from udao.model.embedders.graph_averager import GraphAverager, GraphAveragerParams
from udao.model.model import UdaoModel
from udao.model.module import UdaoModule
from udao.model.regressors.mlp import MLP, MLPParams
from udao.model.utils.losses import WMAPELoss
from udao.model.utils.schedulers import UdaoLRScheduler, setup_cosine_annealing_lr

if __name__ == "__main__":
    tensor_dtypes = th.float32
    device = "gpu" if th.cuda.is_available() else "cpu"
    batch_size = 32

    th.set_default_dtype(tensor_dtypes)  # type: ignore
    #### Data definition ####
    params_getter = create_data_handler_params(QueryPlanIterator, "op_emb")
    params = params_getter(
        index_column="id",
        stratify_on="tid",
        dryrun=True,
        tensor_dtypes=tensor_dtypes,
        tabular_features=FeaturePipeline(
            extractor=(
                TabularFeatureExtractor,
                [
                    lambda df: df[
                        ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "s1", "s2"]
                    ]
                ],
            ),
        ),
        objectives=FeaturePipeline(
            extractor=(TabularFeatureExtractor, [lambda df: df[["latency"]]]),
        ),
        query_structure=FeaturePipeline(
            extractor=(QueryStructureExtractor, []),
            preprocessors=[(NormalizePreprocessor, [MinMaxScaler(), "graph_features"])],
        ),
        op_emb=FeaturePipeline(
            extractor=(PredicateEmbeddingExtractor, [Word2VecEmbedder()]),
        ),
    )

    base_dir = Path(__file__).parent
    lqp_df = pd.read_csv(str(base_dir / "data/LQP.csv"))
    brief_df = pd.read_csv(str(base_dir / "data/brief.csv"))
    cols_to_use = lqp_df.columns.difference(brief_df.columns)

    df = brief_df.merge(
        lqp_df[["id", *cols_to_use]],
        on="id",
    )
    data_handler = DataHandler(df, params)

    split_iterators = data_handler.get_iterators()

    #### Model definition ####

    # Todo: make below parameters more automated from data properties
    # (e.g. dimension of inputs)
    # extract some dimensions from the data directly
    # expect a dimension dataclass that iterator should implement

    embedder = GraphAverager(
        GraphAveragerParams(
            input_size=2,
            n_op_types=4,
            output_size=10,
            op_groups=["ch1_cbo"],
            type_embedding_dim=5,
            embedding_normalizer="BN",
        )
    )
    regressor = MLP(
        MLPParams(
            input_embedding_dim=10,
            input_features_dim=12,
            output_dim=1,
            n_layers=2,
            hidden_dim=2,
            dropout=0,
        )
    )

    model = UdaoModel(embedder=embedder, regressor=regressor)
    module = UdaoModule(
        model,
        ["latency"],
        loss=WMAPELoss(),
        metrics=[WeightedMeanAbsolutePercentageError()],
    )
    tb_logger = TensorBoardLogger("tb_logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch}-val_WMAPE={val_latency_WeightedMeanAbsolutePercentageError:.2f}",
        auto_insert_metric_name=False,
    )

    scheduler = UdaoLRScheduler(setup_cosine_annealing_lr, warmup.UntunedLinearWarmup)
    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=2,
        logger=tb_logger,
        callbacks=[scheduler, checkpoint_callback],
    )
    trainer.fit(
        model=module,
        train_dataloaders=split_iterators["train"].get_dataloader(batch_size),
        val_dataloaders=split_iterators["val"].get_dataloader(batch_size),
    )

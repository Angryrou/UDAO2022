from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import pytorch_warmup as warmup
import torch as th
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torchmetrics import WeightedMeanAbsolutePercentageError
from udao.data.extractors import PredicateEmbeddingExtractor, QueryStructureExtractor
from udao.data.extractors.tabular_extractor import TabularFeatureExtractor
from udao.data.handler.data_handler import DataHandler, DataHandlerParams
from udao.data.handler.data_processor import FeaturePipeline, create_data_processor
from udao.data.iterators import QueryPlanIterator
from udao.data.predicate_embedders import Word2VecEmbedder, Word2VecParams
from udao.data.preprocessors.normalize_preprocessor import NormalizePreprocessor
from udao.model.embedders.graph_averager import GraphAverager
from udao.model.model import UdaoModel
from udao.model.module import UdaoModule
from udao.model.regressors.mlp import MLP
from udao.model.utils.losses import WMAPELoss
from udao.model.utils.schedulers import UdaoLRScheduler, setup_cosine_annealing_lr

if __name__ == "__main__":
    tensor_dtypes = th.float32
    device = "gpu" if th.cuda.is_available() else "cpu"
    batch_size = 32

    th.set_default_dtype(tensor_dtypes)  # type: ignore
    #### Data definition ####
    processor_getter = create_data_processor(QueryPlanIterator, "op_enc")
    data_processor = processor_getter(
        tensor_dtypes=tensor_dtypes,
        tabular_features=FeaturePipeline(
            extractor=TabularFeatureExtractor(lambda df: df[
                        [
                            "k1",
                            "k2",
                            "k3",
                            "k4",
                            "k5",
                            "k6",
                            "k7",
                            "k8",
                            "s1",
                            "s2",
                            "s3",
                            "s4",
                        ]
                        + ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]
                ])
        ),
        objectives=FeaturePipeline(
            extractor=TabularFeatureExtractor(lambda df: df[["latency"]])
        ),
        query_structure=FeaturePipeline(
            extractor=QueryStructureExtractor(),
            preprocessors=[
                NormalizePreprocessor(MinMaxScaler(), "graph_features"),
                NormalizePreprocessor(MinMaxScaler(), "graph_meta_features"),
            ],
        ),
        op_enc=FeaturePipeline(
            extractor=PredicateEmbeddingExtractor(Word2VecEmbedder(Word2VecParams(vec_size=32))),
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
    data_handler = DataHandler(
        df,
        DataHandlerParams(
            index_column="id",
            stratify_on="tid",
            dryrun=True,
            data_processor=data_processor,
        ),
    )

    split_iterators = data_handler.get_iterators()
    #### Model definition ####

    model = UdaoModel.from_config(
        embedder_cls=GraphAverager,
        regressor_cls=MLP,
        iterator_shape=split_iterators["train"].get_iterator_shape(),
        embedder_params={
            "output_size": 16,
            "op_groups": ["cbo", "op_enc", "type"],
            "type_embedding_dim": 5,
            "embedding_normalizer": "BN",
        },
        regressor_params={"n_layers": 2, "hidden_dim": 32, "dropout": 0.1},
    )
    module = UdaoModule(
        model,
        ["latency"],
        loss=WMAPELoss(),
        metrics=[WeightedMeanAbsolutePercentageError],
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

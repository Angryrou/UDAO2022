from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import lightning.pytorch as pl
import pandas as pd
import pytorch_warmup as warmup
import torch as th
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torchmetrics import WeightedMeanAbsolutePercentageError
from udao.data import TabularIterator, QueryStructureContainer

from udao.data.containers import TabularContainer
from udao.data.extractors import PredicateEmbeddingExtractor, QueryStructureExtractor
from udao.data.extractors.tabular_extractor import TabularFeatureExtractor
from udao.data.handler.data_handler import DataHandler
from udao.data.handler.data_processor import FeaturePipeline, create_data_processor, DataProcessor
from udao.data.iterators.query_plan_iterator import QueryPlanIterator
from udao.data.predicate_embedders import Word2VecEmbedder, Word2VecParams
from udao.data.predicate_embedders.utils import prepare_operation
from udao.data.preprocessors.base_preprocessor import StaticPreprocessor
from udao.data.preprocessors.normalize_preprocessor import NormalizePreprocessor
from udao.data.utils.utils import DatasetType
from udao.model.embedders.graph_averager import GraphAverager
from udao.model.model import UdaoModel
from udao.model.module import LearningParams, UdaoModule
from udao.model.regressors.mlp import MLP
from udao.model.utils.losses import WMAPELoss
from udao.model.utils.schedulers import UdaoLRScheduler, setup_cosine_annealing_lr
from udao.optimization import concepts
from udao.optimization.soo.mogd import MOGD
from udao.utils.interfaces import VarTypes
from udao.utils.logging import logger

from udao_trace.configuration import SparkConf
from utils import *

logger.setLevel("INFO")
if __name__ == "__main__":
    tensor_dtypes = th.float32
    device = "gpu" if th.cuda.is_available() else "cpu"
    batch_size = 512
    benchmark = "tpch"
    objectives = ["latency_s", "io_mb"]
    model_sign = "graph_avg"
    th.set_default_dtype(tensor_dtypes)

    #### Data definition ####
    base_dir = Path(__file__).parent
    df = pd.read_csv(str(base_dir / "data" / benchmark / "q_22x2273.csv"))
    logger.info(f"Data shape: {df.shape}")
    df, variable_names = prepare(
        df,
        knob_meta_file=str(base_dir / "assets/spark_configuration_aqe_on.json"),
        alpha=ALPHA_LQP
    )

    # for compile-time df
    df_compile = df[df["lqp_id"] == 0]
    logger.info(f"Data (compile-time) shape: {df_compile.shape}")
    data_processor_getter = create_data_processor(QueryPlanIterator, "op_enc")
    data_processor = data_processor_getter(
        tensor_dtypes=tensor_dtypes,
        tabular_features=FeaturePipeline(
            extractor=TabularFeatureExtractor(columns=variable_names + ALPHA_LQP + BETA + GAMMA),
            preprocessors=[NormalizePreprocessor(MinMaxScaler())]
        ),
        objectives=FeaturePipeline(
            extractor=TabularFeatureExtractor(columns=objectives),
        ),
        query_structure=FeaturePipeline(
            extractor=LQPExtractor(positional_encoding_size=10),
            preprocessors=[
                NormalizePreprocessor(MinMaxScaler(), "graph_features")
            ]
        ),
        op_enc=FeaturePipeline(
            extractor=PredicateEmbeddingExtractor(
                Word2VecEmbedder(Word2VecParams(vec_size=16)),
                extract_operations=extract_operations_from_seralized_json,
            ),
        )
    )

    data_handler = DataHandler(
        df_compile,
        DataHandler.Params(
            index_column="id",
            stratify_on="tid",
            dryrun=False,
            data_processor=data_processor
        )
    )
    split_iterators = data_handler.get_iterators()

    #### Model definition and training ####

    model = UdaoModel.from_config(
        embedder_cls=GraphAverager,
        regressor_cls=MLP,
        iterator_shape=split_iterators["train"].shape,
        embedder_params={
            "output_size": 128,
            "op_groups": ["type", "cbo", "op_enc"],
            "type_embedding_dim": 8,
            "embedding_normalizer": None,
        },
        regressor_params={"n_layers": 3, "hidden_dim": 512, "dropout": 0.1},
    )
    module = UdaoModule(
        model,
        objectives,
        loss=WMAPELoss(),
        learning_params=LearningParams(init_lr=1e-3, min_lr=1e-5, weight_decay=1e-2),
        metrics=[WeightedMeanAbsolutePercentageError],
    )
    tb_logger = TensorBoardLogger("tb_logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{benchmark}-{model_sign}-{epoch}"
                 "-val_obj1_WMAPE={val_latency_s_WeightedMeanAbsolutePercentageError:.3f}"
                 "-val_obj2_WMAPE={val_io_mb_WeightedMeanAbsolutePercentageError:.3f}",
        auto_insert_metric_name=False,
    )
    train_iterator = cast(QueryPlanIterator, split_iterators["train"])
    scheduler = UdaoLRScheduler(setup_cosine_annealing_lr, warmup.UntunedLinearWarmup)
    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=100,
        logger=tb_logger,
        callbacks=[scheduler, checkpoint_callback],
    )
    trainer.fit(
        model=module,
        train_dataloaders=split_iterators["train"].get_dataloader(batch_size, num_workers=15, shuffle=True),
        val_dataloaders=split_iterators["val"].get_dataloader(batch_size, num_workers=15, shuffle=False),
    )

    print(trainer.test(
        model=module,
        dataloaders=split_iterators["test"].get_dataloader(batch_size, num_workers=15, shuffle=False)
    ))

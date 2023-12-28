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

import argparse

from udao_trace.utils import BenchmarkType
from udao_trace.workload import Benchmark
from utils import *

logger.setLevel("INFO")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Udao Script with Input Arguments")
    # Data-related arguments
    parser.add_argument("--benchmark", type=str, default="tpch", help="Benchmark name")

    # Embedder parameters
    parser.add_argument("--lpe_size", type=int, default=8, help="Laplacian Positional encoding size - for GTN only")
    parser.add_argument("--output_size", type=int, default=32, help="Embedder output size")
    parser.add_argument("--op_groups", nargs='+', default=["type", "cbo", "op_enc"], help="List of operation groups")
    parser.add_argument("--type_embedding_dim", type=int, default=8, help="Type embedding dimension")
    parser.add_argument("--vec_size", type=int, default=16, help="Word2Vec embedding size")
    parser.add_argument("--embedding_normalizer", type=str, default=None, help="Embedding normalizer")

    # Regressor parameters
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers in the regressor")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension of the regressor")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Learning parameters
    parser.add_argument("--init_lr", type=float, default=1e-1, help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=15, help="non-debug only")

    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()
    tensor_dtypes = th.float32
    device = "gpu" if th.cuda.is_available() else "cpu"
    model_sign = "graph_avg"
    objectives = ["latency_s", "io_mb"]
    th.set_default_dtype(tensor_dtypes)

    # Data-related arguments
    benchmark = args.benchmark
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

    base_dir = Path(__file__).parent
    if benchmark == "tpch":
        df_path = str(base_dir / f"data/tpch/q_22x{10 if debug else 2273}.csv")
    elif benchmark == "tpcds":
        df_path = str(base_dir / f"data/tpcds/q_102x{4 if debug else 490}.csv")
    else:
        raise NoBenchmarkError

    df_raw = pd.read_csv(df_path)
    logger.info(f"df_raw Data shape: {df_raw.shape}")
    df, variable_names = prepare(
        df_raw,
        benchmark=benchmark,
        knob_meta_file=str(base_dir / "assets/spark_configuration_aqe_on.json"),
        mode="q"
    )
    df_compile = df[df["lqp_id"] == 0]  # for compile-time df
    logger.info(f"Data (compile-time) shape: {df_compile.shape}")
    if "op_enc" in op_groups:
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
                extractor=LQPExtractor(positional_encoding_size=None),
                preprocessors=[
                    NormalizePreprocessor(MinMaxScaler(), "graph_features")
                ]
            ),
            op_enc=FeaturePipeline(
                extractor=PredicateEmbeddingExtractor(
                    Word2VecEmbedder(Word2VecParams(vec_size=vec_size)),
                    extract_operations=extract_operations_from_seralized_json,
                ),
            )
        )
    else:
        data_processor_getter = create_data_processor(QueryPlanIterator)
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
                extractor=LQPExtractor(positional_encoding_size=None),
                preprocessors=[
                    NormalizePreprocessor(MinMaxScaler(), "graph_features")
                ]
            )
        )

    if debug:
        # make sure we have enough data in each tid group for train/val/test split
        df_compile = df_compile.groupby("tid").filter(lambda x: len(x) >= 5)

    data_handler = DataHandler(
        df_compile,
        DataHandler.Params(
            index_column="id",
            stratify_on="tid",
            val_frac=0.2 if debug else 0.1,
            test_frac=0.2 if debug else 0.1,
            dryrun=False,
            data_processor=data_processor
        )
    )
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
            "dropout": dropout  # 0.1
        },
    )
    module = UdaoModule(
        model,
        objectives,
        loss=WMAPELoss(),
        learning_params=LearningParams(
            init_lr=init_lr,  # 1e-3
            min_lr=min_lr,  # 1e-5
            weight_decay=weight_decay  # 1e-2
        ),
        metrics=[WeightedMeanAbsolutePercentageError],
    )
    tb_logger = TensorBoardLogger("tb_logs")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{benchmark}_{model_sign}",
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
            batch_size, num_workers=0 if debug else num_workers, shuffle=True),
        val_dataloaders=split_iterators["val"].get_dataloader(
            batch_size, num_workers=0 if debug else num_workers, shuffle=False),
    )

    print(trainer.test(
        model=module,
        dataloaders=split_iterators["test"].get_dataloader(
            batch_size, num_workers=0 if debug else num_workers, shuffle=False))
    )

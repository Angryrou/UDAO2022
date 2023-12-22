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

from udao.data.containers import TabularContainer
from udao.data.extractors import PredicateEmbeddingExtractor, QueryStructureExtractor
from udao.data.extractors.tabular_extractor import TabularFeatureExtractor
from udao.data.handler.data_handler import DataHandler
from udao.data.handler.data_processor import FeaturePipeline, create_data_processor
from udao.data.iterators.query_plan_iterator import QueryPlanIterator
from udao.data.predicate_embedders import Word2VecEmbedder, Word2VecParams
from udao.data.preprocessors.base_preprocessor import StaticPreprocessor
from udao.data.preprocessors.normalize_preprocessor import NormalizePreprocessor
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

from utils import *

class SparkConfPreprocessor(StaticPreprocessor):
    def preprocess(self, container: TabularContainer) -> TabularContainer:
        return container

    def inverse_transform(self, X: TabularContainer) -> TabularContainer:
        return X


class LQPExtractor(QueryStructureExtractor):
    ...


logger.setLevel("INFO")
if __name__ == "__main__":
    tensor_dtypes = th.float32
    device = "gpu" if th.cuda.is_available() else "cpu"
    batch_size = 512
    th.set_default_dtype(tensor_dtypes)

    #### Data definition ####
    processor_getter = create_data_processor(QueryPlanIterator, "op_enc")
    data_processor = processor_getter(
        tensor_dtypes=tensor_dtypes,
        tabular_features=FeaturePipeline(
            extractor=TabularFeatureExtractor(columns=LQP_FEATURES),
            preprocessors=[
                SparkConfPreprocessor(),
                NormalizePreprocessor(MinMaxScaler())
            ],
        ),
        objectives=FeaturePipeline(
            extractor=TabularFeatureExtractor(["latency", "io"]),
        ),
        query_structure=FeaturePipeline(
            extractor=LQPExtractor(positional_encoding_size=10),
            preprocessors=[
                NormalizePreprocessor(MinMaxScaler(), "graph_features"),
            ],
        ),
        op_enc=FeaturePipeline(
            extractor=PredicateEmbeddingExtractor(
                Word2VecEmbedder(Word2VecParams(vec_size=32))
            ),
        ),
    )

    base_dir = Path(__file__).parent
    df = pd.read_csv(str(base_dir / "data/q_22x10.csv"))
    logger.info(f"Data shape: {df.shape}")

    # for compile-time df
    df_compile = df[df["lqp_id"] == 0]
    logger.info(f"Data (compile-time) shape: {df_compile.shape}")

    data_handler = DataHandler(
        df,
        DataHandler.Params(
            index_column="appid",
            stratify_on="tid",
            dryrun=False,
            data_processor=data_processor
        )
    )
    split_iterators = data_handler.get_iterators()

    # for all-set of df

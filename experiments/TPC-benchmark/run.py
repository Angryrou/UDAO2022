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
from udao.data.handler.data_handler import DataHandler, DataHandlerParams
from udao.data.handler.data_processor import FeaturePipeline, create_data_processor
from udao.data.iterators.query_plan_iterator import QueryPlanIterator
from udao.data.predicate_embedders import Word2VecEmbedder, Word2VecParams
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

LQP_FEATURES = [
    'IM-inputSizeInBytes',
    'IM-inputRowCount',
    'PD',
    'SS-RunningTasksNum',
    'SS-FinishedTasksNum',
    'SS-FinishedTasksTotalTimeInMs',
    'SS-FinishedTasksDistributionInMs-0tile',
    'SS-FinishedTasksDistributionInMs-25tile',
    'SS-FinishedTasksDistributionInMs-50tile',
    'SS-FinishedTasksDistributionInMs-75tile',
    'SS-FinishedTasksDistributionInMs-100tile',
    'theta_c-spark.executor.cores',
    'theta_c-spark.executor.memory',
    'theta_c-spark.executor.instances',
    'theta_c-spark.default.parallelism',
    'theta_c-spark.reducer.maxSizeInFlight',
    'theta_c-spark.shuffle.sort.bypassMergeThreshold',
    'theta_c-spark.shuffle.compress',
    'theta_c-spark.memory.fraction',
    'theta_p-spark.sql.adaptive.advisoryPartitionSizeInBytes',
    'theta_p-spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin',
    'theta_p-spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold',
    'theta_p-spark.sql.adaptive.autoBroadcastJoinThreshold',
    'theta_p-spark.sql.shuffle.partitions',
    'theta_p-spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes',
    'theta_p-spark.sql.adaptive.skewJoin.skewedPartitionFactor',
    'theta_p-spark.sql.files.maxPartitionBytes',
    'theta_p-spark.sql.files.openCostInBytes',
    'theta_s-spark.sql.adaptive.rebalancePartitionsSmallPartitionFactor',
    'theta_s-spark.sql.adaptive.coalescePartitions.minPartitionSize'
]

QS_FEATURES = [
    'InitialPartitionNum',
    'IM-inputSizeInBytes',
    'IM-inputRowCount',
    'PD',
    'SS-RunningTasksNum',
    'SS-FinishedTasksNum',
    'SS-FinishedTasksTotalTimeInMs',
    'SS-FinishedTasksDistributionInMs-0tile',
    'SS-FinishedTasksDistributionInMs-25tile',
    'SS-FinishedTasksDistributionInMs-50tile',
    'SS-FinishedTasksDistributionInMs-75tile',
    'SS-FinishedTasksDistributionInMs-100tile',
    'theta_c-spark.executor.cores',
    'theta_c-spark.executor.memory',
    'theta_c-spark.executor.instances',
    'theta_c-spark.default.parallelism',
    'theta_c-spark.reducer.maxSizeInFlight',
    'theta_c-spark.shuffle.sort.bypassMergeThreshold',
    'theta_c-spark.shuffle.compress',
    'theta_c-spark.memory.fraction',
    'theta_p-spark.sql.adaptive.advisoryPartitionSizeInBytes',
    'theta_p-spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin',
    'theta_p-spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold',
    'theta_p-spark.sql.adaptive.autoBroadcastJoinThreshold',
    'theta_p-spark.sql.shuffle.partitions',
    'theta_p-spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes',
    'theta_p-spark.sql.adaptive.skewJoin.skewedPartitionFactor',
    'theta_p-spark.sql.files.maxPartitionBytes',
    'theta_p-spark.sql.files.openCostInBytes',
    'theta_s-spark.sql.adaptive.rebalancePartitionsSmallPartitionFactor',
    'theta_s-spark.sql.adaptive.coalescePartitions.minPartitionSize'
]


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
            extractor=TabularFeatureExtractor(
                columns=LQP_FEATURES,
            ),
            preprocessors=[NormalizePreprocessor(MinMaxScaler())],
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

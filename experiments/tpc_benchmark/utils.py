from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th
from sklearn.preprocessing import MinMaxScaler
from udao_trace.configuration import SparkConf
from udao_trace.utils import BenchmarkType, JsonHandler, ParquetHandler, PickleHandler
from udao_trace.workload import Benchmark

from udao.data import (
    DataProcessor,
    NormalizePreprocessor,
    PredicateEmbeddingExtractor,
    QueryPlanIterator,
    QueryStructureContainer,
    QueryStructureExtractor,
    TabularFeatureExtractor,
)
from udao.data.handler.data_processor import FeaturePipeline, create_data_processor
from udao.data.predicate_embedders import Word2VecEmbedder, Word2VecParams
from udao.data.predicate_embedders.utils import build_unique_operations
from udao.data.utils.query_plan import QueryPlanOperationFeatures, QueryPlanStructure
from udao.data.utils.utils import DatasetType, train_test_val_split_on_column
from udao.utils.logging import logger

tensor_dtypes = th.float32

THETA_RAW = [
    "theta_c-spark.executor.cores",
    "theta_c-spark.executor.memory",
    "theta_c-spark.executor.instances",
    "theta_c-spark.default.parallelism",
    "theta_c-spark.reducer.maxSizeInFlight",
    "theta_c-spark.shuffle.sort.bypassMergeThreshold",
    "theta_c-spark.shuffle.compress",
    "theta_c-spark.memory.fraction",
    "theta_p-spark.sql.adaptive.advisoryPartitionSizeInBytes",
    "theta_p-spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin",
    "theta_p-spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold",
    "theta_p-spark.sql.adaptive.autoBroadcastJoinThreshold",
    "theta_p-spark.sql.shuffle.partitions",
    "theta_p-spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes",
    "theta_p-spark.sql.adaptive.skewJoin.skewedPartitionFactor",
    "theta_p-spark.sql.files.maxPartitionBytes",
    "theta_p-spark.sql.files.openCostInBytes",
    "theta_s-spark.sql.adaptive.rebalancePartitionsSmallPartitionFactor",
    "theta_s-spark.sql.adaptive.coalescePartitions.minPartitionSize",
]
ALPHA_LQP_RAW = [
    "IM-inputSizeInBytes",
    "IM-inputRowCount",
]
ALPHA_QS_RAW = ["InitialPartitionNum"] + ALPHA_LQP_RAW
BETA_RAW = ["PD"]
THETA = [
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
    "s5",
    "s6",
    "s7",
    "s8",
    "s9",
    "s10",
    "s11",
]
ALPHA_LQP = [
    "IM-sizeInMB",
    "IM-rowCount",
    "IM-sizeInMB-log",
    "IM-rowCount-log",
]
ALPHA_QS = ["IM-init-part-num", "IM-init-part-num-log"] + ALPHA_LQP
BETA = ["PD-std-avg", "PD-skewness-ratio", "PD-range-avg-ratio"]
GAMMA = [
    "SS-RunningTasksNum",
    "SS-FinishedTasksNum",
    "SS-FinishedTasksTotalTimeInMs",
    "SS-FinishedTasksDistributionInMs-0tile",
    "SS-FinishedTasksDistributionInMs-25tile",
    "SS-FinishedTasksDistributionInMs-50tile",
    "SS-FinishedTasksDistributionInMs-75tile",
    "SS-FinishedTasksDistributionInMs-100tile",
]
TABULAR_LQP = THETA + ALPHA_LQP + BETA + GAMMA
TABULAR_QS = THETA + ALPHA_QS + BETA + GAMMA

EPS = 1e-3


class NoBenchmarkError(ValueError):
    """raise when no valid benchmark is found"""


class NoQTypeError(ValueError):
    """raise when no valid mode is found (only q and qs)"""


class OperatorMisMatchError(BaseException):
    """raise when the operator names from `operator` and `link` do not match"""


# Data Processing
def _im_process(df: pd.DataFrame) -> pd.DataFrame:
    df["IM-sizeInMB"] = df["IM-inputSizeInBytes"] / 1024 / 1024
    df["IM-sizeInMB-log"] = np.log(df["IM-sizeInMB"].to_numpy().clip(min=EPS))
    df["IM-rowCount"] = df["IM-inputRowCount"]
    df["IM-rowCount-log"] = np.log(df["IM-rowCount"].to_numpy().clip(min=EPS))
    for c in ALPHA_LQP_RAW:
        del df[c]
    return df


def prepare(
    df: pd.DataFrame, sc: SparkConf, benchmark: str, q_type: str
) -> pd.DataFrame:
    bm = Benchmark(benchmark_type=BenchmarkType[benchmark.upper()])
    df.rename(columns={p: kid for p, kid in zip(THETA_RAW, sc.knob_ids)}, inplace=True)
    df["tid"] = df["template"].apply(lambda x: bm.get_template_id(str(x)))
    variable_names = sc.knob_ids
    if variable_names != THETA:
        raise ValueError(f"variable_names != THETA: {variable_names} != {THETA}")
    df[variable_names] = sc.deconstruct_configuration(
        df[variable_names].astype(str).values
    )

    # extract alpha
    if q_type == "q":
        df[ALPHA_LQP_RAW] = df[ALPHA_LQP_RAW].astype(float)
        df = _im_process(df)
    elif q_type == "qs":
        df[ALPHA_QS_RAW] = df[ALPHA_QS_RAW].astype(float)
        df = _im_process(df)
        df["IM-init-part-num"] = df["InitialPartitionNum"].astype(float)
        df["IM-init-part-num-log"] = np.log(
            df["IM-init-part-num"].to_numpy().clip(min=EPS)
        )
        del df["InitialPartitionNum"]
    else:
        raise NoQTypeError

    # extract beta
    df[BETA] = [
        sc.extract_partition_distribution(pd_raw)
        for pd_raw in df[BETA_RAW].values.squeeze()
    ]
    for c in BETA_RAW:
        del df[c]

    # extract gamma:
    df[GAMMA] = df[GAMMA].astype(float)

    return df


def define_index_with_columns(df: pd.DataFrame, columns: List[str]) -> None:
    if "id" in df.columns:
        raise Exception("id column already exists!")
    df["id"] = df[columns].astype(str).apply("-".join, axis=1)
    df.set_index("id", inplace=True)


def save_and_log_index(
    index_splits: Dict,
    tabular_columns: List,
    cache_header: str,
    name: str,
    debug: bool = False,
) -> None:
    try:
        PickleHandler.save(
            {
                "index_splits": index_splits,
                "tabular_columns": tabular_columns,
            },
            cache_header,
            name,
        )
    except FileExistsError as e:
        if not debug:
            raise e
        logger.warning(f"skip saving {name}")
    lengths = [str(len(index_splits[split])) for split in ["train", "val", "test"]]
    logger.info(f"got index in {name}, tr/val/te={'/'.join(lengths)}")


def save_and_log_df(
    df: pd.DataFrame,
    index_columns: List[str],
    cache_header: str,
    name: str,
    debug: bool,
) -> None:
    define_index_with_columns(df, columns=index_columns)
    try:
        ParquetHandler.save(df, cache_header, f"{name}.parquet")
    except FileExistsError as e:
        if not debug:
            raise e
        logger.warning(f"skip saving {name}.parquet")
    logger.info(f"prepared {name} shape: {df.shape}")


def magic_setup(
    cache_header: str,
    benchmark: str,
    debug: bool,
    seed: int,
) -> None:
    """magic set to make sure
    1. data has been properly processed and effectively saved.
    2. data split to make sure q_compile/q/qs share the same appid for tr/val/te.
    """

    base_dir = Path(__file__).parent
    if benchmark == "tpch":
        df_q_path = str(base_dir / f"data/tpch/q_22x{10 if debug else 2273}.csv")
        df_qs_path = str(base_dir / f"data/tpch/qs_22x{10 if debug else 2273}.csv")
    elif benchmark == "tpcds":
        df_q_path = str(base_dir / f"data/tpcds/q_102x{10 if debug else 490}.csv")
        df_qs_path = str(base_dir / f"data/tpcds/qs_102x{10 if debug else 490}.csv")
    else:
        raise NoBenchmarkError
    df_q_raw = pd.read_csv(df_q_path)
    df_qs_raw = pd.read_csv(df_qs_path)
    logger.info(f"raw df_q shape: {df_q_raw.shape}")
    logger.info(f"raw df_qs shape: {df_qs_raw.shape}")

    # Prepare data
    sc = SparkConf(str(base_dir / "assets/spark_configuration_aqe_on.json"))
    df_q = prepare(df_q_raw, benchmark=benchmark, sc=sc, q_type="q")
    df_qs = prepare(df_qs_raw, benchmark=benchmark, sc=sc, q_type="qs")
    df_q_compile = df_q[df_q["lqp_id"] == 0]  # for compile-time df
    df_rare = df_q_compile.groupby("tid").filter(lambda x: len(x) < 5)
    if df_rare.shape[0] > 0:
        logger.warning(f"Drop rare templates: {df_rare['tid'].unique()}")
        df_q_compile = df_q_compile.groupby("tid").filter(lambda x: len(x) >= 5)
    else:
        logger.info("No rare templates")

    # Compute the index for df_q_compile, df_q and df_qs
    save_and_log_df(df_q_compile, ["appid"], cache_header, "df_q_compile", debug)
    save_and_log_df(df_q, ["appid", "lqp_id"], cache_header, "df_q", debug)
    save_and_log_df(df_qs, ["appid", "qs_id"], cache_header, "df_qs", debug)

    # Split data for df_q_compile
    df_splits_q_compile = train_test_val_split_on_column(
        df=df_q_compile,
        groupby_col="tid",
        val_frac=0.2 if debug else 0.1,
        test_frac=0.2 if debug else 0.1,
        random_state=seed,
    )
    index_splits_q_compile = {
        split: df.index.to_list() for split, df in df_splits_q_compile.items()
    }
    save_and_log_index(
        index_splits_q_compile,
        tabular_columns=TABULAR_LQP,
        cache_header=cache_header,
        name="misc_q_compile.pkl",
        debug=debug,
    )

    index_splits_q = {
        split: df_q[df_q.appid.isin(appid_list)].index.to_list()
        for split, appid_list in index_splits_q_compile.items()
    }
    save_and_log_index(
        index_splits_q,
        tabular_columns=TABULAR_LQP,
        cache_header=cache_header,
        name="misc_q.pkl",
        debug=debug,
    )

    index_splits_qs = {
        split: df_qs[df_qs.appid.isin(appid_list)].index.to_list()
        for split, appid_list in index_splits_q_compile.items()
    }
    save_and_log_index(
        index_splits_qs,
        tabular_columns=TABULAR_QS,
        cache_header=cache_header,
        name="misc_qs.pkl",
        debug=debug,
    )


# Data Split Index
def magic_extract(
    benchmark: str, debug: bool, seed: int, q_type: str, **kwargs: Any
) -> Tuple[DataProcessor, pd.DataFrame, Dict]:
    # Read data
    base_dir = Path(__file__).parent
    if benchmark == "tpch":
        cache_header = str(base_dir / f"data/tpch/cache_22x{10 if debug else 2273}")
    elif benchmark == "tpcds":
        cache_header = str(base_dir / f"data/tpcds/cache_22x{10 if debug else 490}")
    else:
        raise NoBenchmarkError
    if not Path(cache_header).exists():
        magic_setup(cache_header, benchmark, debug, seed)

    if not Path(f"{cache_header}/misc_{q_type}.pkl").exists():
        raise FileNotFoundError(f"{cache_header}/misc_{q_type}.pkl not found")
    if not Path(f"{cache_header}/df_{q_type}.parquet").exists():
        raise FileNotFoundError(f"{cache_header}/df_{q_type}.parquet not found")

    df = ParquetHandler.load(cache_header, f"df_{q_type}.parquet")
    misc = PickleHandler.load(cache_header, f"misc_{q_type}.pkl")
    if not isinstance(misc, Dict):
        raise TypeError(f"misc is not a dict: {misc}")
    index_splits = misc["index_splits"]
    if not isinstance(index_splits, Dict):
        raise TypeError(f"index_splits is not a dict: {index_splits}")
    tabular_columns = misc["tabular_columns"]

    if "op_enc" in kwargs["op_groups"]:
        data_processor_getter = create_data_processor(QueryPlanIterator, "op_enc")
        data_processor = data_processor_getter(
            tensor_dtypes=tensor_dtypes,
            tabular_features=FeaturePipeline(
                extractor=TabularFeatureExtractor(columns=tabular_columns),
                preprocessors=[NormalizePreprocessor(MinMaxScaler())],
            ),
            objectives=FeaturePipeline(
                extractor=TabularFeatureExtractor(columns=kwargs["objectives"]),
            ),
            query_structure=FeaturePipeline(
                extractor=LQPExtractor(positional_encoding_size=None),
                preprocessors=[NormalizePreprocessor(MinMaxScaler(), "graph_features")],
            ),
            op_enc=FeaturePipeline(
                extractor=PredicateEmbeddingExtractor(
                    Word2VecEmbedder(Word2VecParams(vec_size=kwargs["vec_size"])),
                    extract_operations=extract_operations_from_serialized_json,
                ),
            ),
        )
    else:
        data_processor_getter = create_data_processor(QueryPlanIterator)
        data_processor = data_processor_getter(
            tensor_dtypes=tensor_dtypes,
            tabular_features=FeaturePipeline(
                extractor=TabularFeatureExtractor(columns=tabular_columns),
                preprocessors=[NormalizePreprocessor(MinMaxScaler())],
            ),
            objectives=FeaturePipeline(
                extractor=TabularFeatureExtractor(columns=kwargs["objectives"]),
            ),
            query_structure=FeaturePipeline(
                extractor=LQPExtractor(positional_encoding_size=None),
                preprocessors=[NormalizePreprocessor(MinMaxScaler(), "graph_features")],
            ),
        )

    return data_processor, df, index_splits


def extract_operations_from_serialized_json(
    plan_df: pd.DataFrame, operation_processing: Callable[[str], str] = lambda x: x
) -> Tuple[Dict[int, List[int]], List[str]]:
    df = plan_df[["id", "lqp"]].copy()
    df["lqp"] = df["lqp"].apply(
        lambda lqp_str: [
            operation_processing(op["predicate"])
            for op_id, op in JsonHandler.load_json_from_str(lqp_str)[
                "operators"
            ].items()
        ]  # type: ignore
    )
    df = df.explode("lqp", ignore_index=True)
    df.rename(columns={"lqp": "operation"}, inplace=True)
    return build_unique_operations(df)


def extract_query_plan_features_from_serialized_json(
    lqp_str: str,
) -> Tuple[QueryPlanStructure, QueryPlanOperationFeatures]:
    lqp = JsonHandler.load_json_from_str(lqp_str)
    operators, links = lqp["operators"], lqp["links"]
    num_operators = len(operators)
    id2name = {
        int(op_id): op["className"].split(".")[-1] for op_id, op in operators.items()
    }
    incoming_ids: List[int] = []
    outgoing_ids: List[int] = []
    for link in links:
        from_id, to_id = link["fromId"], link["toId"]
        if link["fromName"] != id2name[from_id] or link["toName"] != id2name[to_id]:
            raise OperatorMisMatchError
        incoming_ids.append(from_id)
        outgoing_ids.append(to_id)
    node_names = [id2name[i] for i in range(num_operators)]
    sizes = [
        np.log(
            np.clip(
                operators[str(i)]["sizeInBytes"] / 1024.0 / 1024.0,
                a_min=EPS,
                a_max=None,
            )
        )
        for i in range(num_operators)
    ]
    rows_counts = [
        np.log(np.clip(operators[str(i)]["rowCount"] * 1.0, a_min=EPS, a_max=None))
        for i in range(num_operators)
    ]
    op_features = QueryPlanOperationFeatures(rows_count=rows_counts, size=sizes)
    structure = QueryPlanStructure(
        node_names=node_names, incoming_ids=incoming_ids, outgoing_ids=outgoing_ids
    )
    return structure, op_features


class LQPExtractor(QueryStructureExtractor):
    def __init__(self, positional_encoding_size: Optional[int] = None):
        super(LQPExtractor, self).__init__(positional_encoding_size)

    def _extract_structure_and_features(
        self, idx: str, lqp: str, split: DatasetType
    ) -> Dict:
        structure, op_features = extract_query_plan_features_from_serialized_json(lqp)
        operation_gids = self._extract_operation_types(structure, split)
        self.id_template_dict[idx] = self._extract_structure_template(structure, split)
        return {
            "operation_id": op_features.operation_ids,
            "operation_gid": operation_gids,
            **op_features.features_dict,
        }

    def extract_features(
        self, df: pd.DataFrame, split: DatasetType
    ) -> QueryStructureContainer:
        df_op_features: pd.DataFrame = df.apply(
            lambda row: self._extract_structure_and_features(row.id, row.lqp, split),
            axis=1,
        ).apply(pd.Series)
        df_op_features["plan_id"] = df["id"]
        (
            df_op_features_exploded,
            df_operation_types,
        ) = self._extract_op_features_exploded(df_op_features)
        return QueryStructureContainer(
            graph_features=df_op_features_exploded,
            template_plans=self.template_plans,
            key_to_template=self.id_template_dict,
            graph_meta_features=None,
            operation_types=df_operation_types,
        )

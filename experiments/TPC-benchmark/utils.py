from typing import Optional, Dict, Tuple, List, Callable

from udao.data import QueryStructureExtractor, QueryStructureContainer
from udao.data.predicate_embedders.utils import build_unique_operations
from udao.data.utils.query_plan import QueryPlanStructure, QueryPlanOperationFeatures
from udao.data.utils.utils import DatasetType

import numpy as np
import pandas as pd

from udao_trace.configuration import SparkConf
from udao_trace.utils import JsonHandler, BenchmarkType
from udao_trace.workload import Benchmark

ALPHA_LQP_RAW = [
    'IM-inputSizeInBytes',
    'IM-inputRowCount',
]
ALPHA_QS_RAW = ["InitialPartitionNum"] + ALPHA_LQP_RAW
BETA_RAW = ['PD']

THETA = [
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
ALPHA_LQP = [
    "IM-sizeInMB",
    "IM-rowCount",
    "IM-sizeInMB-log",
    "IM-rowCount-log",
]
ALPHA_QS = [
    "IM-init-part-num",
    "IM-init-part-num-log"
] + ALPHA_LQP
BETA = [
    'PD-std-avg',
    'PD-skewness-ratio',
    'PD-range-avg-ratio'
]
GAMMA = [
    'SS-RunningTasksNum',
    'SS-FinishedTasksNum',
    'SS-FinishedTasksTotalTimeInMs',
    'SS-FinishedTasksDistributionInMs-0tile',
    'SS-FinishedTasksDistributionInMs-25tile',
    'SS-FinishedTasksDistributionInMs-50tile',
    'SS-FinishedTasksDistributionInMs-75tile',
    'SS-FinishedTasksDistributionInMs-100tile',
]

EPS=1e-3

class NoBenchmarkError(ValueError):
    """raise when no valid benchmark is found"""

class NoQTypeError(ValueError):
    """raise when no valid mode is found (only q and qs)"""

def _im_process(df: pd.DataFrame) -> pd.DataFrame:
    df["IM-sizeInMB"] = df["IM-inputSizeInBytes"] / 1024 / 1024
    df["IM-sizeInMB-log"] = np.log(df["IM-sizeInMB"].values.clip(min=EPS))
    df["IM-rowCount"] = df["IM-inputRowCount"]
    df["IM-rowCount-log"] = np.log(df["IM-rowCount"].values.clip(min=EPS))
    for c in ALPHA_LQP_RAW:
        del df[c]
    return df

def prepare(df: pd.DataFrame, benchmark: str, knob_meta_file: str, mode: str) -> [pd.DataFrame, List[str]]:
    bm = Benchmark(benchmark_type = BenchmarkType[benchmark.upper()])
    sc = SparkConf(knob_meta_file)
    df.rename(columns={p: kid for p, kid in zip(THETA, sc.knob_ids)}, inplace=True)
    df.rename(columns={"appid": "id"}, inplace=True)
    df["tid"] = df["template"].apply(lambda x: bm.get_template_id(str(x)))
    variable_names = sc.knob_ids
    df[variable_names] = sc.deconstruct_configuration(df[variable_names].astype(str).values)

    # extract alpha
    if mode == "q":
        df[ALPHA_LQP_RAW] = df[ALPHA_LQP_RAW].astype(float)
        df = _im_process(df)
    elif mode == "qs":
        df[ALPHA_QS_RAW] = df[ALPHA_QS_RAW].astype(float)
        df = _im_process(df)
        df["IM-init-part-num"] = df["InitialPartitionNum"].astype(float)
        df["IM-init-part-num-log"] = np.log(df["IM-init-part-num"].values.clip(min=EPS))
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

    return df, variable_names

def extract_operations_from_seralized_json(
        plan_df: pd.DataFrame,
        operation_processing: Callable[[str], str] = lambda x: x
) -> Tuple[Dict[int, List[int]], List[str]]:
    df = plan_df[["id", "lqp"]].copy()
    df["lqp"] = df["lqp"].apply(
        lambda lqp_str: [
            operation_processing(op["predicate"])
            for op_id, op in JsonHandler.load_json_from_str(lqp_str)["operators"].items()
        ]  # type: ignore
    )
    df = df.explode("lqp", ignore_index=True)
    df.rename(columns={"lqp": "operation"}, inplace=True)
    return build_unique_operations(df)

def extract_query_plan_features_from_seralized_json(lqp_str: str) -> Tuple[QueryPlanStructure, QueryPlanOperationFeatures]:
    lqp = JsonHandler.load_json_from_str(lqp_str)
    operators, links = lqp["operators"], lqp["links"]
    num_operators = len(operators)

    id2name = {}
    incoming_ids: List[int] = []
    outgoing_ids: List[int] = []
    for link in links:
        from_id, to_id = link["fromId"], link["toId"]
        if from_id not in id2name:
            id2name[from_id] = link["fromName"]
        if to_id not in id2name:
            id2name[to_id] = link["toName"]
        incoming_ids.append(from_id)
        outgoing_ids.append(to_id)
    assert(len(id2name) == len(set(incoming_ids) | set(outgoing_ids)))
    node_names = [id2name[i] for i in range(num_operators)]
    try:
        sizes = [np.log(np.clip(operators[str(i)]["sizeInBytes"] / 1024. / 1024., a_min=EPS, a_max=None)) for i in range(num_operators)]
        rows_counts = [np.log(np.clip(operators[str(i)]["rowCount"] * 1.0, a_min=EPS, a_max=None)) for i in range(num_operators)]
    except:
        print([np.clip(operators[str(i)]["sizeInBytes"], a_min=EPS, a_max=None) for i in range(num_operators)])
        raise Exception()
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
        structure, op_features = extract_query_plan_features_from_seralized_json(lqp)
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
        df_op_features_exploded, df_operation_types = self._extract_op_features_exploded(df_op_features)
        return QueryStructureContainer(
            graph_features=df_op_features_exploded,
            template_plans=self.template_plans,
            key_to_template=self.id_template_dict,
            graph_meta_features=None,
            operation_types=df_operation_types
        )

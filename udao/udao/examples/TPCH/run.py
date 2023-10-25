from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ...data.embedders import Word2VecEmbedder
from ...data.extractors import PredicateEmbeddingExtractor, QueryStructureExtractor
from ...data.extractors.tabular_extractor import TabularFeatureExtractor
from ...data.handler.data_handler import (
    DataHandler,
    FeaturePipeline,
    create_data_handler_params,
)
from ...data.iterators import QueryPlanIterator
from ...data.preprocessors.normalize_preprocessor import NormalizePreprocessor
from ...utils.logging import logger

if __name__ == "__main__":
    params_getter = create_data_handler_params(QueryPlanIterator, "op_emb")
    params = params_getter(
        index_column="id",
        stratify_on="tid",
        dryrun=True,
        tabular_features=FeaturePipeline(
            extractor=(
                TabularFeatureExtractor,
                [
                    lambda df: df[
                        ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "s1", "s2"]
                    ]
                ],
            ),
            preprocessors=None,
        ),
        objectives=FeaturePipeline(
            extractor=(TabularFeatureExtractor, [lambda df: df[["latency"]]]),
            preprocessors=None,
        ),
        query_structure=FeaturePipeline(
            extractor=(QueryStructureExtractor, []),
            preprocessors=[(NormalizePreprocessor, [MinMaxScaler(), "graph_features"])],
        ),
        op_emb=FeaturePipeline(
            extractor=(PredicateEmbeddingExtractor, [Word2VecEmbedder()]),
            preprocessors=None,
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
    logger.info(split_iterators["train"][0])

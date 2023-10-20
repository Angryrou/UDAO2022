from typing import Dict, Sequence

import dgl
import pandas as pd
import torch as th

from .dataset import BaseDatasetIterator, DataHandlerParams
from .utils.embedding_utils import QueryEmbeddingExtractor, Word2VecEmbedder
from .utils.query_plan_utils import QueryPlanStructure, QueryStructureExtractor


class QueryPlanIterator(BaseDatasetIterator):
    """
    Iterator that returns a dgl.DGLGraph for each key.

    Parameters
    ----------
    keys : Sequence[str]
        Keys of the dataset, used for accessing all features
    graph_features : pd.DataFrame
        Operation features (typically rows_count, size).
        MultiIndex (plan, operation)
    embeddings : pd.DataFrame
        Embeddings for each operation.
        MultiIndex (plan, operation)
    template_plans : Dict[int, QueryPlanStructure]
        Link a template id to a QueryPlanStructure
    key_to_template : Dict[str, int]
        Link a key to a template id.
    """

    def __init__(
        self,
        keys: Sequence[str],
        graph_features: pd.DataFrame,
        embeddings: pd.DataFrame,
        template_plans: Dict[int, QueryPlanStructure],
        key_to_template: Dict[str, int],
    ):
        self.keys = keys
        self.key_to_template = key_to_template
        self.graph_features = graph_features
        self.embeddings = embeddings
        self.template_plans = template_plans

    def __len__(self) -> int:
        return len(self.keys)

    def _get_graph(self, key: str) -> dgl.DGLGraph:
        """Returns the graph corresponding to the key,
        associated with features as th.tensor"""
        graph = self.template_plans[self.key_to_template[key]].graph.clone()
        graph.ndata["cbo"] = th.tensor(self.graph_features.loc[key].values)
        graph.ndata["op_encs"] = th.tensor(self.embeddings.loc[key].values)
        return graph

    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        key = self.keys[idx]
        return self._get_graph(key)


queryPlanDataHandlerParams = DataHandlerParams(
    index_column="id",
    feature_extractors=[
        (QueryStructureExtractor, []),
        (QueryEmbeddingExtractor, [Word2VecEmbedder()]),
    ],
    Iterator=QueryPlanIterator,
    stratify_on="tid",
)

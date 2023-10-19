from typing import Dict, Sequence

import dgl
import pandas as pd
import torch as th
from udao.data.dataset import BaseDatasetIterator, DataHandlerParams
from udao.data.utils.embedding_utils import QueryEmbeddingExtractor, Word2VecEmbedder
from udao.data.utils.query_plan_utils import QueryPlanStructure, QueryStructureExtractor


class QueryPlanIterator(BaseDatasetIterator):
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
        return self.template_plans[self.key_to_template[key]].graph.clone()

    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        key = self.keys[idx]
        graph = self._get_graph(key)
        graph.ndata["cbo"] = th.tensor(self.graph_features.loc[key].values)
        graph.ndata["op_encs"] = th.tensor(self.embeddings.loc[key].values)
        return graph


queryPlanDataHandlerParams = DataHandlerParams(
    index_column="id",
    feature_extractors=[
        (QueryStructureExtractor, []),
        (QueryEmbeddingExtractor, [Word2VecEmbedder()]),
    ],
    Iterator=QueryPlanIterator,
    stratify_on="tid",
)

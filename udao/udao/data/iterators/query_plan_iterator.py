from typing import Sequence

import dgl
import torch as th
from udao.data.containers.query_embedding_container import DataFrameContainer
from udao.data.containers.query_structure_container import QueryStructureContainer

from .base_iterator import BaseDatasetIterator


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
        query_structure_container: QueryStructureContainer,
        query_embeddings_container: DataFrameContainer,
    ):
        self.keys = keys
        self.query_structure_container = query_structure_container
        self.query_embeddings_container = query_embeddings_container

    def __len__(self) -> int:
        return len(self.keys)

    def _get_graph(self, key: str) -> dgl.DGLGraph:
        """Returns the graph corresponding to the key,
        associated with features as th.tensor"""
        graph, graph_features = self.query_structure_container.get(key)
        embeddings = self.query_embeddings_container.get(key)
        graph.ndata["cbo"] = th.tensor(graph_features)
        graph.ndata["op_encs"] = th.tensor(embeddings)
        return graph

    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        key = self.keys[idx]
        return self._get_graph(key)


"""
queryPlanDataHandlerParams = DataHandlerParams(
    index_column="id",
    feature_extractors={
        "query_structure_extractor": (QueryStructureExtractor, []),
        "query_embedding_extractor": (QueryEmbeddingExtractor, [Word2VecEmbedder()]),
    },
    Iterator=QueryPlanIterator,
    stratify_on="tid",
)
"""

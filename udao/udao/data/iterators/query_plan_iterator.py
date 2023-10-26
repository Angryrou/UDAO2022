from dataclasses import dataclass
from typing import List, Sequence, Tuple

import dgl
import torch as th

from ...data.containers.tabular_container import TabularContainer
from ..containers import QueryStructureContainer
from .base_iterator import BaseDatasetIterator


class QueryPlanIterator(BaseDatasetIterator):
    """
    Iterator that returns a dgl.DGLGraph for each key, with associated node features.
    The features are stored in the graph.ndata dictionary.
    The features are expected to be float tensors, and to be of the same length
    as the number of nodes in the graph.

    Parameters
    ----------
    keys : Sequence[str]
        Keys of the dataset, used for accessing all features
    tabular_features : TabularContainer
        Container for the tabular features associated with the plan
    objectives : TabularContainer
        Container for the objectives associated with the plan
    query_structure : QueryStructureContainer
        Wrapper around the graph structure and the features for each query plan
    kwargs: BaseContainer
        Variable number of other features to add to the graph, e.g. embeddings
    """

    @dataclass
    class FeatureItem:
        """Named tuple for the features of a query plan."""

        graph: dgl.DGLGraph
        features: th.Tensor
        objectives: th.Tensor

    def __init__(
        self,
        keys: Sequence[str],
        tabular_features: TabularContainer,
        objectives: TabularContainer,
        query_structure: QueryStructureContainer,
        **kwargs: TabularContainer,
    ):
        self.keys = keys
        self.tabular_features = tabular_features
        self.objectives = objectives
        self.query_structure_container = query_structure
        self.other_graph_features = kwargs

    def __len__(self) -> int:
        return len(self.keys)

    def _get_graph(self, key: str) -> dgl.DGLGraph:
        """Returns the graph corresponding to the key,
        associated with features as th.tensor"""
        graph, graph_features = self.query_structure_container.get(key)
        graph.ndata["cbo"] = th.tensor(graph_features)
        for feature, container in self.other_graph_features.items():
            graph.ndata[feature] = th.tensor(container.get(key))
        return graph

    def __getitem__(self, idx: int) -> FeatureItem:
        key = self.keys[idx]
        features = th.tensor(self.tabular_features.get(key))
        objectives = th.tensor(self.objectives.get(key))
        graph = self._get_graph(key)
        return QueryPlanIterator.FeatureItem(
            graph=graph, features=features, objectives=objectives
        )

    @staticmethod
    def collate(items: List[FeatureItem]) -> Tuple[dgl.DGLGraph, th.Tensor, th.Tensor]:
        """Collate a list of FeatureItem into a single graph."""
        graphs = [item.graph for item in items]
        features = th.cat([item.features for item in items], dim=0)
        objectives = th.cat([item.objectives for item in items], dim=0)
        return dgl.batch(graphs), features, objectives

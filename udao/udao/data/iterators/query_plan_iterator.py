from dataclasses import dataclass
from typing import List, Sequence, Tuple

import dgl
import torch as th

from ...data.containers.tabular_container import TabularContainer
from ...utils.interfaces import BaseUdaoInput
from ..containers import QueryStructureContainer
from .base_iterator import BaseDatasetIterator


@dataclass
class QueryPlanInput(BaseUdaoInput[dgl.DGLGraph]):
    """The embedding input is a dgl.DGLGraph"""

    pass


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

    @staticmethod
    def _get_meta(g: dgl.DGLGraph) -> th.Tensor:
        """Returns the meta information of the graph,
        defined as the sum of the cbo features of the input nodes.
        """
        input_nodes_index = th.where(g.in_degrees() == 0)[0]
        input_meta = g.ndata["cbo"][input_nodes_index].sum(0)
        return input_meta

    def _get_graph_and_meta(self, key: str) -> Tuple[dgl.DGLGraph, th.Tensor]:
        """Returns the graph corresponding to the key,
        associated with features as th.tensor,
        and the meta information.
        """
        graph, graph_features = self.query_structure_container.get(key)
        graph.ndata["cbo"] = th.tensor(graph_features)
        for feature, container in self.other_graph_features.items():
            graph.ndata[feature] = th.tensor(container.get(key))
        return graph, self._get_meta(graph)

    def __getitem__(self, idx: int) -> Tuple[QueryPlanInput, th.Tensor]:
        key = self.keys[idx]
        features = th.tensor(self.tabular_features.get(key))
        objectives = th.tensor(self.objectives.get(key))
        graph, meta_input = self._get_graph_and_meta(key)
        features = th.cat([features, meta_input])
        input_data = QueryPlanInput(graph, features)
        return input_data, objectives

    @staticmethod
    def collate(
        items: List[Tuple[QueryPlanInput, th.Tensor]],
    ) -> Tuple[QueryPlanInput, th.Tensor]:
        """Collate a list of FeatureItem into a single graph."""
        graphs = [item[0].embedding_input for item in items]
        features = th.cat([item[0].feature_input for item in items], dim=0)
        objectives = th.cat([item[1] for item in items], dim=0)
        return QueryPlanInput(dgl.batch(graphs), features), objectives
